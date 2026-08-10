"""Signal ingestion — the single funnel every connector flows through.

Guarantees:
  * idempotent persistence (DB ON CONFLICT) — the durable, replayable log;
  * a lightweight POINTER event for the bus (ids + lineage only, never the body) written
    to the TRANSACTIONAL OUTBOX in the same transaction as the signal insert. A relay
    worker publishes it at-least-once. Consumers fetch the full signal by event_id. This
    is what makes ingestion cheap at 100M signals/day AND crash-safe (no commit-then-
    publish dual write that can drop downstream work).

Signals stop here. What happens after the bus is none of this domain's business.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from navis.platform.events import EventType
from navis.platform.monitoring import get_logger
from navis.signals.base import SignalEvent, SignalEventType
from navis.signals.importance import score_payload
from navis.signals.models import compute_importance_id
from navis.signals.repository import SignalRepository

logger = get_logger("signals")

# Route by the ACTUAL event_type (Issue A fix) — NOT the source, which collapsed
# calendar.created/updated/cancelled into one CalendarEventCreated. Anything without a
# dedicated platform event falls back to SIGNAL_INGESTED; the precise event_type still
# rides in the pointer payload, so no information is lost.
_BUS_EVENT_FOR_TYPE: dict[SignalEventType, EventType] = {
    SignalEventType.EMAIL_RECEIVED: EventType.GMAIL_RECEIVED,
    SignalEventType.CALENDAR_EVENT_CREATED: EventType.CALENDAR_EVENT_CREATED,
    SignalEventType.CALENDAR_EVENT_UPDATED: EventType.CALENDAR_EVENT_UPDATED,
    SignalEventType.CALENDAR_EVENT_CANCELLED: EventType.CALENDAR_EVENT_CANCELLED,
    SignalEventType.MEETING_COMPLETED: EventType.MEETING_COMPLETED,
    # Drive/Docs file events + meeting created/updated have no dedicated consumer yet:
    # generic SIGNAL_INGESTED (precise type preserved in payload["event_type"]).
}


def bus_event_for_type_value(event_type_value: str) -> EventType:
    """Route an event_type STRING (e.g. a stored record on replay) to its bus EventType —
    same table as event_type_for(), tolerant of unknown/legacy values."""
    try:
        signal_type = SignalEventType(event_type_value)
    except ValueError:
        return EventType.SIGNAL_INGESTED
    return _BUS_EVENT_FOR_TYPE.get(signal_type, EventType.SIGNAL_INGESTED)


def event_type_for(event: SignalEvent) -> EventType:
    return bus_event_for_type_value(event.event_type.value)


class SignalIngestionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = SignalRepository(db)

    async def ingest(self, event: SignalEvent) -> bool:
        """Persist + stage the pointer in the SAME transaction. Returns True if newly
        ingested, False if duplicate. The relay worker publishes the pointer — no
        commit-then-publish, so a crash can never lose the downstream notification."""
        inserted = await self.repo.insert_idempotent(event)
        if not inserted:
            logger.debug("duplicate signal %s skipped", event.dedupe_key)
            return False
        await self.repo.append_outbox([self._outbox_row(event)])
        await self.repo.append_importance([self._importance_row(event)])
        await self.db.commit()  # signal row + outbox pointer + score commit atomically
        logger.info(
            "ingested %s/%s ext=%s v=%s org=%s",
            event.source.value, event.event_type.value, event.external_id,
            event.external_version, event.org_id,
        )
        return True

    async def ingest_many(self, events: list[SignalEvent]) -> tuple[int, int]:
        """Batch path: persist a chunk + stage one outbox pointer per NEW event in ONE
        transaction + ONE commit. Returns (inserted, duplicates).

        The 100M-signals/day path — collapses N inserts + N commits into 1 + 1, and the
        outbox makes delivery crash-safe (vs N publishes that could each drop work).

        On a batch INSERT failure (one poison row aborts the statement) it rolls back and
        re-raises so the caller can fall back to per-event `ingest()` for poison isolation."""
        if not events:
            return 0, 0
        try:
            inserted_ids = await self.repo.insert_many_returning_ids(events)
            by_id: dict[uuid.UUID, SignalEvent] = {}
            for e in events:
                by_id.setdefault(e.event_id, e)
            await self.repo.append_outbox([self._outbox_row(by_id[eid]) for eid in inserted_ids])
            await self.repo.append_importance(
                [self._importance_row(by_id[eid]) for eid in inserted_ids]
            )
        except Exception:
            await self.db.rollback()  # clear the aborted txn so the caller can fall back
            raise
        await self.db.commit()  # signals + outbox pointers commit atomically
        return len(inserted_ids), len(events) - len(inserted_ids)

    def _importance_row(self, event: SignalEvent) -> dict[str, Any]:
        """Score the signal once, at ingestion, in the same transaction as the log.

        The connector seals body fields before handing the event over, so the
        payload is decrypted here first — otherwise every content-derived class
        would silently miss and everything would collapse to a header-only score.
        Scoring never raises (see `score_signal`), so a scoring change can never
        block ingestion.
        """
        importance = score_payload(event.source, event.payload)
        return {
            "id": compute_importance_id(event.event_id, importance.version),
            "organization_id": event.org_id,
            "signal_event_id": event.event_id,
            "version": importance.version,
            "score": importance.score,
            "classification": importance.classification,
            "features": importance.features,
        }

    def _outbox_row(self, event: SignalEvent) -> dict[str, Any]:
        return {
            "organization_id": event.org_id,
            "aggregate_id": event.event_id,
            "event_type": event_type_for(event).value,
            "source": event.source.value,
            "payload": self._pointer_payload(event),
        }

    @staticmethod
    def _pointer_payload(event: SignalEvent) -> dict[str, Any]:
        """Pointer + lineage (Issue E fix) — ids only, NEVER the body. Consumers fetch the
        full signal by event_id; the lineage lets them version/trace without a re-read."""
        payload: dict[str, Any] = {
            "event_id": str(event.event_id),
            "source": event.source.value,
            "event_type": event.event_type.value,
            "external_id": event.external_id,
            "external_version": event.external_version,
            "user_id": str(event.user_id),
            "integration_id": str(event.integration_id),
            # --- lineage ---
            "schema_version": event.schema_version,
            "captured_at": (event.captured_at or event.timestamp).isoformat(),
        }
        # Optional lineage a connector/sync may stamp into metadata — surfaced if present,
        # without duplicating any payload body.
        for key in ("connector_version", "sync_run_id"):
            value = event.metadata.get(key)
            if value is not None:
                payload[key] = value
        return payload
