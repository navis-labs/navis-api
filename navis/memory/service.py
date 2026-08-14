"""Memory V2 ingestion — turn one SignalEvent into immutable, provenanced facts.

Flow (Phase 1 reconnect):
    signal pointer on the bus -> fetch full SignalEvent from signal_events
        -> run_extractors (pure) -> build provenanced MemoryRecords
        -> append (idempotent)

Memory consumes ONLY SignalEvent. It never reads State/Reasoning and computes no
intelligence. `build_memory_records` is pure and deterministic — the heart of
replayability — and is unit-tested without any database.
"""
from __future__ import annotations

import uuid
from typing import Any

from navis.memory.envelope import EntityType, MemoryRecord
from navis.memory.extractors import SignalView, extractors_for
from navis.memory.ids import entity_id, memory_record_id
from navis.memory.repository import MemoryRepository
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, subscribe
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.platform.security import FieldDecryptError, field_cipher
from navis.signals.base import SignalEventType, SignalSource

# Route a newly-written memory fact to the platform bus event that consumers
# subscribe to. Anything unmapped stays out of the outbox (writing an
# unroutable event would produce a permanently-unpublishable row). RELATIONSHIP
# updates use the RELATIONSHIP_UPDATED event that state/reasoning folds on.
_BUS_EVENT_FOR_ENTITY: dict[EntityType, EventType] = {
    EntityType.PERSON: EventType.PERSON_DETECTED,
    EntityType.PROJECT: EventType.PROJECT_DETECTED,
    EntityType.COMMITMENT: EventType.COMMITMENT_DETECTED,
    EntityType.DECISION: EventType.DECISION_DETECTED,
    EntityType.RELATIONSHIP: EventType.RELATIONSHIP_UPDATED,
}

logger = get_logger("memory")


def _decrypt_content(payload: dict) -> dict:
    """Decrypt the content fields a producer sealed at rest (self-described via
    `_content_fields`). Returns a NEW dict — the stored record is never mutated. A field that
    can't be decrypted (key rotated out / tampered) degrades to empty + a loud metric, so one
    unreadable body can't crash ingestion or poison the rest of the extracted facts."""
    fields = payload.get("_content_fields")
    if not fields:
        return payload
    out = dict(payload)
    for f in fields:
        val = out.get(f)
        if isinstance(val, str):
            try:
                out[f] = field_cipher.decrypt(val)
            except FieldDecryptError:
                metrics.incr("memory.content_decrypt_failed")
                logger.error("content field %s failed decryption; degrading to empty", f)
                out[f] = ""
    return out


def build_memory_records(view: SignalView) -> list[MemoryRecord]:
    """PURE: SignalEvent view -> deterministic, provenanced MemoryRecords.

    Iterates extractors in registration order so output is deterministic; each
    record carries the producing extractor's name+version. Same view in →
    identical records out (modulo created_at, which the DB sets).
    """
    records: list[MemoryRecord] = []
    for ext in extractors_for(view.source):
        for draft in ext.extract(view):
            et = draft.entity_type
            records.append(
                MemoryRecord(
                    memory_id=memory_record_id(
                        view.org_id, view.event_id, ext.name, et.value, draft.dedup_key
                    ),
                    org_id=view.org_id,
                    entity_type=et,
                    entity_id=entity_id(view.org_id, et.value, draft.entity_natural_key),
                    source_signal_event_id=view.event_id,
                    source_integration_id=view.integration_id,
                    event_timestamp=view.event_timestamp,
                    extractor_name=ext.name,
                    extractor_version=ext.version,
                    schema_version=draft.schema_version,
                    confidence=draft.confidence,
                    dedup_key=draft.dedup_key,
                    payload=draft.payload,
                )
            )
    return records


class SignalViewFactory:
    """Builds a SignalView from a stored SignalEventRecord (read-only)."""

    @staticmethod
    def from_record(rec) -> SignalView:
        return SignalView(
            event_id=rec.id,
            org_id=rec.organization_id,
            user_id=rec.user_id,
            integration_id=rec.integration_id,
            source=SignalSource(rec.source),
            event_type=SignalEventType(rec.event_type),
            external_id=rec.external_id,
            external_version=rec.external_version,
            event_timestamp=rec.occurred_at,
            payload=_decrypt_content(rec.payload or {}),
            metadata=rec.extra or {},
        )


class MemoryIngestionService:
    """Reads a signal from the log and appends its memory facts. Idempotent."""

    def __init__(self, db, *, memory_repo=None, signal_getter=None) -> None:
        self.memory_repo = memory_repo or MemoryRepository(db)
        self._signal_getter = signal_getter  # async (event_id) -> SignalView | None
        self._db = db

    async def _get_view(self, event_id: uuid.UUID) -> SignalView | None:
        if self._signal_getter is not None:
            return await self._signal_getter(event_id)
        from navis.signals.repository import SignalRepository

        rec = await SignalRepository(self._db).get(event_id)
        return SignalViewFactory.from_record(rec) if rec else None

    async def ingest_signal(self, event_id: uuid.UUID) -> int:
        view = await self._get_view(event_id)
        if view is None:
            logger.warning("memory: signal %s not found in log", event_id)
            return 0
        written = 0
        outbox_rows: list[dict[str, Any]] = []
        for record in build_memory_records(view):
            if not await self.memory_repo.append(record):
                continue
            written += 1
            # Only stage a pointer for entity types that have a bus event downstream
            # consumers subscribe to. Unmapped types (COMPANY, THREAD, CHANNEL,
            # DOCUMENT, MEETING, CUSTOMER) stay memory-only and would produce a
            # permanently-unpublishable row if written here.
            bus_event = _BUS_EVENT_FOR_ENTITY.get(record.entity_type)
            if bus_event is not None:
                outbox_rows.append(self._outbox_row(record, bus_event))
        # Append pointers in the SAME transaction as the memory rows so the
        # signal → memory → domain-event chain commits atomically (commit-before-
        # publish; a crash between write and publish just re-publishes on the
        # next relay pass, and consumers dedupe by memory_id).
        await self.memory_repo.append_outbox(outbox_rows)
        logger.info(
            "memory: signal %s -> %d new facts, %d bus pointers",
            event_id, written, len(outbox_rows),
        )
        return written

    @staticmethod
    def _outbox_row(record: MemoryRecord, event_type: EventType) -> dict[str, Any]:
        # Pointer + lineage — never the raw body. Downstream fetches by memory_id
        # if it needs the fact detail. Mirrors SignalIngestionService._pointer_payload.
        return {
            "organization_id": record.org_id,
            "aggregate_id": record.memory_id,
            "event_type": event_type.value,
            "entity_type": record.entity_type.value,
            "payload": {
                "memory_id": str(record.memory_id),
                "entity_type": record.entity_type.value,
                "entity_id": str(record.entity_id),
                "source_signal_event_id": str(record.source_signal_event_id),
                "extractor_name": record.extractor_name,
                "extractor_version": record.extractor_version,
                "schema_version": record.schema_version,
                "confidence": record.confidence,
                "event_timestamp": record.event_timestamp.isoformat(),
            },
        }


# --- Subscriber: react to the signal pointer events --------------------------
# Memory subscribes to the pointer (ids only) and fetches the full SignalEvent.
async def _on_signal_pointer(event: DomainEvent) -> None:
    event_id = (event.payload or {}).get("event_id")
    if not event_id:
        return
    async with session_scope() as db:
        await MemoryIngestionService(db).ingest_signal(uuid.UUID(event_id))


# Every bus event `signals.service._BUS_EVENT_FOR_TYPE` can route a signal to must
# appear here, or that signal is captured and relayed but never becomes memory.
#
# CALENDAR_EVENT_UPDATED / _CANCELLED were missing. Signals route by their ACTUAL
# event_type, and Google stamps an event UPDATED whenever `createdTime !=
# modifiedTime` — which is the common case — so the canonical Calendar record for a
# real meeting was published to a bus with zero handlers and silently dropped. What
# survived was only the Meet-derived copy (which routes via the SIGNAL_INGESTED
# fallback), carrying no status, no organizer and no join link. Cancellations were
# lost the same way, which is why a cancelled meeting could never reach the
# `status="cancelled"` state the MEETING payload schema already defines.
#
# Re-delivery is safe: memory_id is a deterministic UUIDv5 and inserts are
# ON CONFLICT DO NOTHING, so a replayed pointer converges instead of duplicating.
for _evt in (
    EventType.GMAIL_RECEIVED,
    EventType.CALENDAR_EVENT_CREATED,
    EventType.CALENDAR_EVENT_UPDATED,
    EventType.CALENDAR_EVENT_CANCELLED,
    EventType.MEETING_COMPLETED,
    EventType.SIGNAL_INGESTED,
):
    subscribe(_evt)(_on_signal_pointer)
