"""ReplayService — re-emit stored signals from the append-only log.

Because the log is the source of truth and bus events are pointers, replay is
just: read rows for a tenant/source/time-range, re-publish pointer events. Used
to recover a downstream consumer that lost data, or to reprocess after a bug fix.
Replay is idempotent downstream (event_id is stable).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from navis.platform.config import settings
from navis.platform.events import DomainEvent, event_bus
from navis.platform.monitoring import get_logger
from navis.signals.base import SignalEvent, SignalSource
from navis.signals.repository import DeadLetterRepository, SignalRepository
from navis.signals.service import SignalIngestionService, bus_event_for_type_value

logger = get_logger("signals.replay")


class ReplayService:
    def __init__(self, db: AsyncSession) -> None:
        self.repo = SignalRepository(db)
        self.dlq_repo = DeadLetterRepository(db)
        self.ingestion = SignalIngestionService(db)

    async def replay(
        self,
        org_id: uuid.UUID,
        *,
        source: SignalSource | None = None,
        since: datetime | None = None,
        limit: int | None = 1000,
        batch_size: int = 1000,
    ) -> int:
        """Re-emit stored signals in a STABLE total order, PAGINATING the full backlog
        (not just the first batch) so a tenant's reality can be rebuilt from zero.

        `limit` caps the total replayed (a safety bound against an accidental fleet-wide
        flood); pass None for an unbounded full replay (e.g. a deliberate Memory rebuild).
        Pagination is keyset (occurred_at, id) — O(batch) memory, deterministic, resumable;
        the per-event event_id is stable, so downstream replay is idempotent."""
        replayed = 0
        async for rec in self.repo.stream_replay(
            org_id, source=source, since=since, batch_size=batch_size
        ):
            if limit is not None and replayed >= limit:
                break
            # Route by the stored event_type (event-type routing fix) — replay a
            # calendar.updated as CalendarEventUpdated, not CalendarEventCreated.
            await event_bus.publish(
                DomainEvent(
                    type=bus_event_for_type_value(rec.event_type),
                    org_id=rec.organization_id,
                    source=rec.source,
                    payload={
                        "event_id": str(rec.id),
                        "source": rec.source,
                        "event_type": rec.event_type,
                        "external_id": rec.external_id,
                        "external_version": rec.external_version,
                        "user_id": str(rec.user_id) if rec.user_id else None,
                        "integration_id": str(rec.integration_id) if rec.integration_id else None,
                        # Capture-origin lineage: this event came from a replay, not live.
                        "origin": "replay",
                        "replay": True,
                    },
                )
            )
            replayed += 1
        logger.info("replayed %d signals org=%s source=%s", replayed, org_id, source)
        return replayed

    async def replay_dead_letters(
        self, org_id: uuid.UUID, *, limit: int = 100, max_attempts: int | None = None
    ) -> dict:
        """Re-attempt unresolved poison signals from PostgreSQL (the DLQ source of truth).
        Successful re-ingestion resolves the row. A FAILED attempt bumps `attempts` and
        records `last_error`; after `max_attempts` the row is permanently dead-lettered
        (resolved) so it leaves the retry set — the POISON-LOOP GUARD (without it a
        permanently-poison item is re-attempted every pass forever)."""
        max_attempts = max_attempts or settings.signals_dlq_max_replay_attempts
        rows = await self.dlq_repo.list_unresolved(org_id, limit=limit)
        recovered: list[uuid.UUID] = []
        still_failing = exhausted = 0
        for row in rows:
            payload = row.payload or {}
            if "event_id" not in payload:  # fetch-stage failures aren't replayable as events
                continue
            try:
                event = SignalEvent.from_dict(payload)
                await self.ingestion.ingest(event)
                recovered.append(row.id)
            except Exception as exc:  # account the attempt; permanently dead-letter at max
                gave_up = await self.dlq_repo.record_replay_failure(
                    row.id, error=repr(exc), max_attempts=max_attempts
                )
                if gave_up:
                    exhausted += 1
                    logger.error("DLQ poison permanently failed dlq_id=%s attempts=%d", row.id, max_attempts)
                else:
                    still_failing += 1
                    logger.warning("DLQ replay still failing dlq_id=%s", row.id)
        await self.dlq_repo.mark_replayed(recovered, resolved=True)
        logger.info(
            "DLQ replay org=%s recovered=%d failing=%d exhausted=%d",
            org_id, len(recovered), still_failing, exhausted,
        )
        return {"recovered": len(recovered), "still_failing": still_failing, "exhausted": exhausted}
