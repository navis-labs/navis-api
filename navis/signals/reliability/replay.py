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

from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger
from navis.signals.base import SignalEvent, SignalSource
from navis.signals.repository import DeadLetterRepository, SignalRepository
from navis.signals.service import _BUS_EVENT_FOR_SOURCE, SignalIngestionService

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
        limit: int = 1000,
    ) -> int:
        records = await self.repo.for_replay(org_id, source=source, since=since, limit=limit)
        for rec in records:
            src = SignalSource(rec.source)
            bus_type = _BUS_EVENT_FOR_SOURCE.get(src, EventType.SIGNAL_INGESTED)
            await event_bus.publish(
                DomainEvent(
                    type=bus_type,
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
                        "replay": True,
                    },
                )
            )
        logger.info("replayed %d signals org=%s source=%s", len(records), org_id, source)
        return len(records)

    async def replay_dead_letters(self, org_id: uuid.UUID, *, limit: int = 100) -> dict:
        """Re-attempt unresolved poison signals from PostgreSQL (the DLQ source of
        truth). Successful re-ingestion marks the dead letter resolved."""
        rows = await self.dlq_repo.list_unresolved(org_id, limit=limit)
        recovered: list[uuid.UUID] = []
        still_failing = 0
        for row in rows:
            payload = row.payload or {}
            if "event_id" not in payload:  # fetch-stage failures aren't replayable as events
                continue
            try:
                event = SignalEvent.from_dict(payload)
                await self.ingestion.ingest(event)
                recovered.append(row.id)
            except Exception:  # still poison — leave unresolved for the next pass
                still_failing += 1
                logger.warning("DLQ replay still failing dlq_id=%s", row.id)
        await self.dlq_repo.mark_replayed(recovered, resolved=True)
        logger.info("DLQ replay org=%s recovered=%d failing=%d", org_id, len(recovered), still_failing)
        return {"recovered": len(recovered), "still_failing": still_failing}
