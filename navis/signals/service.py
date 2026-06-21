"""Signal ingestion — the single funnel every connector flows through.

Guarantees:
  * idempotent persistence (DB ON CONFLICT) — the durable, replayable log;
  * a lightweight POINTER event on the bus (ids only, not the payload), so
    ingestion latency is decoupled from downstream processing. Consumers read
    the full signal from the store by event_id. This is what lets ingestion stay
    cheap at 100M signals/day.

Signals stop here. What happens after the bus is none of this domain's business.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger
from navis.signals.base import SignalEvent, SignalSource
from navis.signals.repository import SignalRepository

logger = get_logger("signals")

# Source -> bus event type. Consumers subscribe to these and fetch by event_id.
_BUS_EVENT_FOR_SOURCE = {
    SignalSource.GMAIL: EventType.GMAIL_RECEIVED,
    SignalSource.CALENDAR: EventType.CALENDAR_EVENT_CREATED,
    SignalSource.MEET: EventType.MEETING_COMPLETED,
    SignalSource.DRIVE: EventType.SIGNAL_INGESTED,
    SignalSource.DOCS: EventType.SIGNAL_INGESTED,
    SignalSource.SLACK: EventType.SLACK_MESSAGE_RECEIVED,
}


class SignalIngestionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = SignalRepository(db)

    async def ingest(self, event: SignalEvent) -> bool:
        """Persist + enqueue. Returns True if newly ingested, False if duplicate."""
        inserted = await self.repo.insert_idempotent(event)
        if not inserted:
            logger.debug("duplicate signal %s skipped", event.dedupe_key)
            return False

        # Commit BEFORE announcing: the pointer event tells consumers to fetch the
        # signal by id from the log, so the row must be durably visible to their
        # (separate) session first. Mirrors the signup commit-then-publish pattern;
        # session-level advisory locks survive COMMIT, so single-flight sync is safe.
        await self.db.commit()
        await self._enqueue(event)
        logger.info(
            "ingested %s/%s ext=%s v=%s org=%s",
            event.source.value,
            event.event_type.value,
            event.external_id,
            event.external_version,
            event.org_id,
        )
        return True

    async def _enqueue(self, event: SignalEvent) -> None:
        bus_type = _BUS_EVENT_FOR_SOURCE.get(event.source, EventType.SIGNAL_INGESTED)
        await event_bus.publish(
            DomainEvent(
                type=bus_type,
                org_id=event.org_id,
                source=event.source.value,
                # Pointer only — keeps the stream small and the log the source of truth.
                payload={
                    "event_id": str(event.event_id),
                    "source": event.source.value,
                    "event_type": event.event_type.value,
                    "external_id": event.external_id,
                    "external_version": event.external_version,
                    "user_id": str(event.user_id),
                    "integration_id": str(event.integration_id),
                },
            )
        )
