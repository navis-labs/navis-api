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

from navis.memory.envelope import MemoryRecord
from navis.memory.extractors import SignalView, extractors_for
from navis.memory.ids import entity_id, memory_record_id
from navis.memory.repository import MemoryRepository
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, subscribe
from navis.platform.monitoring import get_logger
from navis.signals.base import SignalEventType, SignalSource

logger = get_logger("memory")


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
            payload=rec.payload or {},
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
        for record in build_memory_records(view):
            if await self.memory_repo.append(record):
                written += 1
        logger.info("memory: signal %s -> %d new facts", event_id, written)
        return written


# --- Subscriber: react to the signal pointer events --------------------------
# Memory subscribes to the pointer (ids only) and fetches the full SignalEvent.
async def _on_signal_pointer(event: DomainEvent) -> None:
    event_id = (event.payload or {}).get("event_id")
    if not event_id:
        return
    async with session_scope() as db:
        await MemoryIngestionService(db).ingest_signal(uuid.UUID(event_id))


for _evt in (
    EventType.GMAIL_RECEIVED,
    EventType.CALENDAR_EVENT_CREATED,
    EventType.MEETING_COMPLETED,
    EventType.SIGNAL_INGESTED,
):
    subscribe(_evt)(_on_signal_pointer)
