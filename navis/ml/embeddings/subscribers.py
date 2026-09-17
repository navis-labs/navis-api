"""Embedding subscribers — real-time embed memory facts as they land.

Push model: when the memory ingestion pipeline writes a MemoryRecord and stages
its pointer event (PERSON_DETECTED / COMMITMENT_DETECTED / ...), this subscriber
picks it up off the bus and embeds that one record immediately. The moment a
Gmail message has extracted into memory, it is also searchable by /chat — no
5-minute beat lag.

Idempotency: `compute_embedding_id(memory_id, model, version)` is deterministic
and the insert uses ON CONFLICT DO NOTHING (see EmbeddingService.embed_record),
so at-least-once event delivery + replay collapse to the same row. Re-firing
the same event costs one wasted embed call and one no-op INSERT — no
duplicates, no races.

Coverage gap (deliberate, matches existing pointer wiring):
`navis.memory.service._BUS_EVENT_FOR_ENTITY` only routes PERSON / PROJECT /
COMMITMENT / DECISION / RELATIONSHIP to the bus. Other entity types (COMPANY,
CUSTOMER, MEETING, THREAD, CHANNEL, DOCUMENT) are memory-only today, so they
will NOT be embedded by this subscriber. Backfill / sweeper via
`navis.ml.embeddings.jobs.embed_org` is the mechanism that catches them (and
also any record whose subscriber invocation dead-letters).

That sweeper is the load-bearing half of this design and it now RUNS: it is on
the Celery beat as `embed-memory-backlog` (every 5 min). It was not, which is
why the six types above had zero vectors and retrieval could not reach a single
email thread, meeting or document. Adding bus events for them purely to feed
this subscriber would make Memory publish domain events for ML's benefit —
the wrong direction — so coverage stays split: five types instant, the rest
within one sweep.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select

from navis.memory.models import MemoryRecord as MemoryRecordRow
from navis.ml.embeddings.service import EmbeddingService
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, subscribe
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("ml.embeddings.subscribers")


# Every memory-layer event carries the memory_id in its payload (see
# navis.memory.service.MemoryIngestionService._outbox_row). This subscriber
# fans in across all of them so one handler = one embed regardless of type.
_MEMORY_EVENTS = (
    EventType.PERSON_DETECTED,
    EventType.PROJECT_DETECTED,
    EventType.COMMITMENT_DETECTED,
    EventType.DECISION_DETECTED,
    EventType.RELATIONSHIP_UPDATED,
)


async def _embed_memory_pointer(event: DomainEvent) -> None:
    """Fetch the memory record referenced by the pointer event and embed it.

    Guards drop expected empty cases (missing payload key, record erased by
    GDPR cascade) with a log + metric — those must not raise, otherwise every
    tombstoned record would generate a dead-letter. Real failures (provider
    timeout, DB down) DO raise so the bus dead-letters them for replay.
    """
    payload = event.payload or {}
    raw_memory_id = payload.get("memory_id")
    if not raw_memory_id:
        # Pointer without a memory_id is a producer bug — log so it's visible
        # but don't dead-letter; there's nothing to retry.
        logger.warning(
            "embed subscriber: %s missing memory_id (event_id=%s)",
            event.type.value, event.id,
        )
        return
    try:
        memory_id = uuid.UUID(str(raw_memory_id))
    except (TypeError, ValueError):
        logger.warning(
            "embed subscriber: %s bad memory_id=%r (event_id=%s)",
            event.type.value, raw_memory_id, event.id,
        )
        return

    async with session_scope() as db:
        record = await db.scalar(
            select(MemoryRecordRow).where(MemoryRecordRow.memory_id == memory_id)
        )
        if record is None:
            # Record was erased between publish and this handler running
            # (GDPR cascade drops memory_records + embeddings together, so
            # attempting to embed now would just violate the FK).
            metrics.incr("ml.embeddings.subscriber.record_missing")
            logger.info(
                "embed subscriber: memory %s no longer exists; skipping",
                memory_id,
            )
            return

        service = EmbeddingService(db)
        newly_written = await service.embed_record(record)

    if newly_written:
        metrics.incr("ml.embeddings.subscriber.embedded")
    else:
        # ON CONFLICT DO NOTHING — replay or duplicate event. Not an error;
        # tracked separately so we can watch for unexpectedly high replay rates.
        metrics.incr("ml.embeddings.subscriber.duplicate")


for _evt in _MEMORY_EVENTS:
    subscribe(_evt)(_embed_memory_pointer)
