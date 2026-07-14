"""Memory outbox relay — publishes staged memory-derived domain events at-least-once.

MemoryIngestionService writes a `memory_outbox` row in the SAME transaction as the
memory_records append (see service.py). This relay drains pending rows (published_at IS
NULL), publishes each to the event bus, then marks them published. A crash between publish
and mark simply re-publishes on the next pass — consumers dedupe by memory_id carried in
the payload (idempotent, at-least-once). Mirrors navis.signals.outbox.

Wired as a Celery task (registered via navis.registry) + a beat schedule entry.
"""
from __future__ import annotations

import asyncio
from typing import Any

from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("memory.outbox")


async def relay_memory_pass(repo: Any, *, limit: int = 500) -> int:
    """Publish persisted memory pointer events, then mark them published. Returns the count
    published. The whole pass runs in one transaction (the caller's session_scope), so the
    FOR-UPDATE-SKIP-LOCKED claim and the mark commit together."""
    rows = await repo.pending_outbox(limit=limit)
    published_ids = []
    for row in rows:
        try:
            await event_bus.publish(
                DomainEvent(
                    type=EventType(row.event_type),
                    org_id=row.organization_id,
                    source="memory",
                    payload=row.payload,  # pointer + lineage; memory_id inside for dedupe
                )
            )
        except ValueError:
            # Unknown event_type value — leave the row unpublished + loud; a code deploy
            # that removed an EventType shouldn't silently swallow the pointer.
            logger.exception(
                "memory outbox skipped row=%s: unknown event_type=%s", row.id, row.event_type,
            )
            continue
        published_ids.append(row.id)
    await repo.mark_outbox_published(published_ids)
    if published_ids:
        metrics.incr("memory.outbox.published", value=len(published_ids))
        logger.info("memory outbox relay published=%d", len(published_ids))
    return len(published_ids)


async def _relay() -> int:
    from navis.memory.repository import MemoryRepository
    from navis.platform.database import session_scope

    async with session_scope() as db:
        return await relay_memory_pass(MemoryRepository(db))


async def _purge() -> int:
    """Delete published pointer rows past the retention window (bounds outbox growth)."""
    from datetime import datetime, timedelta, timezone

    from navis.memory.repository import MemoryRepository
    from navis.platform.config import settings
    from navis.platform.database import session_scope

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.memory_outbox_retention_seconds)
    async with session_scope() as db:
        purged = await MemoryRepository(db).purge_published_outbox(before=cutoff)
    if purged:
        logger.info("memory outbox purged published rows=%d", purged)
    return purged


def _register_tasks():
    from navis.platform.queue import celery_app, run_async

    @celery_app.task(name="navis.memory.outbox.relay_memory_outbox")
    def relay_memory_outbox() -> int:
        return run_async(_relay())

    @celery_app.task(name="navis.memory.outbox.purge_memory_outbox")
    def purge_memory_outbox() -> int:
        return run_async(_purge())

    return relay_memory_outbox, purge_memory_outbox


relay_memory_outbox, purge_memory_outbox = _register_tasks()
