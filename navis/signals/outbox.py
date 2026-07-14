"""Signal outbox relay — publishes staged pointer events to the bus at-least-once.

The ingestion path writes a `signal_outbox` row in the SAME transaction as the signal
(see SignalIngestionService). This relay drains pending rows (published_at IS NULL),
publishes each to the event bus, then marks them published. A crash between publish and
mark simply re-publishes next pass — consumers dedupe by the `event_id` carried in the
payload (idempotent, at-least-once). Mirrors navis.execution.worker's outbox relay.

Wired as a Celery task (registered via navis.registry) + a beat schedule entry.
"""
from __future__ import annotations

from typing import Any

from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("signals.outbox")


async def relay_signals_pass(repo: Any, *, limit: int = 500) -> int:
    """Publish persisted pointer events, then mark them published. Returns the count
    published. The whole pass runs in one transaction (the caller's session_scope), so the
    FOR-UPDATE-SKIP-LOCKED claim and the mark commit together."""
    rows = await repo.pending_outbox(limit=limit)
    published_ids = []
    for row in rows:
        try:
            etype = EventType(row.event_type)
        except ValueError:
            # Poison row (event-type enum drift across deploys): left pending it would sit
            # at the front of every oldest-first batch and wedge the signal->memory pointer
            # relay for EVERY org. Drain it (mark published) with a loud error; the row
            # stays for audit. Transient publish failures keep abort-and-retry semantics.
            logger.error("unpublishable signal outbox row %s type=%r — drained", row.id, row.event_type)
            metrics.incr("signals.outbox.poisoned")
            published_ids.append(row.id)
            continue
        await event_bus.publish(
            DomainEvent(
                type=etype,
                org_id=row.organization_id,
                source="signals",
                payload=row.payload,  # pointer + lineage; event_id inside for dedupe
            )
        )
        published_ids.append(row.id)
    await repo.mark_outbox_published(published_ids)
    if published_ids:
        metrics.incr("signals.outbox.published", value=len(published_ids))
        logger.info("signal outbox relay published=%d", len(published_ids))
    return len(published_ids)


async def _relay() -> int:
    from navis.platform.database import session_scope
    from navis.signals.repository import SignalRepository

    async with session_scope() as db:
        return await relay_signals_pass(SignalRepository(db))


async def _purge() -> int:
    """Delete published pointer rows past the retention window (bounds outbox growth)."""
    from datetime import datetime, timedelta, timezone

    from navis.platform.config import settings
    from navis.platform.database import session_scope
    from navis.signals.repository import SignalRepository

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.signals_outbox_retention_seconds)
    async with session_scope() as db:
        purged = await SignalRepository(db).purge_published_outbox(before=cutoff)
    if purged:
        logger.info("signal outbox purged published rows=%d", purged)
    return purged


def _register_tasks():
    from navis.platform.queue import celery_app, run_async

    @celery_app.task(name="navis.signals.outbox.relay_signal_outbox")
    def relay_signal_outbox() -> int:
        return run_async(_relay())

    @celery_app.task(name="navis.signals.outbox.purge_signal_outbox")
    def purge_signal_outbox() -> int:
        return run_async(_purge())

    return relay_signal_outbox, purge_signal_outbox


relay_signal_outbox, purge_signal_outbox = _register_tasks()
