"""Signal outbox relay — publishes staged pointer events to the bus at-least-once.

The ingestion path writes a `signal_outbox` row in the SAME transaction as the signal
(see SignalIngestionService). This relay drains pending rows (published_at IS NULL),
publishes each to the event bus, then marks them published. A crash between publish and
mark simply re-publishes next pass — consumers dedupe by the `event_id` carried in the
payload (idempotent, at-least-once). Mirrors navis.execution.worker's outbox relay.

Wired as a Celery task (registered via navis.registry) + a beat schedule entry.
"""
from __future__ import annotations

import time
from typing import Any

from navis.platform.composition import ensure_subscribers
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("signals.outbox")

# Wall-clock budget for one relay pass, well under celery_app's 300s soft /
# 360s hard task limit. Delivery is synchronous — each row runs its subscribers
# inline (memory extraction parses the full email body), so a large backlog can
# easily outrun a fixed row count. Without a deadline, a pass that claims 500
# rows gets hard-killed mid-flight, the whole transaction rolls back, and the
# next tick claims the same rows: a backlog that can never drain. Stopping early
# commits the work already done and leaves the rest for the next tick.
RELAY_TIME_BUDGET_SECONDS = 120.0


async def relay_signals_pass(
    repo: Any, *, limit: int = 500, time_budget: float = RELAY_TIME_BUDGET_SECONDS
) -> int:
    """Publish persisted pointer events, then mark the SUCCESSFULLY HANDLED ones
    published. Returns the count published. The whole pass runs in one transaction
    (the caller's session_scope), so the FOR-UPDATE-SKIP-LOCKED claim and the mark
    commit together.

    Delivery semantics: `published_at` means "every subscriber that was supposed to
    run for this event ran without raising" — NOT merely "publish() was called".
    A row whose handler failed (e.g. MemoryIngestionService hit a DB timeout) stays
    unpublished and is retried on the next pass; without that, a transient error
    permanently dropped the email's memory extraction with only a best-effort Redis
    dead-letter behind it.

    Registering subscribers is a precondition, not an assumption: publishing into a
    process whose composition root never loaded finds zero handlers and would mark
    every row published while doing nothing at all.
    """
    ensure_subscribers()
    rows = await repo.pending_outbox(limit=limit)
    published_ids = []
    failed = 0
    deadline = time.monotonic() + time_budget
    stopped_early = 0
    for index, row in enumerate(rows):
        if time.monotonic() >= deadline:
            # Commit what's done; the unclaimed remainder is picked up next tick.
            stopped_early = len(rows) - index
            break
        try:
            event_type = EventType(row.event_type)
        except ValueError:
            # Unknown event_type — a deploy removed an EventType. Leave the row
            # unpublished + loud rather than silently swallowing the pointer.
            failed += 1
            logger.exception(
                "signal outbox skipped row=%s: unknown event_type=%s", row.id, row.event_type
            )
            continue
        delivery = await event_bus.publish(
            DomainEvent(
                type=event_type,
                org_id=row.organization_id,
                source="signals",
                payload=row.payload,  # pointer + lineage; event_id inside for dedupe
            )
        )
        if delivery.ok:
            published_ids.append(row.id)
        else:
            # Already logged + dead-lettered by the bus. Staying unpublished is what
            # makes it retryable; re-delivery is safe because memory append is
            # idempotent on the deterministic memory_id.
            failed += 1
            logger.warning(
                "signal outbox row=%s retained for retry: %d/%d handler(s) failed on %s",
                row.id, delivery.failed, delivery.handlers, row.event_type,
            )
    await repo.mark_outbox_published(published_ids)
    if published_ids:
        metrics.incr("signals.outbox.published", value=len(published_ids))
    if failed:
        metrics.incr("signals.outbox.retained", value=failed)
    if stopped_early:
        metrics.incr("signals.outbox.deferred", value=stopped_early)
        logger.info(
            "signal outbox relay hit its %.0fs budget; deferred=%d to next pass",
            time_budget, stopped_early,
        )
    if published_ids or failed:
        logger.info(
            "signal outbox relay published=%d retained=%d", len(published_ids), failed
        )
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
