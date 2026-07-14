"""Celery app for background work (signal sync, brief generation, observation polling).

Tasks themselves live in their owning domains (e.g. orchestration/, signals/),
never here. This module only configures the broker.
"""
from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

from celery import Celery

from navis.platform.config import settings

_T = TypeVar("_T")

_worker_loop: asyncio.AbstractEventLoop | None = None


def run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    """Drive a task coroutine on the worker's persistent event loop.

    Every Celery task in this repo unwraps an async body — the natural pattern
    is ``asyncio.run(coro)``. But ``asyncio.run`` creates AND closes a fresh
    loop per call, and SQLAlchemy's async engine caches asyncpg connections
    against the loop that first touched them. On task #2, the pooled connection
    points at a closed loop and asyncpg raises "Event loop is closed" from
    ``BaseProtocol._new_waiter``. Windows + Python 3.13 makes this fail loudly.

    Keeping ONE loop for the worker's lifetime means the pool binds to it once
    and stays valid across every task. Lazy init covers --pool=solo, threads,
    and (via first-use) prefork worker children — no signal wiring needed.
    """
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop.run_until_complete(coro)

celery_app = Celery(
    "navis",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # Reliability: redeliver if a worker dies mid-task; reject on hard timeout/loss.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Throughput/fairness: don't let one worker hoard the queue.
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=200,  # guard against slow memory leaks
    # Timeouts so a hung task can't pin a worker forever.
    task_soft_time_limit=300,  # SoftTimeLimitExceeded — task can clean up
    task_time_limit=360,       # hard kill
    # Results: expire so the backend doesn't grow unbounded.
    result_expires=60 * 60 * 24,  # 24h
    # Celery 6 removes implicit startup retry; keep it explicit.
    broker_connection_retry_on_startup=True,
    # Redis broker: redeliver tasks not acked within the visibility window.
    broker_transport_options={"visibility_timeout": 3600},
    result_backend_transport_options={"visibility_timeout": 3600},
)

# Periodic schedule (the morning brief is the heartbeat of the product).
celery_app.conf.beat_schedule = {
    "build-daily-briefs": {
        "task": "navis.orchestration.brief.build_all_briefs",
        "schedule": 60 * 60,  # hourly check; orchestrator decides per-org timing
    },
    "sync-signals": {
        "task": "navis.orchestration.sync.sync_all_sources",
        "schedule": 60 * 5,
    },
    # Gmail watches expire in ~7 days; check renewals well ahead.
    "renew-watches": {
        "task": "navis.orchestration.sync.renew_watches",
        "schedule": 60 * 60 * 6,  # every 6h
    },
    # Execution: drain approved actions + relay the outbox to the bus.
    "execute-approved-actions": {
        "task": "navis.execution.worker.execute_approved_actions",
        "schedule": 30,
    },
    "relay-execution-outbox": {
        "task": "navis.execution.worker.relay_execution_outbox",
        "schedule": 15,
    },
    # Signals: relay the transactional outbox (staged pointer events) to the bus.
    "relay-signal-outbox": {
        "task": "navis.signals.outbox.relay_signal_outbox",
        "schedule": 5,
    },
    # Signals: purge published outbox rows past the retention window (bounds growth).
    "purge-signal-outbox": {
        "task": "navis.signals.outbox.purge_signal_outbox",
        "schedule": 60 * 60,  # hourly
    },
    # Memory: relay the transactional outbox (staged domain events) to the bus. Same
    # cadence as the signal outbox — a signal ingested at t drives a memory write and a
    # memory pointer within one relay tick, so the onboarding funnel advances end-to-end
    # within ~10s of the sync completing.
    "relay-memory-outbox": {
        "task": "navis.memory.outbox.relay_memory_outbox",
        "schedule": 5,
    },
    "purge-memory-outbox": {
        "task": "navis.memory.outbox.purge_memory_outbox",
        "schedule": 60 * 60,  # hourly
    },
    # Recover orphaned EXECUTING requests (dead workers).
    "recover-stale-executions": {
        "task": "navis.execution.worker.recover_stale_executions",
        "schedule": 60,
    },
    # Identity hygiene: reap expired sessions + expired revoked-token rows. Hourly is
    # ample — expiry is already enforced at auth time; this only reclaims storage. Both
    # are idempotent index-served bulk DELETEs, so overlap/re-run is harmless.
    "purge-expired-sessions": {
        "task": "navis.identity.maintenance.purge_expired_sessions",
        "schedule": 60 * 60,
    },
    "purge-expired-tokens": {
        "task": "navis.identity.maintenance.purge_expired_tokens",
        "schedule": 60 * 60,
    },
}


# Composition-root registration (matches main.py:14). Without this the worker
# registers its Celery tasks but NOT the event subscribers in navis.memory.service
# / navis.identity_resolution.service / etc. Consequence: the outbox relay task
# publishes SIGNAL_INGESTED / MEMORY_APPENDED events into a bus with no handlers,
# so new Gmail messages never extract into memory_records and the pipeline
# silently stalls at Layer 1 → Layer 2.
#
# Deferred via worker_process_init because a top-level `import navis.registry`
# hits a real circular-import: registry imports navis.execution.worker (etc.)
# whose module body calls _register_tasks() which does
# `from navis.platform.queue import celery_app, run_async` — but at that instant
# navis.platform.queue.__init__ is still executing its one and only import line
# and `run_async` is not yet in its namespace. Signal-based init runs AFTER
# celery has fully wired the queue package.
from celery.signals import worker_process_init  # noqa: E402


@worker_process_init.connect
def _register_event_subscribers(**_kwargs) -> None:
    import navis.registry  # noqa: F401  (subscribe side-effects)
