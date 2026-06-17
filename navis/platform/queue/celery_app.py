"""Celery app for background work (signal sync, brief generation, observation polling).

Tasks themselves live in their owning domains (e.g. orchestration/, signals/),
never here. This module only configures the broker.
"""
from __future__ import annotations

from celery import Celery

from navis.platform.config import settings

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
}
