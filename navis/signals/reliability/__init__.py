"""Signals reliability primitives.

Boring, proven building blocks for capturing reality at scale:

    RetryPolicy          exponential backoff + jitter, honors Retry-After
    DeadLetterQueue      poison-item isolation (durable Redis stream)
    ReplayService        re-emit stored signals from the append-only log
    BackfillService      bounded historical capture for an integration
    WatchRecoveryService renew/recreate expiring push channels

Only the leaf primitives (retry, dead_letter) load eagerly. The DB/connector
backed services are loaded lazily via __getattr__ so importing this package never
triggers a connector/transport import cycle.
"""
from typing import Any

from navis.signals.reliability.retry import (
    DEFAULT_RETRY_POLICY,
    RetryPolicy,
    async_retrying,
)
from navis.signals.reliability.dead_letter import DeadLetterQueue, dead_letter_queue

__all__ = [
    "RetryPolicy",
    "DEFAULT_RETRY_POLICY",
    "async_retrying",
    "DeadLetterQueue",
    "dead_letter_queue",
    "ReplayService",
    "BackfillService",
    "WatchRecoveryService",
]


def __getattr__(name: str) -> Any:  # lazy heavy services (PEP 562)
    if name == "ReplayService":
        from navis.signals.reliability.replay import ReplayService

        return ReplayService
    if name == "BackfillService":
        from navis.signals.reliability.backfill import BackfillService

        return BackfillService
    if name == "WatchRecoveryService":
        from navis.signals.reliability.watch_recovery import WatchRecoveryService

        return WatchRecoveryService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
