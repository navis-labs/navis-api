"""Structured logging + timing for Context assembly.

`timed` records a latency histogram (P50/P95/P99 fall out of the sink) and never
swallows errors. Logs are structured key=value so an assembly is fully explainable:
which task, which actor, how big the scope, how much was dropped by the budget.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from navis.context import metrics
from navis.platform.monitoring import get_logger

logger = get_logger("context")


@contextmanager
def timed(metric: str, *, labels: dict[str, str] | None = None) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics.observe_ms(metric, (time.perf_counter() - start) * 1000.0, labels=labels)


def log_built(*, org_id, task: str, context_id: str, admitted: int, dropped: int, cache_hit: bool) -> None:
    logger.info(
        "context built org=%s task=%s context_id=%s admitted=%d dropped=%d cache_hit=%s",
        org_id, task, context_id, admitted, dropped, cache_hit,
    )


# Dedicated audit logger — a structured, forensics-grade record per bundle generation,
# routed to the standard log pipeline (the SIEM/audit sink). Carries NO entity content or
# PII: only ids, NON-REVERSIBLE fingerprints, counts, sizes, and timings. GDPR-safe.
audit_logger = get_logger("context.audit")


def log_audit(
    *,
    org_id,
    actor_fingerprint: str,
    actor_kind: str,
    task: str,
    context_id: str,
    permission_fingerprint: str,
    policy_fingerprint: str,
    entity_count: int,
    bundle_tokens: int,
    state_as_of,
    latency_ms: float,
    cache_hit: bool,
) -> None:
    """Emit the immutable audit record for one ContextBundle generation."""
    audit_logger.info(
        "context.audit org=%s actor_fp=%s actor_kind=%s task=%s context_id=%s perm_fp=%s "
        "policy_fp=%s entities=%d tokens=%d state_as_of=%s latency_ms=%.2f cache_hit=%s",
        org_id, actor_fingerprint, actor_kind, task, context_id, permission_fingerprint,
        policy_fingerprint, entity_count, bundle_tokens,
        state_as_of.isoformat() if state_as_of else "", latency_ms, cache_hit,
    )
    metrics.incr("audit", labels={"task": task, "cache_hit": str(cache_hit).lower()})
