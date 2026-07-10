"""Structured logging + timing for Context assembly — the layer's ONLY logging boundary.

Telemetry is observability-only. It MUST NEVER change a ContextBundle, the context_id,
ordering, replay, caching, or determinism, and a telemetry failure MUST NEVER fail an
assembly. Both guarantees live HERE, at the boundary:

  * Every log emission is BEST-EFFORT (`_safe_emit`): a logger, formatter, sink, or
    serialization failure is swallowed — it can never propagate. This matters because
    `log_built` runs on the non-isolated build path (a cache hit / a just-persisted bundle),
    where a propagated logging error would otherwise fail an already-complete request.
  * Metric emission is best-effort by virtue of the metrics façade (it swallows sink errors).
  * `timed` records a latency histogram (P50/P95/P99 fall out of the sink) on a MONOTONIC
    timer, ALWAYS (in `finally`), and never swallows the wrapped block's own exception.

Logs are a stable `key=value` schema so an assembly is fully explainable (which task, which
actor, scope size, how much the budget dropped) and SIEM/forensics-friendly. They carry NO
entity content / PII: only ids, NON-REVERSIBLE fingerprints, counts, sizes, and timings.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from navis.context import metrics
from navis.platform.monitoring import get_logger

logger = get_logger("context")
# Dedicated audit logger — a forensics-grade record per bundle generation, routed to the
# standard log pipeline (the SIEM/audit sink). Same PII guarantee: ids, fingerprints, counts.
audit_logger = get_logger("context.audit")


def _safe_emit(log: object, message: str, *args: object) -> None:
    """Best-effort structured log. A logging / formatting / sink / serialization failure must
    never propagate into Context assembly — telemetry is observability, not correctness."""
    try:
        log.info(message, *args)  # type: ignore[attr-defined]  # duck-typed logger
    except Exception:
        pass


def _iso(ts: object) -> str:
    """Render a logical timestamp deterministically and defensively (never raises)."""
    try:
        return ts.isoformat() if ts is not None else ""  # type: ignore[attr-defined]
    except Exception:
        return ""


@contextmanager
def timed(metric: str, *, labels: dict[str, str] | None = None) -> Iterator[None]:
    """Record a latency histogram for the wrapped block. Monotonic timer; ALWAYS records (in
    `finally`); the record is itself best-effort (the metrics façade swallows sink errors) so
    it never masks the block's own exception. Reentrant — uses only locals, so nesting and
    parallel builds are safe. The timing never enters the bundle/context_id (determinism)."""
    start = time.perf_counter()  # monotonic; observability-only, never folded into the bundle
    try:
        yield
    finally:
        metrics.observe_ms(metric, (time.perf_counter() - start) * 1000.0, labels=labels)


def log_built(*, org_id, task: str, context_id: str, admitted: int, dropped: int, cache_hit: bool) -> None:
    _safe_emit(
        logger,
        "context built org=%s task=%s context_id=%s admitted=%d dropped=%d cache_hit=%s",
        org_id, task, context_id, admitted, dropped, cache_hit,
    )


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
    """Emit the immutable audit record for one ContextBundle generation (SOC2/GDPR/forensics):
    enough to reconstruct WHO (actor fingerprint+kind) asked for WHAT (task) over WHICH inputs
    (context_id, perm/policy fingerprints, state_as_of) and HOW MUCH (entities, tokens, latency,
    cache_hit) — and nothing sensitive. Every field is deterministic given the bundle."""
    _safe_emit(
        audit_logger,
        "context.audit org=%s actor_fp=%s actor_kind=%s task=%s context_id=%s perm_fp=%s "
        "policy_fp=%s entities=%d tokens=%d state_as_of=%s latency_ms=%.2f cache_hit=%s",
        org_id, actor_fingerprint, actor_kind, task, context_id, permission_fingerprint,
        policy_fingerprint, entity_count, bundle_tokens, _iso(state_as_of), latency_ms, cache_hit,
    )
    metrics.incr("audit", labels={"task": task, "cache_hit": str(cache_hit).lower()})
