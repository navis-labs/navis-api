"""Context-layer metrics — a tiny, deterministic façade over the platform metrics sink.

This is NOT an observability framework: it adds no registry, no batching, no sampling, no
state. It only (1) namespaces every metric under `context.`, (2) validates names/values so a
typo or a NaN can never corrupt a dashboard, (3) drops known high-cardinality label keys so
the metrics backend cannot explode in cost, and (4) makes emission BEST-EFFORT — a failing
sink can never break product correctness (metrics.incr is called inline in permissions /
compression / repository, not only behind the service's telemetry guard, so the guarantee
must live here). Stateless, allocation-light, thread-safe, replay-safe: no clock, no uuid,
no randomness, no mutable globals; it reads nothing (no State/Memory/DB/Signals).
"""
from __future__ import annotations

import math
import re
from typing import Final, Mapping

from navis.platform.monitoring.metrics import metrics

# Immutable namespace — must never drift (dashboards key off it).
_PREFIX: Final[str] = "context."
_MAX_NAME_LEN: Final[int] = 96
# A valid metric name: lowercase, starts with a letter, only [a-z0-9_.], bounded length —
# rejects empty/whitespace/control/newline/absurd-length names structurally.
_NAME_RE: Final = re.compile(r"\A[a-z][a-z0-9_.]{0,%d}\Z" % (_MAX_NAME_LEN - 1))
# Label KEYS that explode cardinality (per-user/per-request/per-id) — dropped to protect the
# metrics backend from unbounded series. Metrics here are infrastructure, never per-subject.
_FORBIDDEN_LABEL_KEYS: Final[frozenset[str]] = frozenset({
    "user_id", "actor_id", "org_id", "organization_id", "email", "request_id", "trace_id",
    "span_id", "session_id", "uuid", "id", "context_id", "memory_id", "signal_id",
    "timestamp", "ts", "created_at", "filename", "path", "url",
})


def _valid_name(name: str) -> bool:
    return isinstance(name, str) and _NAME_RE.match(name) is not None


def _finite(value: float) -> bool:
    # bool is an int subclass — exclude it so True/False can't masquerade as a count.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _safe_labels(labels: Mapping[str, str] | None) -> dict[str, str] | None:
    """Return a NEW dict with high-cardinality keys removed (never mutate the caller's
    mapping). None for empty so the sink gets a clean unlabelled metric."""
    if not labels:
        return None
    cleaned = {k: v for k, v in labels.items() if k not in _FORBIDDEN_LABEL_KEYS}
    return cleaned or None


def incr(name: str, *, labels: Mapping[str, str] | None = None, value: float = 1.0) -> None:
    """Increment a `context.<name>` counter. Invalid name/value is silently skipped (never
    raises); a sink failure is swallowed — emitting a metric must never break the product."""
    if not _valid_name(name) or not _finite(value):
        return
    try:
        metrics.incr(_PREFIX + name, value, _safe_labels(labels))
    except Exception:
        pass  # best-effort: a metrics outage is an observability problem, not a product one


def observe_ms(name: str, ms: float, *, labels: Mapping[str, str] | None = None) -> None:
    """Observe a `context.<name>` latency (ms) into a histogram (P50/P95/P99 fall out of the
    sink). A non-finite or negative observation is skipped so it can never poison the
    histogram; a sink failure is swallowed."""
    if not _valid_name(name) or not _finite(ms) or ms < 0:
        return
    try:
        metrics.observe_ms(_PREFIX + name, ms, _safe_labels(labels))
    except Exception:
        pass
