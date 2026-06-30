"""Context-layer metrics — thin, namespaced wrappers over the platform metrics sink.

No new metrics system: reuses navis.platform.monitoring.metrics. Every counter/histogram
is prefixed `context.` so dashboards can isolate the layer. Latency is an observed
histogram so P50/P95/P99 fall out of the sink.
"""
from __future__ import annotations

from navis.platform.monitoring.metrics import metrics

_PREFIX = "context."


def incr(name: str, *, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
    metrics.incr(_PREFIX + name, value, labels)


def observe_ms(name: str, ms: float, *, labels: dict[str, str] | None = None) -> None:
    metrics.observe_ms(_PREFIX + name, ms, labels)
