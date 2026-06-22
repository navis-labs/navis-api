"""Minimal in-process metrics: counters + latency histograms.

Zero new dependencies (no Prometheus client, no OpenTelemetry) — a pure-Python
registry that any process can snapshot and a future `/metrics` endpoint or exporter
can render. Deliberately small: this is the abstraction point for richer
observability, not a full telemetry stack (which would be premature today).

Usage:
    from navis.platform.monitoring.metrics import metrics, timed

    metrics.incr("events.published", labels={"type": "OrgCreated"})
    with timed("pipeline.brief"):
        ...
Histograms record fixed buckets (ms) so percentiles can be approximated cheaply.
"""
from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

from navis.platform.config import settings

# Latency buckets in milliseconds (cumulative-friendly).
_BUCKETS_MS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


def _key(name: str, labels: dict[str, str] | None) -> str:
    if not labels:
        return name
    tags = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{tags}}}"


class Metrics:
    """Thread-safe (locks are uncontended in async single-thread workers too)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._hist_counts: dict[str, list[int]] = {}
        self._hist_sum: dict[str, float] = defaultdict(float)
        self._hist_n: dict[str, int] = defaultdict(int)

    def incr(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        if not settings.metrics_enabled:
            return
        with self._lock:
            self._counters[_key(name, labels)] += value

    def observe_ms(self, name: str, ms: float, labels: dict[str, str] | None = None) -> None:
        if not settings.metrics_enabled:
            return
        k = _key(name, labels)
        with self._lock:
            buckets = self._hist_counts.setdefault(k, [0] * len(_BUCKETS_MS))
            for i, edge in enumerate(_BUCKETS_MS):
                if ms <= edge:
                    buckets[i] += 1
                    break
            else:
                buckets[-1] += 1
            self._hist_sum[k] += ms
            self._hist_n[k] += 1

    def snapshot(self) -> dict:
        """A plain dict suitable for a /metrics endpoint or structured log dump."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {
                    k: {
                        "count": self._hist_n[k],
                        "sum_ms": round(self._hist_sum[k], 3),
                        "avg_ms": round(self._hist_sum[k] / self._hist_n[k], 3) if self._hist_n[k] else 0.0,
                        "buckets_ms": dict(zip(_BUCKETS_MS, self._hist_counts[k])),
                    }
                    for k in self._hist_counts
                },
            }


metrics = Metrics()


@contextmanager
def timed(name: str, labels: dict[str, str] | None = None) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics.observe_ms(name, (time.perf_counter() - start) * 1000.0, labels)


# --- Prometheus text exposition (render-only; no new collection, no deps) ---

# Prometheus metric/label names: [a-zA-Z_:][a-zA-Z0-9_:]* — sanitize everything else.
_PROM_NAME_RE = re.compile(r"[^a-zA-Z0-9_:]")
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _prom_name(name: str) -> str:
    safe = _PROM_NAME_RE.sub("_", name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "_"


def _split_key(key: str) -> tuple[str, dict[str, str]]:
    """Reverse of _key(): 'name' or 'name{k=v,k2=v2}' -> (name, {k: v})."""
    if key.endswith("}") and "{" in key:
        name, rest = key.split("{", 1)
        labels: dict[str, str] = {}
        for pair in rest[:-1].split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
        return name, labels
    return key, {}


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    body = ",".join(f'{_prom_name(k)}="{_esc(v)}"' for k, v in sorted(labels.items()))
    return "{" + body + "}"


def render_prometheus(snap: dict | None = None) -> str:
    """Render metrics.snapshot() as Prometheus text exposition (v0.0.4).

    Pure function of the snapshot — no I/O and no secrets (only metric names, the
    low-cardinality labels callers pass, and integer counts). Names are sanitized to the
    Prometheus charset; the in-process per-bucket histogram counts are accumulated into
    the cumulative `le` buckets Prometheus expects. Scrapeable by Prometheus & Grafana."""
    snap = snap if snap is not None else metrics.snapshot()
    out: list[str] = []
    typed: set[str] = set()
    counters = snap.get("counters", {})
    for key in sorted(counters):
        name, labels = _split_key(key)
        pname = _prom_name(name)
        if pname not in typed:
            out.append(f"# TYPE {pname} counter")
            typed.add(pname)
        out.append(f"{pname}{_labels(labels)} {counters[key]}")
    histograms = snap.get("histograms", {})
    for key in sorted(histograms):
        h = histograms[key]
        name, labels = _split_key(key)
        pname = _prom_name(name)
        out.append(f"# TYPE {pname} histogram")
        cumulative = 0
        for edge, cnt in h["buckets_ms"].items():
            cumulative += cnt
            out.append(f'{pname}_bucket{_labels({**labels, "le": str(edge)})} {cumulative}')
        out.append(f'{pname}_bucket{_labels({**labels, "le": "+Inf"})} {h["count"]}')
        out.append(f"{pname}_sum{_labels(labels)} {h['sum_ms']}")
        out.append(f"{pname}_count{_labels(labels)} {h['count']}")
    return "\n".join(out) + "\n"
