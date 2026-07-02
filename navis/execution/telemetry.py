"""Execution telemetry — provider latency/success/failure/retry + request durations.

In-memory counters (scraped per process) + structured logs. No new tables; request
durations are derived from the timestamps already on the row.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


class ProviderTelemetry:
    def __init__(self) -> None:
        self.counters: Counter = Counter()

    # No `retried` counter: the in-call retry count lives inside execute_with_retry
    # (tenacity) and cannot reach this seam without changing the ProviderResult contract.
    # The old flag counted "failed" (already counted) as "retried" — false data that would
    # mislead an incident. Add a real retry metric only when the retry layer can report it.
    def record(self, provider: str, *, ok: bool, latency_s: float) -> None:
        self.counters[f"{provider}.calls"] += 1
        self.counters[f"{provider}.{'success' if ok else 'failure'}"] += 1
        self.counters[f"{provider}.latency_ms_total"] += int(latency_s * 1000)

    def success_rate(self, provider: str) -> float | None:
        calls = self.counters.get(f"{provider}.calls", 0)
        if not calls:
            return None
        return self.counters.get(f"{provider}.success", 0) / calls

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)


provider_telemetry = ProviderTelemetry()


def request_durations(req) -> dict[str, Any]:
    """approval_latency (created->approved) and execution_duration (approved->executed)."""
    approval = None
    duration = None
    if req.approved_at and req.created_at:
        approval = (req.approved_at - req.created_at).total_seconds()
    if req.executed_at and req.approved_at:
        duration = (req.executed_at - req.approved_at).total_seconds()
    return {"approval_latency_s": approval, "execution_duration_s": duration}
