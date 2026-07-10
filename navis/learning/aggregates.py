"""Deterministic count aggregation. Counts commute, so aggregates are order-
independent and replay-identical — no EMA, no float drift.

One learning event fans out to: each reasoning signal_id (SignalLearning), its
decision_id (DecisionLearning), and its outcome_type (OutcomeLearning).
"""
from __future__ import annotations

from dataclasses import dataclass

from navis.learning.models import AggregateDimension, LearningEvent, LearningResult


@dataclass(frozen=True, slots=True)
class Delta:
    successes: int
    failures: int
    neutrals: int
    total: int


def delta_for(result: str) -> Delta:
    if result == LearningResult.SUCCESS.value:
        return Delta(1, 0, 0, 1)
    if result == LearningResult.FAILURE.value:
        return Delta(0, 1, 0, 1)
    return Delta(0, 0, 1, 1)  # neutral


def aggregate_keys(learning_event: LearningEvent) -> list[tuple[str, str, Delta]]:
    """(dimension, key, delta) increments for one learning event."""
    delta = delta_for(learning_event.result)
    out: list[tuple[str, str, Delta]] = []
    # Order-preserving dedup of the signal ids. reasoning_signal_ids is a plural JSONB
    # list; a repeated id would fan out to TWO SIGNAL increments for ONE event, double-
    # counting that signal's success/failure and skewing its success_rate + pattern.
    # Upstream guarantees uniqueness today (the reasoning runner dedups by signal_id, and
    # a decision carries one), so this is a no-op for valid input — it just makes the
    # count invariant robust to that plural field without any behaviour change.
    for sid in dict.fromkeys(str(s) for s in (learning_event.reasoning_signal_ids or [])):
        out.append((AggregateDimension.SIGNAL.value, sid, delta))
    if learning_event.decision_id is not None:
        out.append((AggregateDimension.DECISION.value, str(learning_event.decision_id), delta))
    if learning_event.outcome_type:
        out.append((AggregateDimension.OUTCOME.value, learning_event.outcome_type, delta))
    return out


def success_rate(successes: int, failures: int) -> float | None:
    """Wins / (wins + losses). Neutrals don't count toward the rate. None if no
    decisive samples yet."""
    decisive = successes + failures
    if decisive == 0:
        return None
    return round(successes / decisive, 4)
