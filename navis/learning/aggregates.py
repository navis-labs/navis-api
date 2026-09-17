"""Deterministic count aggregation. Counts commute, so aggregates are order-
independent and replay-identical — no EMA, no float drift.

One learning event fans out to: each reasoning signal_id (SIGNAL), its
decision_id (DECISION), and its outcome_type (OUTCOME).
"""
from __future__ import annotations

from dataclasses import dataclass

from navis.learning.models import AggregateDimension, LearningResult, OutcomeSurface


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


def attributable_delta(result: str, surface: str) -> Delta:
    """The delta a SIGNAL or DECISION aggregate receives.

    These two dimensions answer one question — "when this signal fired / this
    decision was made, did it turn out to be right?" — and only two of the three
    surfaces are evidence for it.

    A DELIVERY outcome is counted, never counted AGAINST. `action_failed` means
    the send broke: the org lost its Google credential, the channel was gone,
    the retry budget ran out (see `execution.contract.FailureKind` — every
    member is an infrastructure or entitlement fact). None of that is evidence
    that firing the signal was the wrong call, yet it used to increment
    `failures` on that signal, so a week of Gmail outages read identically to a
    week of customers saying no. Recording it as a neutral keeps the attempt in
    `total` — nothing is lost, and `learning_events` still holds the failure with
    its `failure_kind` — while keeping the channel's health out of the
    reasoning's success rate.

    JUDGEMENT and WORLD outcomes are decisive as classified: a human rejecting
    the proposal and a customer replying to it are both direct verdicts on what
    was decided.

    The OUTCOME dimension does NOT use this function. Its key IS the
    outcome_type, so delivery and world facts already occupy different rows
    there and counting them as classified conflates nothing — which is what
    keeps delivery failure rates measurable.
    """
    if surface == OutcomeSurface.DELIVERY.value:
        return Delta(0, 0, 1, 1)
    return delta_for(result)


def aggregate_keys(learning_event) -> list[tuple[str, str, Delta]]:
    """(dimension, key, delta) increments for one learning event."""
    attributable = attributable_delta(learning_event.result, learning_event.outcome_surface)
    out: list[tuple[str, str, Delta]] = []
    for sid in (learning_event.reasoning_signal_ids or []):
        out.append((AggregateDimension.SIGNAL.value, str(sid), attributable))
    if learning_event.decision_id is not None:
        out.append((AggregateDimension.DECISION.value, str(learning_event.decision_id), attributable))
    if learning_event.outcome_type:
        out.append(
            (AggregateDimension.OUTCOME.value, learning_event.outcome_type, delta_for(learning_event.result))
        )
    return out


def success_rate(successes: int, failures: int) -> float | None:
    """Wins / (wins + losses). Neutrals don't count toward the rate. None if no
    decisive samples yet."""
    decisive = successes + failures
    if decisive == 0:
        return None
    return round(successes / decisive, 4)
