"""Attribution — classify an observed outcome and attach its lineage.

Learning owns the outcome_type -> (surface, result) mapping (NOT Observation,
which stores the fact; NOT an LLM). Pure + deterministic. Produces the immutable
LearningEvent values from an OUTCOME_OBSERVED payload.

TWO AXES, NOT ONE. `result` says whether the fact was a win, a loss or neither;
`surface` says which of Observation's three sources it came from. Both are
needed, because "did it work?" and "is this evidence about our judgement?" are
different questions and only the second one decides whether an outcome may move
a signal's success rate. See `OutcomeSurface` and `aggregates.attributable_delta`.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from navis.learning.models import (
    LEARNING_SCHEMA_VERSION,
    LearningResult,
    OutcomeSurface,
    learning_event_id,
)

# THE VOCABULARY, and it is Observation's — every key below is a real member of
# `navis.observation.models.OutcomeType`, kept in step by a test. Nothing here is
# an invented ML label.
#
# The two entries that were MISSING are the reason this table changed:
# ACTION_REJECTED and ACTION_CANCELLED have been emitted by Execution and
# recorded by Observation all along, and neither appeared here — so both fell
# through `classify`'s default to NEUTRAL. A founder pressing Reject on a
# proposal, which Observation's own docstring calls "the clearest negative
# signal the product can get", was being learned as "no information".
_OUTCOME_VOCABULARY: dict[str, tuple[OutcomeSurface, LearningResult]] = {
    # ── Delivery: what the machine did ────────────────────────────────
    # An attempt, not a win: the real result arrives later, from the world.
    "action_executed": (OutcomeSurface.DELIVERY, LearningResult.NEUTRAL),
    # A loss at the DELIVERY surface — the send genuinely failed. Decisive
    # about the channel; `attributable_delta` keeps it from being counted as
    # decisive about the reasoning that asked for the send.
    "action_failed": (OutcomeSurface.DELIVERY, LearningResult.FAILURE),

    # ── Judgement: what a person decided about a proposal ─────────────
    # A verdict: "no, not this". Decisive evidence about the proposal itself.
    "action_rejected": (OutcomeSurface.JUDGEMENT, LearningResult.FAILURE),
    # A withdrawal, usually of a request overtaken by events — NOT a verdict on
    # quality. Observation deliberately keeps these two apart ("collapsing them
    # would teach a future Learning pass that the two are the same evidence,
    # which they are not"); scoring a cancellation as a failure here would
    # perform exactly that collapse one layer down.
    "action_cancelled": (OutcomeSurface.JUDGEMENT, LearningResult.NEUTRAL),
    # A verdict on the CONCLUSION rather than on a proposed action: the founder
    # removed this decision from their brief. Decisive, and negative — Navis
    # raised something that did not need them, which is direct evidence about
    # the reasoning that raised it. That is what makes it useful: `risk`
    # decisions produce no execution at all, so a dismissal is the ONLY
    # judgement signal their signals can ever receive, and leaving it
    # unclassified meant the most-raised conclusions were also the least
    # measured.
    #
    # FAILURE for all three dismissal reasons. "explicit" (did not need me),
    # "stale" (out of date) and "duplicate" (already told me) are three ways of
    # saying the brief should not have carried it; the reason itself is stored
    # on the outcome payload, so splitting them later needs no re-derivation.
    "decision_dismissed": (OutcomeSurface.JUDGEMENT, LearningResult.FAILURE),

    # ── World: what actually happened afterwards ──────────────────────
    "reply_received": (OutcomeSurface.WORLD, LearningResult.SUCCESS),
    "meeting_accepted": (OutcomeSurface.WORLD, LearningResult.SUCCESS),
    "meeting_completed": (OutcomeSurface.WORLD, LearningResult.SUCCESS),
    "no_response": (OutcomeSurface.WORLD, LearningResult.FAILURE),
}

# An outcome_type Learning has not been taught. Recorded as a delivery attempt:
# NEUTRAL so it can never move a rate in either direction, DELIVERY so it can
# never be mistaken for a verdict or a world response. `outcome_type` is a plain
# String(40) upstream precisely so a new member needs no migration — it stays
# inert here until someone decides what it means.
_UNCLASSIFIED = (OutcomeSurface.DELIVERY, LearningResult.NEUTRAL)


def classify(outcome_type: str) -> LearningResult:
    """Win, loss, or neither."""
    return _OUTCOME_VOCABULARY.get(outcome_type, _UNCLASSIFIED)[1]


def classify_surface(outcome_type: str) -> OutcomeSurface:
    """Delivery, judgement, or world."""
    return _OUTCOME_VOCABULARY.get(outcome_type, _UNCLASSIFIED)[0]


def _maybe_uuid(value):
    if isinstance(value, str) and value:
        try:
            return uuid.UUID(value)
        except ValueError:
            return None
    return None


def learning_event_values(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Build LearningEvent column values from an OUTCOME_OBSERVED payload.

    Requires outcome_id (dedupe key + deterministic id) and occurred_at (no now()).
    Reads the event and NOTHING else — no Execution, Observation, provider or
    connector lookup — so replaying the same event always rebuilds the same row.
    """
    outcome_id = payload.get("outcome_id")
    org_id = _maybe_uuid(payload.get("org_id"))
    occurred_raw = payload.get("occurred_at")
    if not outcome_id or org_id is None or not occurred_raw:
        return None
    try:
        occurred_at = datetime.fromisoformat(occurred_raw)
    except (TypeError, ValueError):
        return None

    outcome_type = payload.get("outcome_type", "")
    surface, result = _OUTCOME_VOCABULARY.get(outcome_type, _UNCLASSIFIED)
    # Observation nests the cause fields under `payload` and omits a key entirely
    # when it does not apply, so "absent" stays distinguishable from "null".
    observed = payload.get("payload") or {}
    return {
        "id": learning_event_id(str(outcome_id)),
        "organization_id": org_id,
        "outcome_id": _maybe_uuid(outcome_id),
        "execution_id": _maybe_uuid(payload.get("execution_id")),
        "decision_id": _maybe_uuid(payload.get("decision_id")),
        "reasoning_signal_ids": payload.get("reasoning_signal_ids", []),
        "reasoning_version": payload.get("reasoning_version"),
        "state_watermark": payload.get("state_watermark"),
        "provider_reference": payload.get("provider_reference") or {},
        "outcome_type": outcome_type,
        "outcome_surface": surface.value,
        "result": result.value,
        "failure_kind": observed.get("failure_kind"),
        "schema_version": LEARNING_SCHEMA_VERSION,
        "source_event_id": str(outcome_id),
        "occurred_at": occurred_at,
    }
