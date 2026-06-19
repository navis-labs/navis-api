"""Attribution — classify an observed outcome and attach its lineage.

Learning owns the outcome_type -> result mapping (NOT Observation, which stores
the fact; NOT an LLM). Pure + deterministic. Produces the immutable LearningEvent
values from an OUTCOME_OBSERVED payload.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from navis.learning.models import (
    LEARNING_SCHEMA_VERSION,
    LearningResult,
    learning_event_id,
)

_RESULT_FOR_OUTCOME = {
    "reply_received": LearningResult.SUCCESS,
    "meeting_accepted": LearningResult.SUCCESS,
    "meeting_completed": LearningResult.SUCCESS,
    "action_failed": LearningResult.FAILURE,
    "no_response": LearningResult.FAILURE,
    "action_executed": LearningResult.NEUTRAL,  # attempted; real result arrives later
}


def classify(outcome_type: str) -> LearningResult:
    return _RESULT_FOR_OUTCOME.get(outcome_type, LearningResult.NEUTRAL)


def _maybe_uuid(value):
    import uuid

    if isinstance(value, str) and value:
        try:
            return uuid.UUID(value)
        except ValueError:
            return None
    return None


def learning_event_values(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Build LearningEvent column values from an OUTCOME_OBSERVED payload.

    Requires outcome_id (dedupe key + deterministic id) and occurred_at (no now()).
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
    return {
        "id": learning_event_id(str(outcome_id)),
        "organization_id": org_id,
        "outcome_id": _maybe_uuid(outcome_id),
        "execution_id": _maybe_uuid(payload.get("execution_id")),
        "decision_id": _maybe_uuid(payload.get("decision_id")),
        "reasoning_signal_ids": payload.get("reasoning_signal_ids", []),
        "reasoning_version": payload.get("reasoning_version"),
        "state_watermark": payload.get("state_watermark"),
        "outcome_type": outcome_type,
        "result": classify(outcome_type).value,
        "schema_version": LEARNING_SCHEMA_VERSION,
        "source_event_id": str(outcome_id),
        "occurred_at": occurred_at,
    }
