"""The governing-policy schema: the variables Adaptive may propose changes to.

Pure module — no DB, no clock, no business-domain imports. It is the single
source of truth for *what* the governing variables are, their safe bounds, and
how fast they may move. DEFAULTS EXACTLY MATCH the current hardcoded constants in
reasoning/decision/execution, so externalizing policy changes behavior by nothing
until a founder activates a different snapshot. (A parity test asserts this.)

Adaptive sits above Intelligence; this module is consumed, never a consumer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# A change may not move a value by more than this fraction of its current value
# in a single policy version (catastrophic-jump guard). Mission: <= 25%.
MAX_CHANGE_RATIO = 0.25


@dataclass(frozen=True, slots=True)
class FieldSpec:
    default: float
    min: float
    max: float
    kind: str  # "int" | "float"
    max_change_ratio: float = MAX_CHANGE_RATIO

    def cast(self, value: float) -> float | int:
        return int(round(value)) if self.kind == "int" else round(float(value), 4)


# name -> spec. Defaults mirror the live constants (see parity test):
#   commitment_slipping_days   reasoning/v2/engines/risk.py  COMMITMENT_SLIPPING_DAYS=5.0
#   relationship_cooling_days  reasoning/v2/engines/risk.py  RELATIONSHIP_COOLING_DAYS=7.0
#   project_stalled_days       reasoning/v2/engines/risk.py  PROJECT_STALLED_DAYS=10.0
#   customer_drop_days         reasoning/v2/engines/risk.py  CUSTOMER_DROP_DAYS=14.0
#   meeting_overload_count     reasoning/v2/engines/risk.py  MEETING_OVERLOAD_COUNT=5
#   leverage_min_people        reasoning/v2/engines/meta.py  LEVERAGE_MIN_PEOPLE=2
#   brief_max_items            decisions/what_matters_today.py  max_items=7
#   brief_min_confidence       decisions/what_matters_today.py  min_confidence=0.45
#   execution_retry_limit      execution/config.py  MAX_EXECUTION_ATTEMPTS=5
POLICY_FIELDS: dict[str, FieldSpec] = {
    "commitment_slipping_days": FieldSpec(5.0, 1.0, 60.0, "float"),
    "relationship_cooling_days": FieldSpec(7.0, 1.0, 60.0, "float"),
    "project_stalled_days": FieldSpec(10.0, 1.0, 90.0, "float"),
    "customer_drop_days": FieldSpec(14.0, 1.0, 120.0, "float"),
    "meeting_overload_count": FieldSpec(5, 1, 50, "int"),
    "leverage_min_people": FieldSpec(2, 1, 20, "int"),
    "brief_max_items": FieldSpec(7, 1, 25, "int"),
    "brief_min_confidence": FieldSpec(0.45, 0.0, 0.95, "float"),
    "execution_retry_limit": FieldSpec(5, 1, 10, "int"),
}

DEFAULT_POLICY: dict[str, Any] = {name: spec.cast(spec.default) for name, spec in POLICY_FIELDS.items()}


class PolicyViolation(ValueError):
    """Raised when a proposed value escapes its bounds or magnitude cap."""


def clamp(field: str, value: float) -> float | int:
    spec = POLICY_FIELDS[field]
    bounded = min(spec.max, max(spec.min, float(value)))
    return spec.cast(bounded)


def cap_change(field: str, old: float, new: float) -> tuple[float | int, bool]:
    """Clamp `new` to <= max_change_ratio away from `old`, then to bounds.

    Returns (value, was_capped). Pure and deterministic.
    """
    spec = POLICY_FIELDS[field]
    span = abs(old) * spec.max_change_ratio
    lo, hi = old - span, old + span
    limited = min(hi, max(lo, float(new)))
    was_capped = limited != float(new)
    if spec.kind == "int":
        # Round TOWARD `old` so an integer cast can never overshoot the cap.
        rounded = math.floor(limited) if limited >= old else math.ceil(limited)
        return int(clamp(field, rounded)), was_capped
    return clamp(field, limited), was_capped


def validate_value(field: str, value: float) -> None:
    if field not in POLICY_FIELDS:
        raise PolicyViolation(f"unknown policy field: {field}")
    spec = POLICY_FIELDS[field]
    if not (spec.min <= float(value) <= spec.max):
        raise PolicyViolation(f"{field}={value} out of bounds [{spec.min}, {spec.max}]")


def validate_change(field: str, old: float, new: float) -> None:
    validate_value(field, new)
    span = abs(old) * POLICY_FIELDS[field].max_change_ratio
    if abs(float(new) - float(old)) > span + 1e-9:
        raise PolicyViolation(
            f"{field} change {old}->{new} exceeds {int(POLICY_FIELDS[field].max_change_ratio*100)}% cap"
        )


def merge_defaults(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Full policy = defaults overlaid by any provided fields (unknowns dropped).

    TOTAL and fail-safe: this is the read seam that produces the EFFECTIVE policy the
    pipeline consumes (orchestration -> Decision), and its caller documents "never
    raises". Every sanctioned writer already validates against POLICY_FIELDS, so for
    valid stored data this is byte-identical to a bare cast — but a corrupt or
    historical snapshot value must degrade, never propagate: an out-of-bounds number
    is clamped to the field's declared safe bounds, and an un-castable value falls
    back to the field default. Without this, a single bad stored value either crashes
    brief assembly for the org (ValueError here) or silently empties its briefs
    (e.g. brief_min_confidence=5.0 filters every signal)."""
    out = dict(DEFAULT_POLICY)
    for k, v in (payload or {}).items():
        if k in POLICY_FIELDS:
            try:
                out[k] = clamp(k, v)
            except (TypeError, ValueError):
                pass  # un-castable stored value -> keep the safe code default
    return out


def validate_payload(payload: dict[str, Any]) -> None:
    for field, value in payload.items():
        validate_value(field, value)
