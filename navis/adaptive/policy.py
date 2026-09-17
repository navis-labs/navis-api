"""The governing-policy schema: the variables Adaptive may propose changes to.

Pure module — no DB, no clock, no business-domain imports. It is the single
source of truth for *what* the governing variables are, their safe bounds, and
how fast they may move. DEFAULTS EXACTLY MATCH the current hardcoded constants in
reasoning/decision/execution, so externalizing policy changes behavior by nothing
until a founder activates a different snapshot. (A parity test asserts this.)

Adaptive sits above Intelligence; this module is consumed, never a consumer.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any

# A change may not move a value by more than this fraction of its current value
# in a single policy version (catastrophic-jump guard). Mission: <= 25%.
MAX_CHANGE_RATIO = 0.25


class PolicyTarget(str, enum.Enum):
    """WHICH layer a governed variable tunes.

    The vocabulary was a flat dict of numbers, so nothing recorded what a given
    knob actually affects. That mattered for one reason above all: the boundary
    "Adaptive may tune how the system THINKS, never what it is ALLOWED to do" was
    a convention held in prose, with nothing in the code able to test it.

    The members are the layers that governed variables really reach today, read
    off `POLICY_FIELDS` — not an aspiration. There is deliberately no member for
    capabilities, entitlements, credentials, authorization or provider access:
    the absence is the invariant, and `test_no_policy_field_can_grant_capability`
    fails if a field is ever added that would need one.
    """

    # Thresholds the reasoning engines compare against (risk/meta detectors).
    REASONING = "reasoning"
    # What reaches the founder: brief size and confidence floor.
    DECISION = "decision"
    # Operational limits only — how many times a FAILED action retries. NOT
    # whether an action may run: that is Execution's authorization path, which
    # Adaptive has no field for and must never acquire one.
    EXECUTION = "execution"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    default: float
    min: float
    max: float
    kind: str  # "int" | "float"
    target: PolicyTarget
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
    "commitment_slipping_days": FieldSpec(5.0, 1.0, 60.0, "float", PolicyTarget.REASONING),
    "relationship_cooling_days": FieldSpec(7.0, 1.0, 60.0, "float", PolicyTarget.REASONING),
    "project_stalled_days": FieldSpec(10.0, 1.0, 90.0, "float", PolicyTarget.REASONING),
    "customer_drop_days": FieldSpec(14.0, 1.0, 120.0, "float", PolicyTarget.REASONING),
    "meeting_overload_count": FieldSpec(5, 1, 50, "int", PolicyTarget.REASONING),
    "leverage_min_people": FieldSpec(2, 1, 20, "int", PolicyTarget.REASONING),
    "brief_max_items": FieldSpec(7, 1, 25, "int", PolicyTarget.DECISION),
    "brief_min_confidence": FieldSpec(0.45, 0.0, 0.95, "float", PolicyTarget.DECISION),
    "execution_retry_limit": FieldSpec(5, 1, 10, "int", PolicyTarget.EXECUTION),
}


def fields_for(target: PolicyTarget) -> tuple[str, ...]:
    """The governed variables that tune one layer, in stable order."""
    return tuple(sorted(n for n, s in POLICY_FIELDS.items() if s.target is target))

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
    """Full policy = defaults overlaid by any provided fields. FAILS CLOSED.

    This is the last thing between a stored policy row and the running system —
    `runtime_policy.resolve_policy` returns exactly what it produces — so it is
    the right place to distrust the row.

    Three ways a field is refused, each falling back to the code default rather
    than to the stored value:

      * UNKNOWN name — dropped, as before. A field this build does not govern
        cannot be made to mean anything by storing it.
      * NON-NUMERIC value — a string or None where a number belongs is a
        corrupted row, not a policy.
      * OUT OF BOUNDS — this is the one that was missing. `cast()` rounds; it
        does not clamp, so a row holding `brief_max_items = 10_000` (a bug, a
        migration, a hand-edit) was cast to 10000 and served. Proposals are
        bounds-checked when written and again at approval, so no legitimate row
        reaches here out of range — which is exactly why one that does should be
        treated as damage and ignored.

    Falling back to the default rather than clamping is deliberate: clamping
    invents a value nobody approved, while the default is the value the system
    ships with and the one `kill_switch_payload()` restores.
    """
    out = dict(DEFAULT_POLICY)
    if not isinstance(payload, dict):
        # A JSONB column can hold a list, a string or a number as legally as an
        # object. `(payload or {}).items()` raised AttributeError on all three,
        # which would have taken down the brief rather than degrading it — the
        # opposite of failing closed.
        return out
    for name, value in payload.items():
        spec = POLICY_FIELDS.get(name)
        if spec is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not (spec.min <= numeric <= spec.max) or numeric != numeric:  # NaN-safe
            continue
        out[name] = spec.cast(numeric)
    return out


def validate_payload(payload: dict[str, Any]) -> None:
    for field, value in payload.items():
        validate_value(field, value)
