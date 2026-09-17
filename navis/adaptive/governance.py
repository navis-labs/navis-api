"""Governance + safety rails. Adaptive proposes; only a founder activates.

- Approval is restricted to FOUNDER (the org's governing seat). NOTE: the mission
  named FOUNDER/OWNER, but identity.Role has no OWNER — the closest governing role
  that exists is FOUNDER, so that is the gate. (Grounded in identity/models.py.)
- Every change is independently re-validated against bounds + magnitude caps here,
  so a malformed proposal can never be activated even if generation is bypassed.
- The kill switch reverts an org to the code-default policy.
"""
from __future__ import annotations

from typing import Any

from navis.adaptive.contract import ProposalDraft
from navis.adaptive.policy import DEFAULT_POLICY, PolicyViolation, validate_change

# Only these org roles may approve/activate/rollback policy. OWNER is absent from
# identity.Role; FOUNDER is the governing seat that exists.
APPROVER_ROLES = frozenset({"founder"})


class NotAuthorized(PermissionError):
    pass


def require_approver(role: str) -> None:
    if role not in APPROVER_ROLES:
        raise NotAuthorized(f"role '{role}' may not govern policy (requires founder)")


def validate_proposal(draft: ProposalDraft, active_policy: dict[str, Any]) -> None:
    """Independent safety check: every change within bounds + within the cap."""
    for change in draft.changes:
        old = active_policy.get(change.field, change.from_value)
        validate_change(change.field, old, change.to_value)


def validate_changes(changes: list[dict[str, Any]], active_policy: dict[str, Any]) -> None:
    """The same gate, applied to a STORED proposal at activation time.

    `validate_proposal` takes a live `ProposalDraft` and so can only run when a
    proposal is generated. Activation happens later, against a policy baseline
    that may have moved (a rollback or kill switch in between), and the 25%
    magnitude cap is defined relative to that baseline — so the check has to be
    repeatable from the persisted `proposed_changes` alone.

    Fails closed on a malformed record: a change missing its field or target
    value cannot be shown to be safe, so it is refused rather than skipped.
    """
    for change in changes:
        if not isinstance(change, dict):
            raise PolicyViolation(f"malformed change record: {change!r}")
        field = change.get("field")
        to_value = change.get("to")
        if not isinstance(field, str) or to_value is None:
            raise PolicyViolation(f"malformed change record: {change!r}")
        old = active_policy.get(field, change.get("from"))
        if old is None:
            raise PolicyViolation(f"no baseline for {field!r}")
        validate_change(field, old, to_value)


def kill_switch_payload() -> dict[str, Any]:
    """The safe baseline: exactly the code defaults (no learned drift)."""
    return dict(DEFAULT_POLICY)


__all__ = [
    "APPROVER_ROLES",
    "NotAuthorized",
    "PolicyViolation",
    "require_approver",
    "validate_changes",
    "validate_proposal",
    "kill_switch_payload",
]
