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


def kill_switch_payload() -> dict[str, Any]:
    """The safe baseline: exactly the code defaults (no learned drift)."""
    return dict(DEFAULT_POLICY)


__all__ = [
    "APPROVER_ROLES",
    "NotAuthorized",
    "PolicyViolation",
    "require_approver",
    "validate_proposal",
    "kill_switch_payload",
]
