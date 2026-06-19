"""AdaptiveService — propose (deterministic) + govern (human-gated).

Projection (propose) is a pure, deterministic fold over Intelligence: same
Intelligence => same proposal id => same rows (idempotent append-only). Governance
acts (approve/reject/rollback/kill) are human decisions; their timestamps enter as
an explicit `at` argument from the edge, so this module calls no wall clock.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from navis.adaptive import governance, proposals
from navis.adaptive.contract import (
    ProposalDraft,
    ProposalStatus,
    SnapshotStatus,
    compute_approval_id,
    compute_snapshot_id,
)
from navis.adaptive.policy import merge_defaults
from navis.adaptive.repository import AdaptiveRepository


def _change_to_dict(c) -> dict[str, Any]:
    return {
        "field": c.field,
        "from": c.from_value,
        "to": c.to_value,
        "direction": c.direction,
        "capped": c.capped,
        "reason": c.reason,
        "supporting_insight_ids": c.supporting_insight_ids,
        "supporting_learning_ids": c.supporting_learning_ids,
    }


def proposal_values(draft: ProposalDraft, policy_id: uuid.UUID) -> dict[str, Any]:
    return {
        "proposal_id": draft.proposal_id,
        "organization_id": draft.org_id,
        "adaptive_version": draft.adaptive_version,
        "status": ProposalStatus.PENDING_APPROVAL.value,
        "proposed_changes": [_change_to_dict(c) for c in draft.changes],
        "proposed_payload": draft.proposed_payload,
        "supporting_insight_ids": draft.supporting_insight_ids,
        "supporting_learning_ids": draft.supporting_learning_ids,
        "policy_id": policy_id,
        "created_at": draft.as_of,
    }


def draft_snapshot_values(draft: ProposalDraft, policy_id: uuid.UUID) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "organization_id": draft.org_id,
        "policy_version": draft.adaptive_version,
        "status": SnapshotStatus.DRAFT.value,
        "payload": draft.proposed_payload,
        "source_intelligence_ids": draft.supporting_insight_ids,
        "source_learning_event_ids": draft.supporting_learning_ids,
        "created_by": "adaptive",
        "created_at": draft.as_of,
    }


class AdaptiveService:
    def __init__(self, repo: AdaptiveRepository) -> None:
        self.repo = repo

    # ---- deterministic projection ----

    async def propose(self, org_id: uuid.UUID) -> dict[str, Any] | None:
        view = await self.repo.build_view(org_id)
        if view.adaptive_version == 0:
            return None  # nothing learned/understood yet
        draft = proposals.propose(view)
        if draft is None:
            return None
        governance.validate_proposal(draft, view.active_policy)  # independent safety gate
        policy_id = compute_snapshot_id(org_id, draft.adaptive_version, kind="draft")
        pvals = proposal_values(draft, policy_id)
        await self.repo.append_proposal(pvals)
        await self.repo.append_snapshot(draft_snapshot_values(draft, policy_id))
        await self.repo.flush()
        return pvals

    # ---- governance (human-gated; `at` supplied by the edge) ----

    async def approve(self, org_id: uuid.UUID, proposal_id: uuid.UUID, *, approver_id: uuid.UUID, role: str, at: datetime):
        governance.require_approver(role)
        prop = await self.repo.get_proposal(proposal_id)
        if prop is None or prop.organization_id != org_id:
            raise LookupError("proposal not found")
        if prop.status != ProposalStatus.PENDING_APPROVAL.value:
            raise ValueError(f"proposal not pending (status={prop.status})")

        await self.repo.append_approval({
            "approval_id": compute_approval_id(proposal_id, approver_id, "approved"),
            "organization_id": org_id,
            "proposal_id": proposal_id,
            "decision": "approved",
            "approved_by": approver_id,
            "approved_at": at,
        })
        await self.repo.set_proposal_status(proposal_id, ProposalStatus.APPROVED.value, decided_by=approver_id)
        await self.repo.supersede_active(org_id)
        await self.repo.set_snapshot_status(
            prop.policy_id, SnapshotStatus.ACTIVE.value, activated_at=at, approved_by=approver_id
        )
        await self.repo.flush()
        return prop

    async def reject(self, org_id: uuid.UUID, proposal_id: uuid.UUID, *, approver_id: uuid.UUID, role: str, at: datetime):
        governance.require_approver(role)
        prop = await self.repo.get_proposal(proposal_id)
        if prop is None or prop.organization_id != org_id:
            raise LookupError("proposal not found")
        await self.repo.append_approval({
            "approval_id": compute_approval_id(proposal_id, approver_id, "rejected"),
            "organization_id": org_id,
            "proposal_id": proposal_id,
            "decision": "rejected",
            "approved_by": approver_id,
            "approved_at": at,
        })
        await self.repo.set_proposal_status(proposal_id, ProposalStatus.REJECTED.value, decided_by=approver_id)
        if prop.policy_id is not None:
            await self.repo.set_snapshot_status(prop.policy_id, SnapshotStatus.REJECTED.value)
        await self.repo.flush()
        return prop

    async def _activate_payload(self, org_id, payload, *, created_by, approver_id, role, at, kind):
        """Append a fresh ACTIVE snapshot from an explicit payload (rollback/kill)."""
        governance.require_approver(role)
        full = merge_defaults(payload)
        version = await self.repo.next_policy_version(org_id)
        policy_id = compute_snapshot_id(org_id, version, kind=kind)
        await self.repo.supersede_active(org_id)
        await self.repo.append_snapshot({
            "policy_id": policy_id,
            "organization_id": org_id,
            "policy_version": version,
            "status": SnapshotStatus.ACTIVE.value,
            "payload": full,
            "source_intelligence_ids": [],
            "source_learning_event_ids": [],
            "created_by": created_by,
            "approved_by": approver_id,
            "created_at": at,
            "activated_at": at,
        })
        await self.repo.flush()
        return policy_id

    async def rollback(self, org_id: uuid.UUID, target_policy_id: uuid.UUID, *, approver_id, role, at):
        target = await self.repo.get_snapshot(target_policy_id)
        if target is None or target.organization_id != org_id:
            raise LookupError("target policy not found")
        return await self._activate_payload(
            org_id, target.payload, created_by=f"rollback:{target_policy_id}",
            approver_id=approver_id, role=role, at=at, kind="rollback",
        )

    async def kill_switch(self, org_id: uuid.UUID, *, approver_id, role, at):
        return await self._activate_payload(
            org_id, governance.kill_switch_payload(), created_by="kill_switch",
            approver_id=approver_id, role=role, at=at, kind="killswitch",
        )

    async def get_active_policy(self, org_id: uuid.UUID) -> dict[str, Any]:
        return await self.repo.get_active_payload(org_id)
