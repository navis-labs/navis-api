"""In-memory fakes for Adaptive V1 (append-only governance over an AdaptiveView)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from navis.adaptive.contract import AdaptiveView, InsightRow, SnapshotStatus
from navis.adaptive.policy import merge_defaults

ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")
FOUNDER = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
AS_OF = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
AT = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)


def org_insight(org_id=ORG_A, *, positive=14, negative=6, learning_version=20):
    return InsightRow(
        insight_id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"{org_id}|org")),
        insight_type="organizational_intelligence",
        key="outcome_profile",
        evidence={"positive": positive, "negative": negative, "patterns": 2},
        sample_size=positive + negative,
        confidence=0.2,
        supporting_learning_event_ids=[f"e{i}" for i in range(5)],
        learning_version=learning_version,
    )


def make_view(
    *,
    org_id=ORG_A,
    insights=None,
    active_policy=None,
    as_of=AS_OF,
    adaptive_version=20,
) -> AdaptiveView:
    rows = insights if insights is not None else [org_insight(org_id)]
    version = max((i.learning_version for i in rows), default=0) if rows else adaptive_version
    return AdaptiveView(
        org_id=org_id,
        adaptive_version=version if rows else 0,
        as_of=as_of,
        active_policy=merge_defaults(active_policy),
        insights=rows,
    )


class FakeAdaptiveRepo:
    """Holds a fixed AdaptiveView + append-only proposals/snapshots/approvals."""

    def __init__(self, view: AdaptiveView) -> None:
        self.view = view
        self.proposals: dict[uuid.UUID, SimpleNamespace] = {}
        self.snapshots: dict[uuid.UUID, SimpleNamespace] = {}
        self.approvals: dict[uuid.UUID, SimpleNamespace] = {}

    async def build_view(self, org_id) -> AdaptiveView:
        assert org_id == self.view.org_id
        return self.view

    async def get_active_snapshot(self, org_id):
        for s in self.snapshots.values():
            if s.organization_id == org_id and s.status == SnapshotStatus.ACTIVE.value:
                return s
        return None

    async def get_active_payload(self, org_id) -> dict[str, Any]:
        snap = await self.get_active_snapshot(org_id)
        return merge_defaults(snap.payload if snap else None)

    async def get_snapshot(self, policy_id):
        return self.snapshots.get(policy_id)

    async def append_proposal(self, values) -> bool:
        pid = values["proposal_id"]
        if pid in self.proposals:
            return False
        self.proposals[pid] = SimpleNamespace(**values)
        return True

    async def append_snapshot(self, values) -> bool:
        pid = values["policy_id"]
        if pid in self.snapshots:
            return False
        self.snapshots[pid] = SimpleNamespace(**values)
        return True

    async def append_approval(self, values) -> bool:
        aid = values["approval_id"]
        if aid in self.approvals:
            return False
        self.approvals[aid] = SimpleNamespace(**values)
        return True

    async def set_proposal_status(self, proposal_id, status, decided_by=None):
        p = self.proposals.get(proposal_id)
        if p:
            p.status = status
            if decided_by is not None:
                p.decided_by = decided_by

    async def set_snapshot_status(self, policy_id, status, *, activated_at=None, approved_by=None):
        s = self.snapshots.get(policy_id)
        if s:
            s.status = status
            if activated_at is not None:
                s.activated_at = activated_at
            if approved_by is not None:
                s.approved_by = approved_by

    async def supersede_active(self, org_id):
        s = await self.get_active_snapshot(org_id)
        if s:
            s.status = SnapshotStatus.SUPERSEDED.value

    async def get_proposal(self, proposal_id):
        return self.proposals.get(proposal_id)

    async def get_proposals(self, org_id):
        return [p for p in self.proposals.values() if p.organization_id == org_id]

    async def next_policy_version(self, org_id):
        versions = [s.policy_version for s in self.snapshots.values() if s.organization_id == org_id]
        return (max(versions) if versions else 0) + 1

    async def delete_projection(self, org_id):
        self.proposals = {k: v for k, v in self.proposals.items() if v.organization_id != org_id}
        self.snapshots = {
            k: v for k, v in self.snapshots.items()
            if not (v.organization_id == org_id and v.status == SnapshotStatus.DRAFT.value)
        }

    async def flush(self):
        pass


def proposal_snapshot(rows) -> set[tuple]:
    """Order-independent fingerprint of proposal rows (SimpleNamespace)."""
    return {
        (
            str(r.proposal_id),
            r.adaptive_version,
            tuple(sorted((c["field"], c["from"], c["to"]) for c in r.proposed_changes)),
            tuple(sorted(r.proposed_payload.items())),
            tuple(r.supporting_insight_ids),
            tuple(r.supporting_learning_ids),
            r.created_at,
        )
        for r in rows
    }
