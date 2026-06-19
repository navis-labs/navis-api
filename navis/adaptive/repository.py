"""AdaptiveRepository — reads ONLY Intelligence + Learning; writes adaptive_*.

Upstream dependency is strictly {Intelligence, Learning}. This module imports
navis.intelligence.models and navis.learning.models and nothing else from the
pipeline. It never touches Signals/Memory/State/Reasoning/Decision/Execution/
Observation.
"""
from __future__ import annotations

import uuid
from typing import Any, Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.adaptive.contract import (
    AdaptiveView,
    AggRow,
    InsightRow,
    PatternRow,
    SnapshotStatus,
)
from navis.adaptive.models import PolicyApproval, PolicyProposal, PolicySnapshot
from navis.adaptive.policy import merge_defaults
from navis.intelligence.models import IntelligenceInsight
from navis.learning.models import AggregateDimension, LearningAggregate, LearningPattern


class AdaptiveRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ---- read model (Intelligence + Learning only) ----

    async def _aggs(self, org_id: uuid.UUID, dimension: str) -> list[AggRow]:
        rows = (
            await self.db.scalars(
                select(LearningAggregate).where(
                    LearningAggregate.organization_id == org_id,
                    LearningAggregate.dimension == dimension,
                )
            )
        ).all()
        return [AggRow(r.key, r.successes, r.failures, r.neutrals, r.total) for r in rows]

    async def build_view(self, org_id: uuid.UUID) -> AdaptiveView:
        insight_rows = (
            await self.db.scalars(
                select(IntelligenceInsight).where(IntelligenceInsight.organization_id == org_id)
            )
        ).all()
        insights = [
            InsightRow(
                insight_id=str(r.insight_id),
                insight_type=r.insight_type,
                key=r.key,
                evidence=r.evidence or {},
                sample_size=r.sample_size,
                confidence=r.confidence,
                supporting_learning_event_ids=list(r.supporting_learning_event_ids or []),
                learning_version=r.learning_version,
            )
            for r in insight_rows
        ]
        adaptive_version = max((i.learning_version for i in insights), default=0)
        as_of = max((r.created_at for r in insight_rows if r.created_at is not None), default=None)

        patterns = [
            PatternRow(p.signal_id, p.kind, p.success_rate, p.samples)
            for p in (
                await self.db.scalars(
                    select(LearningPattern).where(LearningPattern.organization_id == org_id)
                )
            ).all()
        ]

        active = await self.get_active_payload(org_id)
        return AdaptiveView(
            org_id=org_id,
            adaptive_version=adaptive_version,
            as_of=as_of,
            active_policy=active,
            insights=insights,
            signal_aggs=await self._aggs(org_id, AggregateDimension.SIGNAL.value),
            decision_aggs=await self._aggs(org_id, AggregateDimension.DECISION.value),
            outcome_aggs=await self._aggs(org_id, AggregateDimension.OUTCOME.value),
            patterns=patterns,
        )

    # ---- policy resolution ----

    async def get_active_snapshot(self, org_id: uuid.UUID) -> PolicySnapshot | None:
        return await self.db.scalar(
            select(PolicySnapshot).where(
                PolicySnapshot.organization_id == org_id,
                PolicySnapshot.status == SnapshotStatus.ACTIVE.value,
            )
        )

    async def get_active_payload(self, org_id: uuid.UUID) -> dict[str, Any]:
        snap = await self.get_active_snapshot(org_id)
        return merge_defaults(snap.payload if snap else None)

    async def get_snapshot(self, policy_id: uuid.UUID) -> PolicySnapshot | None:
        return await self.db.get(PolicySnapshot, policy_id)

    # ---- append-only writes (idempotent) ----

    async def append_proposal(self, values: dict[str, Any]) -> bool:
        stmt = pg_insert(PolicyProposal).values(**values).on_conflict_do_nothing(
            index_elements=["proposal_id"]
        )
        return (await self.db.execute(stmt)).rowcount == 1

    async def append_snapshot(self, values: dict[str, Any]) -> bool:
        stmt = pg_insert(PolicySnapshot).values(**values).on_conflict_do_nothing(
            index_elements=["policy_id"]
        )
        return (await self.db.execute(stmt)).rowcount == 1

    async def append_approval(self, values: dict[str, Any]) -> bool:
        stmt = pg_insert(PolicyApproval).values(**values).on_conflict_do_nothing(
            index_elements=["approval_id"]
        )
        return (await self.db.execute(stmt)).rowcount == 1

    async def set_proposal_status(self, proposal_id: uuid.UUID, status: str, decided_by=None) -> None:
        prop = await self.db.get(PolicyProposal, proposal_id)
        if prop is not None:
            prop.status = status
            if decided_by is not None:
                prop.decided_by = decided_by

    async def set_snapshot_status(self, policy_id: uuid.UUID, status: str, *, activated_at=None, approved_by=None) -> None:
        snap = await self.db.get(PolicySnapshot, policy_id)
        if snap is not None:
            snap.status = status
            if activated_at is not None:
                snap.activated_at = activated_at
            if approved_by is not None:
                snap.approved_by = approved_by

    async def supersede_active(self, org_id: uuid.UUID) -> None:
        snap = await self.get_active_snapshot(org_id)
        if snap is not None:
            snap.status = SnapshotStatus.SUPERSEDED.value

    async def get_proposal(self, proposal_id: uuid.UUID) -> PolicyProposal | None:
        return await self.db.get(PolicyProposal, proposal_id)

    async def get_proposals(self, org_id: uuid.UUID) -> Sequence[PolicyProposal]:
        return (
            await self.db.scalars(
                select(PolicyProposal)
                .where(PolicyProposal.organization_id == org_id)
                .order_by(PolicyProposal.adaptive_version.desc())
            )
        ).all()

    async def next_policy_version(self, org_id: uuid.UUID) -> int:
        current = await self.db.scalar(
            select(func.max(PolicySnapshot.policy_version)).where(
                PolicySnapshot.organization_id == org_id
            )
        )
        return int(current or 0) + 1

    # ---- replay: drop the deterministic projection only ----

    async def delete_projection(self, org_id: uuid.UUID) -> None:
        """Remove proposals and DRAFT snapshots (the deterministic projection).

        Active/superseded snapshots and approvals are committed governance history
        — human acts — and are intentionally NOT regenerable by replay.
        """
        await self.db.execute(
            delete(PolicyProposal).where(PolicyProposal.organization_id == org_id)
        )
        await self.db.execute(
            delete(PolicySnapshot).where(
                PolicySnapshot.organization_id == org_id,
                PolicySnapshot.status == SnapshotStatus.DRAFT.value,
            )
        )

    async def flush(self) -> None:
        await self.db.flush()
