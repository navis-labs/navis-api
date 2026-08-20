"""AdaptiveRepository — reads ONLY Intelligence + Learning; writes adaptive_*.

Upstream dependency is strictly {Intelligence, Learning}. This module imports
navis.intelligence.repository and navis.learning.models and nothing else from the
pipeline. It never touches Signals/Memory/State/Reasoning/Decision/Execution/
Observation.

Intelligence is read THROUGH ITS OWN REPOSITORY, not with a query of our own.
`intelligence_insights` is append-only, so every projection an org has ever run is
still in the table and a bare `select()` returns all of them at once — which is
what this module used to do. `IntelligenceRepository.get_insights` owns the rule
that decides which projection is current (newest per key for Learning-derived
insights, the newest whole snapshot for reality-derived ones), and duplicating
that rule here would mean Adaptive folding superseded conclusions into policy
while `GET /intelligence/insights` served the corrected ones. One read boundary,
one answer.
"""
from __future__ import annotations

import uuid
from typing import Any, Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.adaptive.contract import (
    AdaptiveView,
    InsightRow,
    LearningEvidence,
    OutcomeEvidence,
    SnapshotStatus,
)
from navis.adaptive.models import PolicyApproval, PolicyProposal, PolicySnapshot
from navis.adaptive.policy import merge_defaults
from navis.intelligence.repository import IntelligenceRepository
from navis.learning.metrics import learning_snapshot


class AdaptiveRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ---- read model (Intelligence + Learning only) ----

    # THE LEARNING BOUNDARY. Adaptive reads NO Learning table. It reads
    # Intelligence through `IntelligenceRepository.get_insights`, and when it
    # eventually needs raw learned evidence that an insight does not carry, that
    # evidence arrives through `navis.learning.metrics.learning_snapshot` — the
    # same contract Intelligence and ML consume — and never through a private
    # query against `learning_*`.
    #
    # No such call exists yet ON PURPOSE: candidate generation is a later phase,
    # and a read with no consumer is the exact dead weight the two removals below
    # describe. The boundary is declared and pinned by a test; it is not
    # pre-wired.
    #
    # `_aggs()` USED TO BE HERE, and it is gone rather than kept "for later".
    #
    # It ran three unbounded `learning_aggregates` reads per view — SIGNAL,
    # DECISION and OUTCOME — and nothing in this package ever read the fields
    # they filled. The DECISION one was the worst of them: `learning_aggregates`
    # holds one decision row per decision the organization has ever made, so on
    # a 100k-decision org this fetched 100k rows, built 100k `AggRow` objects,
    # and threw all of them away.
    #
    # Deleting it also removed Adaptive's only direct read of a Learning TABLE.
    # What Adaptive needs from Learning now arrives through
    # `navis.learning.metrics.learning_snapshot` — the contract Intelligence and
    # ML already consume — rather than through a private copy of Learning's
    # storage layout.

    @staticmethod
    def _evidence(org_id: uuid.UUID, snap) -> LearningEvidence:
        """Flatten Learning's snapshot to the evidence Adaptive may reason about.

        Deliberately NARROW. The snapshot also carries signal and decision
        aggregates and `learning_patterns`; none of them are passed on, because
        their keys are opaque uuids and a rule cannot say what tuning a threshold
        for `sig-8f3c…` would MEAN. Outcome types are named, so they are the only
        learned evidence a policy rule can act on without guessing — the same
        limitation `proposals.py` has documented since V1, now expressed as a
        boundary rather than a comment.
        """
        outcomes = []
        for surface, aggs in sorted(snap.outcomes_by_surface().items()):
            for agg in aggs:
                outcomes.append(
                    OutcomeEvidence(
                        outcome_type=agg.key,
                        surface=surface,
                        successes=agg.successes,
                        failures=agg.failures,
                        neutrals=agg.neutrals,
                        total=agg.total,
                        # The events behind this counter, so a candidate can cite
                        # what caused it rather than asserting it.
                        supporting_learning_event_ids=tuple(
                            snap.lineage.get(agg.key, [])
                        ),
                    )
                )
        return LearningEvidence(
            org_id=org_id,
            learning_version=snap.learning_version,
            as_of=snap.as_of,
            outcomes=tuple(sorted(outcomes, key=lambda o: o.outcome_type)),
        )

    async def build_view(self, org_id: uuid.UUID) -> AdaptiveView:
        # The org's CURRENT insights — the same rows GET /intelligence/insights
        # serves. Org isolation, the current/superseded rule and the ordering all
        # come from Intelligence's own boundary; nothing about them is restated
        # here. See the module docstring.
        insight_rows = await IntelligenceRepository(self.db).get_insights(org_id)
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
        evidence = self._evidence(org_id, await learning_snapshot(self.db, org_id))
        # WHICH CLOCK VERSIONS A PROPOSAL — the high-water mark of BOTH inputs.
        #
        # `proposal_id` is `uuid5(org, adaptive_version)` and proposals are
        # appended with ON CONFLICT DO NOTHING, so any evidence change that does
        # NOT move this number is silently discarded: the write is a no-op and
        # the older proposal stands.
        #
        # A max over insights alone was safe while insights were the only input.
        # They no longer are — `_judgement_changes` folds Learning's own counters
        # — so the number has to move when EITHER moves:
        #
        #   * Intelligence current  -> both agree, nothing changes;
        #   * Intelligence lagging  -> Learning's count is higher and wins, so a
        #     new rejection cannot be lost to a stale id while the projection
        #     catches up;
        #   * no insights at all    -> Learning's count is the only clock, which
        #     is what lets a purely learned candidate exist (`service.propose`
        #     short-circuits on `adaptive_version == 0`);
        #   * insights but no learning events (fixtures / an imported
        #     projection) -> the insight max wins and behaviour is unchanged.
        #
        # Monotonic in both arguments, so an id is never reused for weaker
        # evidence.
        adaptive_version = max(
            max((i.learning_version for i in insights), default=0),
            evidence.learning_version,
        )
        as_of = max((r.created_at for r in insight_rows if r.created_at is not None), default=None)

        active = await self.get_active_payload(org_id)
        return AdaptiveView(
            org_id=org_id,
            adaptive_version=adaptive_version,
            as_of=as_of,
            active_policy=active,
            insights=insights,
            learning=evidence,
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
