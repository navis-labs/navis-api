"""DecisionService — convert ReasoningSignals into a persisted DailyBrief.

Decision consumes ReasoningSignal[] ONLY. It does not build State, run Reasoning,
or touch Memory/Signals/Execution — the pipeline assembly (State V2 -> Reasoning V2
-> Decision) lives in orchestration. This service ranks, persists (idempotent
upsert on the deterministic brief id), and announces BRIEF_READY.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import desc, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.decisions.base import (
    IntelligenceView,
    check_decision_provenance,
    compute_brief_id,
    obligation_counts,
)
from navis.decisions.models import DailyBrief
from navis.decisions.what_matters_today import WhatMattersTodayEngine
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.reasoning.v2.contract import ReasoningSignal

logger = get_logger("decisions")


class DecisionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.engine = WhatMattersTodayEngine()

    async def decide(
        self,
        org_id: uuid.UUID,
        signals: list[ReasoningSignal],
        *,
        reasoning_version: int,
        state_watermark: str | None,
        as_of: datetime | None,
        policy: dict | None = None,
        intelligence: IntelligenceView | None = None,
    ) -> DailyBrief:
        """Rank this org's signals into a brief.

        `intelligence` is what the organization currently knows about itself,
        projected to primitives by orchestration (Decision never queries
        Intelligence — see `navis.decisions.base.IntelligenceView`). Keyword-only
        with a `None` default so every existing caller keeps working and keeps
        producing byte-identical briefs.

        The engine CORROBORATES with it: a fact that is about a signal (it names
        the signal or its entity) nudges that signal's rank, bounded by the
        largest nudge the category system already applies, and is recorded on the
        decision as `supporting_insight_ids`. Decision still owns priority,
        recommendation, policy evaluation and approval; Intelligence still owns
        facts and never manufactures a decision. An absent or empty view produces
        byte-identical output to before. See `what_matters_today`.

        `decision_id` and `brief_id` do not change: identity is (org, reasoning
        version, watermark, signal), and the view is not part of it.
        """
        if intelligence is not None and intelligence.org_id != org_id:
            # Fail loud rather than rank one organization's signals against
            # another's understanding of itself. Nothing here would collide —
            # every id hashes org_id — so the corruption would be silent
            # cross-tenant attribution, which is worse than a crash.
            raise ValueError(
                f"intelligence view is for org {intelligence.org_id}, not {org_id}"
            )
        # Runtime policy is injected by orchestration (Adaptive → active snapshot).
        # Absent a policy, the engine's own defaults apply — identical behavior.
        engine = self.engine
        if policy and ("brief_max_items" in policy or "brief_min_confidence" in policy):
            engine = WhatMattersTodayEngine(
                max_items=int(policy.get("brief_max_items", self.engine.max_items)),
                min_confidence=float(policy.get("brief_min_confidence", self.engine.min_confidence)),
            )
        decisions = engine.decide(
            signals,
            org_id=org_id,
            reasoning_version=reasoning_version,
            state_watermark=state_watermark,
            as_of=as_of,
            intelligence=intelligence,
        )
        brief_id = compute_brief_id(org_id, reasoning_version, state_watermark)
        payload = [d.to_dict() for d in decisions]
        # WHICH GOVERNING POLICY PRODUCED THIS BRIEF.
        #
        # `brief_max_items` and `brief_min_confidence` change what a founder
        # sees, and until now nothing recorded that they had been applied: a
        # brief built under an approved Adaptive policy was byte-indistinguishable
        # from one built under code defaults, so "why did Monday show four items
        # and Tuesday nine?" had no answer in the data.
        #
        # Read off the resolved policy rather than added as a parameter —
        # `runtime_policy.resolve_policy` already stamps it and every caller
        # passes that dict through — so no signature changes and no new plumbing.
        # `None` means "no active policy", which is the honest value for a brief
        # that ran on defaults and stays distinguishable from version 0.
        policy_version = policy.get("policy_version") if policy else None

        # Idempotent upsert: replaying the same Reasoning rewrites the same brief.
        # `policy_version` is in the update set, not the key: the brief id is
        # (org, reasoning_version, state_watermark) and a policy change must
        # REPLACE that generation's brief rather than mint a parallel one.
        stmt = (
            pg_insert(DailyBrief)
            .values(
                id=brief_id,
                organization_id=org_id,
                reasoning_version=reasoning_version,
                state_watermark=state_watermark,
                as_of=as_of,
                decisions=payload,
                policy_version=policy_version,
            )
            .on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "decisions": payload,
                    "as_of": as_of,
                    "state_watermark": state_watermark,
                    "policy_version": policy_version,
                },
            )
        )
        await self.db.execute(stmt)

        await event_bus.publish(
            DomainEvent(
                type=EventType.BRIEF_READY,
                org_id=org_id,
                source="decisions",
                payload={"brief_id": str(brief_id), "decision_count": len(decisions)},
            )
        )
        # `intelligence_facts` is how production shows whether the join is live:
        # a brief built with a view Decision does not yet rank on is otherwise
        # indistinguishable from one built without it.
        logger.info(
            "brief %s org=%s decisions=%d intelligence_facts=%d",
            brief_id, org_id, len(decisions),
            len(intelligence.facts) if intelligence else 0,
        )
        return await self.db.get(DailyBrief, brief_id)

    async def obligations(self, brief: DailyBrief) -> dict:
        """How much this brief is a window ONTO, resolved from Reasoning's record.

        THE DEFECT THIS CLOSES. A brief shows at most `brief_max_items`
        decisions (seven by default), ranked across every category and deduped
        by entity. Nothing in the response said so, so the only count a surface
        could form was `len(decisions)` — and a Today tile that reads "Waiting
        on you: 2" off a brief holding two waiting decisions and five risks is
        reporting the size of a display window as if it were the number of
        people waiting for an answer. Every other number on that surface was
        worse: with no backend count at all, the frontend fell back to counting
        rendered cards, and one tile counted a thread feed that is not served.

        The real answer is already persisted. `orchestration.brief` calls
        `persist_signals` on EVERY signal of the pass — not just the ones that
        became decisions — and a brief's identity IS
        (org, reasoning_version, state_watermark), so the same tuple resolves
        the exact reasoning pass this brief was ranked from.

        `available` is the load-bearing field. A brief built before
        `reasoning_signals` existed, or one whose pass was never persisted,
        resolves to no rows — and reporting that as `{"waiting_on_user_reply":
        0}` would be Navis stating "nobody is waiting on you" on the strength
        of a table it could not read. It is the same distinction
        `state.v2.models` draws between False and None, and the one the
        frontend already understands as `threadActivityAvailable`. When
        `available` is false the counts carry NO information; when it is true, a
        type absent from `by_signal_type` is genuinely zero.

        Read-only, one indexed query, and it never fabricates: `shown` is the
        brief's own decision list and `by_signal_type` counts stored rows.
        """
        from navis.reasoning.store import signal_type_counts_at

        emitted = await signal_type_counts_at(
            self.db,
            brief.organization_id,
            reasoning_version=brief.reasoning_version,
            state_watermark=brief.state_watermark,
        )
        counts = obligation_counts(emitted)
        return {
            "available": bool(emitted),
            "reasoning_version": brief.reasoning_version,
            "state_watermark": brief.state_watermark,
            "by_signal_type": counts,
            "total": sum(counts.values()),
            # How many of them this brief is actually showing. The gap between
            # `total` and `shown` is the honest "and N more" a surface needs in
            # order not to imply the list is exhaustive.
            "shown": len(brief.decisions or []),
        }

    async def provenance(self, brief: DailyBrief) -> dict[str, dict]:
        """Prove, per decision, that its reasoning is reproducible AT ITS OWN SNAPSHOT.

        THE P0-3 CHECK. A decision carries the values the engines produced during
        the pass it was ranked in; its `reasoning_signal_ids` pointer resolves to
        a stored conclusion. Those used to be the same row, and that row kept only
        its FIRST detection — so a decision could publish 11.85 days / 0.8951 while
        the signal it cites recorded 7.16 / 0.7386. Nothing detected it, because
        nothing ever compared them.

        `reasoning_signal_observations` makes the comparison possible, and this
        performs it: resolve each cited conclusion AT `brief.state_watermark`, then
        check field by field. ONE query for the whole brief, not one per decision.

        FAILS CLOSED, in both directions. `observations_at` returns nothing for a
        conclusion never recorded at this snapshot — no older observation, no
        latest, no fallback to the identity row's own columns — and a decision
        with no resolvable observation is reported `verified: false` with the
        reason. It is never dropped and never silently presented as provenanced:
        a brief that cannot back its own reasoning must say so, because a founder
        acting on it is the one who bears the cost.

        Keyed by `decision_id` so a caller can attach the result without
        re-deriving the pairing.
        """
        from navis.reasoning.store import observations_at

        decisions = list(brief.decisions or [])
        if not decisions:
            return {}
        cited = [
            (d.get("reasoning_signal_ids") or [None])[0]
            for d in decisions
        ]
        observations = await observations_at(
            self.db,
            brief.organization_id,
            [c for c in cited if c],
            state_watermark=brief.state_watermark,
        )
        out: dict[str, dict] = {}
        for decision in decisions:
            signal_id = (decision.get("reasoning_signal_ids") or [None])[0]
            check = check_decision_provenance(
                decision, observations.get(str(signal_id)) if signal_id else None
            )
            if not check.verified:
                metrics.incr("decisions.provenance.unverified")
                logger.warning(
                    "decision provenance unverified brief=%s decision=%s reason=%s",
                    brief.id, decision.get("decision_id"), check.reason,
                )
            out[str(decision.get("decision_id"))] = check.to_dict()
        return out

    async def latest_brief(self, org_id: uuid.UUID) -> DailyBrief | None:
        """The newest brief this org has, by the point in time it describes.

        NULLS LAST is load-bearing, not a style choice. Postgres orders NULLs
        FIRST under `DESC`, and a brief written before the org's first sync has
        `as_of = NULL` with zero decisions — so a plain `desc(as_of)` returned
        that empty row ahead of every real brief, permanently. The visible
        symptom: `POST /brief/generate` produced the correct decisions, the
        dashboard rendered them, and the next SWR revalidation of
        `GET /brief/today` replaced them with an empty brief about a second
        later. Every generated brief was unreadable through this method.

        A NULL `as_of` means "we do not know what moment this describes", which
        is strictly less useful than any dated brief — so it sorts last and is
        returned only when nothing dated exists.

        `reasoning_version` breaks the tie, and it is load-bearing for the same
        kind of reason. `as_of` is State's logical time, so REPROJECTING one
        snapshot under corrected reasoning produces a second brief with a byte
        identical `as_of` — live, org 071ae55b holds two briefs at
        2026-08-17 16:11:12, one from reasoning v3 and one from v4, and org
        df7d1c0a holds the same pair at 15:26:36. `as_of DESC LIMIT 1` alone
        picked between them by whatever order the scan happened to return, so the
        superseded v3 brief could be served indefinitely. The newer reasoning
        version is by construction the corrected projection of that snapshot.
        `id` last makes the order total: it is uuid5 over
        (org, reasoning_version, state_watermark), so two briefs that tie on both
        preceding keys still order identically on every query.

        No brief is deleted or rewritten — the older projections stay stored and
        stay reachable by id.
        """
        return await self.db.scalar(
            select(DailyBrief)
            .where(DailyBrief.organization_id == org_id)
            .order_by(
                desc(DailyBrief.as_of).nulls_last(),
                desc(DailyBrief.reasoning_version),
                desc(DailyBrief.id),
            )
            .limit(1)
        )
