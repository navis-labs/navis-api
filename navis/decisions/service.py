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

from navis.decisions.base import IntelligenceView, compute_brief_id
from navis.decisions.models import DailyBrief
from navis.decisions.what_matters_today import WhatMattersTodayEngine
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger
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

        # Idempotent upsert: replaying the same Reasoning rewrites the same brief.
        stmt = (
            pg_insert(DailyBrief)
            .values(
                id=brief_id,
                organization_id=org_id,
                reasoning_version=reasoning_version,
                state_watermark=state_watermark,
                as_of=as_of,
                decisions=payload,
            )
            .on_conflict_do_update(
                index_elements=["id"],
                set_={"decisions": payload, "as_of": as_of, "state_watermark": state_watermark},
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
