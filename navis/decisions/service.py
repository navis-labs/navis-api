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

from navis.decisions.base import compute_brief_id
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
    ) -> DailyBrief:
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
        logger.info("brief %s org=%s decisions=%d", brief_id, org_id, len(decisions))
        brief = await self.db.get(DailyBrief, brief_id)
        assert brief is not None  # upserted above in this same transaction
        return brief

    async def latest_brief(self, org_id: uuid.UUID) -> DailyBrief | None:
        # NULLS LAST is load-bearing: as_of is nullable (an empty-state brief — new org, no
        # facts folded yet — persists with as_of=None under its own watermark-keyed row).
        # Postgres sorts NULLs FIRST under DESC, so without it that early empty brief would
        # permanently shadow every later real brief here (empty /brief/today, zero proposed
        # actions). created_at desc is a stable read tiebreak when as_of values tie; it is
        # cache-write metadata, never decision identity.
        return await self.db.scalar(
            select(DailyBrief)
            .where(DailyBrief.organization_id == org_id)
            .order_by(desc(DailyBrief.as_of).nulls_last(), desc(DailyBrief.created_at))
            .limit(1)
        )
