"""Decision service: assemble the brief from state + reasoning, persist, announce.

This is the seam where the whole pipeline converges:
    state -> reasoning -> what-matters-today -> DailyBrief
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.decisions.models import DailyBrief
from navis.decisions.what_matters_today import WhatMattersTodayEngine
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger
from navis.reasoning import run_reasoning
from navis.state.builder import build_organizational_state

logger = get_logger("decisions")


class DecisionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.engine = WhatMattersTodayEngine()

    async def generate_brief(self, org_id: uuid.UUID) -> DailyBrief:
        state = await build_organizational_state(self.db, org_id)
        signals = run_reasoning(state)
        items = self.engine.rank(signals)

        brief = DailyBrief(
            organization_id=org_id,
            generated_at=datetime.now(timezone.utc),
            state_summary=state.summary(),
            items=[item.to_dict() for item in items],
        )
        self.db.add(brief)
        await self.db.flush()

        await event_bus.publish(
            DomainEvent(
                type=EventType.BRIEF_READY,
                org_id=org_id,
                source="decisions",
                payload={"brief_id": str(brief.id), "item_count": len(items)},
            )
        )
        logger.info("brief generated org=%s items=%d", org_id, len(items))
        return brief

    async def latest_brief(self, org_id: uuid.UUID) -> DailyBrief | None:
        return await self.db.scalar(
            select(DailyBrief)
            .where(DailyBrief.organization_id == org_id)
            .order_by(desc(DailyBrief.generated_at))
            .limit(1)
        )
