"""IntelligenceRepository — reads ONLY Learning, writes ONLY intelligence_insights.

Learning is the single allowed upstream dependency (per the Core Law). This module
imports navis.learning.models and nothing else from the pipeline.
"""
from __future__ import annotations

import uuid
from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.intelligence.insights import AggRow, LearningView, PatternRow
from navis.intelligence.models import IntelligenceInsight
from navis.learning.models import (
    AggregateDimension,
    LearningAggregate,
    LearningEvent,
    LearningPattern,
)

_MAX_LINEAGE = 50
_LINEAGE_SCAN_LIMIT = 20000  # bounded scan; full counts come from aggregates


class IntelligenceRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _aggs(self, org_id: uuid.UUID, dimension: str) -> list[AggRow]:
        rows = (
            await self.db.scalars(
                select(LearningAggregate).where(
                    LearningAggregate.organization_id == org_id,
                    LearningAggregate.dimension == dimension,
                )
            )
        ).all()
        return [
            AggRow(r.key, r.successes, r.failures, r.neutrals, r.total) for r in rows
        ]

    async def build_view(self, org_id: uuid.UUID) -> LearningView:
        learning_version = int(
            await self.db.scalar(
                select(func.count(LearningEvent.id)).where(
                    LearningEvent.organization_id == org_id
                )
            ) or 0
        )
        as_of = await self.db.scalar(
            select(func.max(LearningEvent.occurred_at)).where(
                LearningEvent.organization_id == org_id
            )
        )

        signal_aggs = await self._aggs(org_id, AggregateDimension.SIGNAL.value)
        decision_aggs = await self._aggs(org_id, AggregateDimension.DECISION.value)
        outcome_aggs = await self._aggs(org_id, AggregateDimension.OUTCOME.value)

        pattern_rows = (
            await self.db.scalars(
                select(LearningPattern).where(LearningPattern.organization_id == org_id)
            )
        ).all()
        patterns = [
            PatternRow(p.signal_id, p.kind, p.success_rate, p.samples) for p in pattern_rows
        ]

        lineage = await self._lineage(org_id)
        return LearningView(
            org_id=org_id, learning_version=learning_version, as_of=as_of,
            signal_aggs=signal_aggs, decision_aggs=decision_aggs,
            outcome_aggs=outcome_aggs, patterns=patterns, lineage=lineage,
        )

    async def _lineage(self, org_id: uuid.UUID) -> dict[str, list[str]]:
        """Bounded sample of supporting learning_event ids per signal/decision key."""
        events = (
            await self.db.scalars(
                select(LearningEvent)
                .where(LearningEvent.organization_id == org_id)
                .order_by(LearningEvent.id.asc())
                .limit(_LINEAGE_SCAN_LIMIT)
            )
        ).all()
        lineage: dict[str, list[str]] = {}
        for e in events:
            eid = str(e.id)
            for sid in (e.reasoning_signal_ids or []):
                bucket = lineage.setdefault(str(sid), [])
                if len(bucket) < _MAX_LINEAGE:
                    bucket.append(eid)
            if e.decision_id is not None:
                bucket = lineage.setdefault(str(e.decision_id), [])
                if len(bucket) < _MAX_LINEAGE:
                    bucket.append(eid)
        return lineage

    async def append_insight(self, values: dict[str, Any]) -> bool:
        stmt = (
            pg_insert(IntelligenceInsight)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["insight_id"])
        )
        result = await self.db.execute(stmt)
        return (result.rowcount or 0) == 1

    async def get_insights(self, org_id: uuid.UUID) -> Sequence[IntelligenceInsight]:
        return (
            await self.db.scalars(
                select(IntelligenceInsight)
                .where(IntelligenceInsight.organization_id == org_id)
                .order_by(IntelligenceInsight.insight_type.asc(), IntelligenceInsight.key.asc())
            )
        ).all()

    async def delete_org_insights(self, org_id: uuid.UUID) -> None:
        from sqlalchemy import delete

        await self.db.execute(
            delete(IntelligenceInsight).where(IntelligenceInsight.organization_id == org_id)
        )

    async def flush(self) -> None:
        await self.db.flush()
