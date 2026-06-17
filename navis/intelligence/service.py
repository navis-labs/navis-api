"""Compute an Organizational Intelligence Score.

Inputs: current State (how much is on fire), Learning (how reliable our signals
have become), Memory (how much the company actually knows about itself).

This is intentionally a transparent, explainable formula in phase 1 — trust-first.
It can be replaced by a learned model later without changing the contract.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.learning.models import ConfidenceWeight
from navis.memory.models import Person
from navis.state.builder import build_organizational_state


@dataclass(slots=True)
class IntelligenceScore:
    org_id: uuid.UUID
    score: float            # 0..100
    coverage: float         # how much memory exists
    reliability: float      # how trustworthy our signals are
    control: float          # how few risks are currently open
    explanation: str


class IntelligenceService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def score(self, org_id: uuid.UUID) -> IntelligenceScore:
        state = await build_organizational_state(self.db, org_id)

        people = await self.db.scalar(
            select(func.count(Person.id)).where(Person.organization_id == org_id)
        ) or 0
        coverage = min(1.0, people / 50.0)  # saturates around a 50-person graph

        avg_rate = await self.db.scalar(
            select(func.avg(ConfidenceWeight.success_rate)).where(
                ConfidenceWeight.organization_id == org_id
            )
        )
        reliability = float(avg_rate) if avg_rate is not None else 0.5

        open_risks = (
            state.commitments.total_open
            + len(state.relationships.going_cold())
            + len(state.projects.stalled())
        )
        control = max(0.0, 1.0 - open_risks / 20.0)

        raw = 0.4 * coverage + 0.3 * reliability + 0.3 * control
        return IntelligenceScore(
            org_id=org_id,
            score=round(raw * 100, 1),
            coverage=round(coverage, 3),
            reliability=round(reliability, 3),
            control=round(control, 3),
            explanation=(
                f"coverage={coverage:.2f} (memory depth), "
                f"reliability={reliability:.2f} (signal track record), "
                f"control={control:.2f} ({open_risks} open risks)."
            ),
        )
