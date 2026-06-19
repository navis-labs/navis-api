"""Intelligence HTTP surface: append-only organizational insights.

Read what the organization has learned about itself (facts, not recommendations),
and trigger a (re)projection from Learning. No scores, no advice.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.dependencies import CurrentContext, get_current_context
from navis.intelligence.repository import IntelligenceRepository
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.intelligence.service import IntelligenceService
from navis.platform.database import get_session

router = APIRouter(prefix="/intelligence", tags=["intelligence"])


def _serialize(row) -> dict[str, Any]:
    return {
        "insight_id": str(row.insight_id),
        "insight_type": row.insight_type,
        "key": row.key,
        "title": row.title,
        "summary": row.summary,
        "evidence": row.evidence,
        "sample_size": row.sample_size,
        "confidence": row.confidence,
        "supporting_learning_event_ids": row.supporting_learning_event_ids,
        "learning_version": row.learning_version,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/insights", dependencies=[Depends(require_capability(Capability.INTELLIGENCE))])
async def list_insights(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = await IntelligenceService(IntelligenceRepository(db)).get_insights(ctx.org_id)
    return {"insights": [_serialize(r) for r in rows]}


@router.post("/project", dependencies=[Depends(require_capability(Capability.INTELLIGENCE))])
async def project(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = await IntelligenceService(IntelligenceRepository(db)).project(ctx.org_id)
    return {"projected": len(rows)}
