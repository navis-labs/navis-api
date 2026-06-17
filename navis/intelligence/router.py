"""Intelligence HTTP surface: the Organizational Intelligence Score."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.dependencies import CurrentContext, get_current_context
from navis.intelligence.service import IntelligenceService
from navis.platform.database import get_session

router = APIRouter(prefix="/intelligence", tags=["intelligence"])


@router.get("/score")
async def score(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    result = await IntelligenceService(db).score(ctx.org_id)
    return {
        "score": result.score,
        "coverage": result.coverage,
        "reliability": result.reliability,
        "control": result.control,
        "explanation": result.explanation,
    }
