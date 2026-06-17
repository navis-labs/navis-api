"""The product endpoint: "What matters today?"."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from navis.decisions.service import DecisionService
from navis.identity.dependencies import CurrentContext, get_current_context
from navis.platform.database import get_session

router = APIRouter(prefix="/brief", tags=["decisions"])


def _serialize(brief: Any) -> dict[str, Any]:
    return {
        "id": str(brief.id),
        "generated_at": brief.generated_at.isoformat(),
        "state_summary": brief.state_summary,
        "items": brief.items,
    }


@router.post("/generate")
async def generate(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Build a fresh brief on demand (also runs on a schedule via orchestration)."""
    brief = await DecisionService(db).generate_brief(ctx.org_id)
    return _serialize(brief)


@router.get("/today")
async def today(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """What the founder opens in the morning. Generates one if none exists yet."""
    svc = DecisionService(db)
    brief = await svc.latest_brief(ctx.org_id)
    if brief is None:
        brief = await svc.generate_brief(ctx.org_id)
    return _serialize(brief)
