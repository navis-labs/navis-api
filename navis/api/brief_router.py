"""Edge router for the brief — owned by the API layer, backed by orchestration.

This route lives here (not in the decisions domain) so the dependency arrow stays
API → Orchestration → Decision. A domain must never import the layer above it.
Capability-gated on DECISION; the propose-actions bridge additionally requires
EXECUTE_ACTIONS permission + the EXECUTION capability.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from navis.decisions.service import DecisionService
from navis.identity.authz import Principal, require_permission
from navis.identity.permissions import Permission
from navis.orchestration.brief import build_brief, propose_actions
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session

router = APIRouter(prefix="/brief", tags=["decisions"])


def _serialize(brief: Any) -> dict[str, Any]:
    return {
        "id": str(brief.id),
        "org_id": str(brief.organization_id),
        "as_of": brief.as_of.isoformat() if brief.as_of else None,
        "reasoning_version": brief.reasoning_version,
        "state_watermark": brief.state_watermark,
        "decisions": brief.decisions,
    }


@router.post("/generate")
async def generate(
    p: Principal = Depends(require_capability(Capability.DECISION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Assemble a fresh brief: State V2 -> Reasoning V2 -> Decision (orchestrated)."""
    brief = await build_brief(db, p.org_id)
    return _serialize(brief)


@router.get("/today")
async def today(
    p: Principal = Depends(require_capability(Capability.DECISION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """What the founder opens in the morning. Builds one if none exists yet."""
    brief = await DecisionService(db).latest_brief(p.org_id)
    if brief is None:
        brief = await build_brief(db, p.org_id)
    return _serialize(brief)


@router.post("/propose-actions")
async def propose(
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    _: Principal = Depends(require_capability(Capability.EXECUTION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Decision -> Execution bridge: turn the latest brief into REQUESTED actions
    (idempotent; human approval still required before anything runs)."""
    created = await propose_actions(db, p.org_id)
    return {"proposed": created}
