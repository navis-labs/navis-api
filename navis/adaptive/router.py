"""Adaptive HTTP surface: propose (machine) + govern (founder only).

Adaptive never activates policy on its own. The edge supplies the wall-clock
`at` for governance acts (via platform.utcnow), keeping the projection pure.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from navis.adaptive.governance import NotAuthorized, PolicyViolation
from navis.adaptive.repository import AdaptiveRepository
from navis.adaptive.service import AdaptiveService
from navis.identity.dependencies import CurrentContext, get_current_context
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session
from navis.platform.database.base import utcnow

router = APIRouter(prefix="/adaptive", tags=["adaptive"])


def _svc(db: AsyncSession) -> AdaptiveService:
    return AdaptiveService(AdaptiveRepository(db))


def _serialize_proposal(row) -> dict[str, Any]:
    return {
        "proposal_id": str(row["proposal_id"] if isinstance(row, dict) else row.proposal_id),
        "adaptive_version": row["adaptive_version"] if isinstance(row, dict) else row.adaptive_version,
        "status": row["status"] if isinstance(row, dict) else row.status,
        "proposed_changes": row["proposed_changes"] if isinstance(row, dict) else row.proposed_changes,
        "proposed_payload": row["proposed_payload"] if isinstance(row, dict) else row.proposed_payload,
        "supporting_insight_ids": row["supporting_insight_ids"] if isinstance(row, dict) else row.supporting_insight_ids,
        "supporting_learning_ids": row["supporting_learning_ids"] if isinstance(row, dict) else row.supporting_learning_ids,
    }


@router.get("/policy")
async def active_policy(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    return {"policy": await _svc(db).get_active_policy(ctx.org_id)}


@router.get("/proposals")
async def list_proposals(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = await AdaptiveRepository(db).get_proposals(ctx.org_id)
    return {"proposals": [_serialize_proposal(r) for r in rows]}


@router.post("/propose", dependencies=[Depends(require_capability(Capability.ADAPTIVE))])
async def propose(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    result = await _svc(db).propose(ctx.org_id)
    return {"proposal": _serialize_proposal(result) if result else None}


@router.post("/proposals/{proposal_id}/approve")
async def approve(
    proposal_id: str,
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    import uuid

    try:
        await _svc(db).approve(
            ctx.org_id, uuid.UUID(proposal_id),
            approver_id=ctx.user.id, role=ctx.role, at=utcnow(),
        )
    except NotAuthorized as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc))
    except (LookupError,) as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    except (ValueError, PolicyViolation) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return {"status": "approved"}


@router.post("/proposals/{proposal_id}/reject")
async def reject(
    proposal_id: str,
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    import uuid

    try:
        await _svc(db).reject(
            ctx.org_id, uuid.UUID(proposal_id),
            approver_id=ctx.user.id, role=ctx.role, at=utcnow(),
        )
    except NotAuthorized as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc))
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return {"status": "rejected"}


@router.post("/rollback/{policy_id}")
async def rollback(
    policy_id: str,
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    import uuid

    try:
        new_id = await _svc(db).rollback(
            ctx.org_id, uuid.UUID(policy_id),
            approver_id=ctx.user.id, role=ctx.role, at=utcnow(),
        )
    except NotAuthorized as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc))
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return {"status": "active", "policy_id": str(new_id)}


@router.post("/kill-switch")
async def kill_switch(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        new_id = await _svc(db).kill_switch(
            ctx.org_id, approver_id=ctx.user.id, role=ctx.role, at=utcnow(),
        )
    except NotAuthorized as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc))
    return {"status": "reverted_to_default", "policy_id": str(new_id)}
