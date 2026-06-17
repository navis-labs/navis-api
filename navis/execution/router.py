"""Execution HTTP surface: list proposed actions, approve & execute."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.execution.models import Action, ActionStatus, ActionType
from navis.execution.service import ExecutionService
from navis.identity.dependencies import CurrentContext, get_current_context
from navis.platform.database import get_session

router = APIRouter(prefix="/actions", tags=["execution"])


class ActionRequest(BaseModel):
    type: ActionType
    params: dict[str, Any] = {}
    origin_ref: str | None = None


def _serialize(a: Action) -> dict[str, Any]:
    return {
        "id": str(a.id),
        "type": a.type.value,
        "status": a.status.value,
        "params": a.params,
        "result": a.result,
        "origin_ref": a.origin_ref,
    }


@router.post("", status_code=201)
async def request_action(
    body: ActionRequest,
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    action = await ExecutionService(db).request(ctx.org_id, body.type, body.params, body.origin_ref)
    return _serialize(action)


@router.get("")
async def list_actions(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    rows = (
        await db.scalars(
            select(Action)
            .where(Action.organization_id == ctx.org_id, Action.status == ActionStatus.REQUESTED)
            .order_by(Action.created_at.desc())
        )
    ).all()
    return [_serialize(a) for a in rows]


@router.post("/{action_id}/approve")
async def approve_action(
    action_id: uuid.UUID,
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    action = await db.get(Action, action_id)
    if not action or action.organization_id != ctx.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "action not found")
    action = await ExecutionService(db).approve_and_execute(action, ctx.user.id)
    return _serialize(action)
