"""Execution HTTP surface — approval workflow only. NO synchronous provider calls.

Approve/reject/cancel flip state + write the outbox; the actual side effect runs
out-of-band in the worker (enqueued via Celery). The request returns in milliseconds.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.execution.models import ActionType, ExecutionRequest
from navis.execution.service import ExecutionNotFound, ExecutionService
from navis.execution.status import ExecutionStatus, InvalidTransition
from navis.identity.authz import Principal, require_permission
from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity.permissions import Permission
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session
from navis.platform.queue import celery_app

router = APIRouter(prefix="/actions", tags=["execution"])


class ActionRequestBody(BaseModel):
    decision_id: uuid.UUID
    action_type: ActionType
    payload: dict[str, Any] = {}
    reasoning_signal_ids: list[str] = []
    reasoning_version: int | None = None
    state_watermark: str | None = None


def _serialize(r: ExecutionRequest) -> dict[str, Any]:
    return {
        "execution_id": str(r.id),
        "org_id": str(r.organization_id),
        "decision_id": str(r.decision_id),
        "action_type": r.action_type,
        "status": r.status,
        "payload": r.payload,
        "reasoning_signal_ids": r.reasoning_signal_ids,
        "reasoning_version": r.reasoning_version,
        "state_watermark": r.state_watermark,
        "provider_reference": r.provider_reference,
        "attempts": r.attempts,
    }


def _enqueue_execution() -> None:
    try:
        celery_app.send_task("navis.execution.worker.execute_approved_actions")
    except Exception:  # broker may be down in dev; the beat schedule still picks it up
        pass


async def _require_org(db: AsyncSession, eid: uuid.UUID, org_id: uuid.UUID) -> ExecutionRequest:
    req = await db.get(ExecutionRequest, eid)
    if req is None or req.organization_id != org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "execution request not found")
    return req


@router.post("", status_code=201, dependencies=[Depends(require_capability(Capability.EXECUTION))])
async def request_action(
    body: ActionRequestBody,
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    req, created = await ExecutionService(db).request(
        p.org_id, body.decision_id, body.action_type.value, body.payload,
        reasoning_signal_ids=body.reasoning_signal_ids,
        reasoning_version=body.reasoning_version,
        state_watermark=body.state_watermark,
    )
    return {"created": created, **_serialize(req)}


@router.get("")
async def list_pending(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    rows = (
        await db.scalars(
            select(ExecutionRequest)
            .where(
                ExecutionRequest.organization_id == ctx.org_id,
                ExecutionRequest.status == ExecutionStatus.REQUESTED.value,
            )
            .order_by(ExecutionRequest.created_at.desc())
        )
    ).all()
    return [_serialize(r) for r in rows]


@router.post("/{execution_id}/approve", dependencies=[Depends(require_capability(Capability.EXECUTION))])
async def approve(
    execution_id: uuid.UUID,
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await _require_org(db, execution_id, p.org_id)
    try:
        req = await ExecutionService(db).approve(execution_id, p.subject_id)
    except InvalidTransition as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    except ExecutionNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    _enqueue_execution()  # side effect runs out-of-band in the worker
    return _serialize(req)


@router.post("/{execution_id}/reject", dependencies=[Depends(require_capability(Capability.EXECUTION))])
async def reject(
    execution_id: uuid.UUID,
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await _require_org(db, execution_id, p.org_id)
    try:
        req = await ExecutionService(db).reject(execution_id, p.subject_id)
    except InvalidTransition as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return _serialize(req)


@router.post("/{execution_id}/cancel", dependencies=[Depends(require_capability(Capability.EXECUTION))])
async def cancel(
    execution_id: uuid.UUID,
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await _require_org(db, execution_id, p.org_id)
    try:
        req = await ExecutionService(db).cancel(execution_id)
    except InvalidTransition as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return _serialize(req)
