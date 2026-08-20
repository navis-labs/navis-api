"""Edge router for the brief — owned by the API layer, backed by orchestration.

This route lives here (not in the decisions domain) so the dependency arrow stays
API → Orchestration → Decision. A domain must never import the layer above it.
Capability-gated on DECISION; the propose-actions bridge additionally requires
EXECUTE_ACTIONS permission + the EXECUTION capability.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.decisions.models import DailyBrief, DecisionDismissal
from navis.decisions.service import DecisionService
from navis.identity.authz import Principal, require_permission
from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity.permissions import Permission
from navis.identity.throttle import rate_limit
from navis.execution.proposer import decision_obligation_key
from navis.execution.repository import ExecutionRepository
from navis.execution.summary import execution_summary
from navis.orchestration.brief import build_brief, propose_actions
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session
from navis.platform.events import DomainEvent, EventType, event_bus

# Deterministic namespace for dismissal ids — replay-safe and disjoint from
# every other UUIDv5 space in the app.
_DISMISSAL_NAMESPACE = uuid.UUID("7e2b1c4a-9d5f-5e11-b8c2-4a9f6c3d2b18")

router = APIRouter(prefix="/brief", tags=["decisions"])


def _serialize(brief: Any, decisions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": str(brief.id),
        "org_id": str(brief.organization_id),
        "as_of": brief.as_of.isoformat() if brief.as_of else None,
        "reasoning_version": brief.reasoning_version,
        "state_watermark": brief.state_watermark,
        "decisions": brief.decisions if decisions is None else decisions,
    }


async def _with_executions(db: AsyncSession, brief: Any) -> dict[str, Any]:
    """The brief, with each decision told what has already been done about it.

    Without this the dashboard cannot tell "nobody has replied yet" from "the
    reply was sent an hour ago": `GET /actions` returns REQUESTED rows only, so
    a completed execution is invisible and the drawer keeps offering to send a
    reply that has already gone out.

    Resolution is by the P0 STABLE OBLIGATION KEY, never by decision_id. The
    same unanswered email carries a different decision_id in every brief, so
    matching on lineage would lose the execution the moment the brief was
    rebuilt — the exact defect that let one email become three executions.

    Strictly additive: every existing field is passed through untouched, the
    decision list keeps its order, and nothing about brief generation or
    `latest_brief` changes. Decisions gain `execution`, which is null whenever
    no execution exists — the common case, since most decisions are findings to
    read rather than replies to send.
    """
    decisions = list(brief.decisions or [])
    if not decisions:
        return _serialize(brief)

    # Same `_map` the proposer uses, so the read path and the write path cannot
    # disagree about which obligation a decision represents.
    keys = [decision_obligation_key(d) for d in decisions]
    found = await ExecutionRepository(db).get_by_obligation_keys(
        brief.organization_id, [k for k in keys if k]
    )

    enriched = [
        {**d, "execution": execution_summary(found.get(k)) if k else None}
        for d, k in zip(decisions, keys)
    ]
    return _serialize(brief, enriched)


@router.post(
    "/generate",
    # T-23: a whole-pipeline recompute (state fold -> reasoning -> decision -> LLM).
    dependencies=[Depends(rate_limit("pipeline", "rate_limit_pipeline_per_window"))],
)
async def generate(
    p: Principal = Depends(require_capability(Capability.DECISION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Assemble a fresh brief: State V2 -> Reasoning V2 -> Decision (orchestrated)."""
    brief = await build_brief(db, p.org_id)
    return await _with_executions(db, brief)


@router.get("/today")
async def today(
    p: Principal = Depends(require_capability(Capability.DECISION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """What the founder opens in the morning. Builds one if none exists yet."""
    brief = await DecisionService(db).latest_brief(p.org_id)
    if brief is None:
        brief = await build_brief(db, p.org_id)
    return await _with_executions(db, brief)


@router.post(
    "/propose-actions",
    dependencies=[Depends(rate_limit("pipeline", "rate_limit_pipeline_per_window"))],
)
async def propose(
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    _: Principal = Depends(require_capability(Capability.EXECUTION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Decision -> Execution bridge: turn the latest brief into REQUESTED actions
    (idempotent; human approval still required before anything runs)."""
    created = await propose_actions(db, p.org_id)
    return {"proposed": created}


@router.post("/{brief_id}/open")
async def mark_brief_opened(
    brief_id: uuid.UUID,
    ctx: CurrentContext = Depends(get_current_context),
    _: Any = Depends(require_capability(Capability.DECISION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The founder opened today's brief. Emits BRIEF_OPENED (Learning consumes it
    as a positive engagement signal). No persistence — the event stream is the
    source of truth; the brief itself is a replayable cache."""
    brief = await db.get(DailyBrief, brief_id)
    if brief is None or brief.organization_id != ctx.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "brief not found")

    await event_bus.publish(DomainEvent(
        type=EventType.BRIEF_OPENED,
        org_id=ctx.org_id,
        source="brief.router",
        payload={
            "brief_id": str(brief_id),
            "user_id": str(ctx.user.id),
            "reasoning_version": brief.reasoning_version,
            "state_watermark": brief.state_watermark,
        },
    ))
    return {"status": "opened", "brief_id": str(brief_id)}


class DismissDecisionBody(BaseModel):
    reason: Literal["explicit", "stale", "duplicate"] = "explicit"


@router.post("/decisions/{decision_ref}/dismiss")
async def dismiss_decision(
    decision_ref: str,
    body: DismissDecisionBody,
    ctx: CurrentContext = Depends(get_current_context),
    _: Any = Depends(require_capability(Capability.DECISION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The founder dismissed a decision from today's brief. Persists an idempotent
    row (org, decision_ref, user) so replay converges, then emits DECISION_DISMISSED
    for Learning to consume as an explicit negative signal.

    `decision_ref` is opaque by design — the frontend may pass a Decision id, a
    reasoning signal id, or any stable event id; we don't couple to a specific
    identifier space, so removed decisions still carry their dismissal history."""
    dismissal_id = uuid.uuid5(
        _DISMISSAL_NAMESPACE, f"{ctx.org_id}|{decision_ref}|{ctx.user.id}"
    )
    stmt = (
        pg_insert(DecisionDismissal)
        .values(
            id=dismissal_id,
            organization_id=ctx.org_id,
            decision_ref=decision_ref,
            reason=body.reason,
            user_id=ctx.user.id,
        )
        .on_conflict_do_nothing(index_elements=["id"])
    )
    await db.execute(stmt)
    await db.commit()

    await event_bus.publish(DomainEvent(
        type=EventType.DECISION_DISMISSED,
        org_id=ctx.org_id,
        source="brief.router",
        payload={
            "dismissal_id": str(dismissal_id),
            "decision_ref": decision_ref,
            "reason": body.reason,
            "user_id": str(ctx.user.id),
        },
    ))
    return {"status": "dismissed", "dismissal_id": str(dismissal_id)}
