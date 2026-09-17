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

from navis.decisions.explanation import explain
from navis.decisions.models import DailyBrief, DecisionDismissal
from navis.decisions.service import DecisionService
from navis.identity.authz import Principal, require_permission
from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity.permissions import Permission
from navis.identity.throttle import rate_limit
from navis.execution.proposer import decision_action
from navis.execution.repository import ExecutionRepository
from navis.execution.summary import execution_summary
from navis.orchestration.brief import brief_for_today, build_brief, propose_actions
from navis.orchestration.decision_history import (
    DEFAULT_EPISODE_LIMIT,
    decision_history,
)
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session
from navis.platform.events import DomainEvent, EventType, event_bus

# Deterministic namespace for dismissal ids — replay-safe and disjoint from
# every other UUIDv5 space in the app.
_DISMISSAL_NAMESPACE = uuid.UUID("7e2b1c4a-9d5f-5e11-b8c2-4a9f6c3d2b18")

router = APIRouter(prefix="/brief", tags=["decisions"])


def _serialize(
    brief: Any,
    decisions: list[dict[str, Any]] | None = None,
    obligations: dict[str, Any] | None = None,
    freshness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": str(brief.id),
        "org_id": str(brief.organization_id),
        "as_of": brief.as_of.isoformat() if brief.as_of else None,
        "reasoning_version": brief.reasoning_version,
        "state_watermark": brief.state_watermark,
        # WHETHER THIS BRIEF STILL DESCRIBES THE CURRENT ORGANIZATION — see
        # `orchestration.brief._freshness`. Null on the routes that do not make
        # the comparison (`/generate` has just built the brief from the state it
        # folded), which stays distinguishable from `current: null` meaning "the
        # comparison was attempted and could not be made".
        "freshness": freshness,
        "decisions": brief.decisions if decisions is None else decisions,
        # HOW MANY OBLIGATIONS THIS BRIEF IS A WINDOW ONTO — see
        # `DecisionService.obligations`. Always present so a client never has to
        # decide whether a missing key means zero; `available: false` is what
        # says the counts carry no information.
        "obligations": obligations,
    }


async def _with_executions(
    db: AsyncSession, brief: Any, freshness: dict[str, Any] | None = None
) -> dict[str, Any]:
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

    They also gain `action`: WHETHER THIS DECISION CAN BE ACTED ON AT ALL, and
    through which existing action type. The proposer's answer to that question
    was already being computed here and thrown away — `_map` runs for every
    decision to derive the obligation key — while a client had no way to ask it,
    so an Execute affordance could only be offered on every card or none. Most
    decisions are findings: `risk`, `opportunity` and `delegation` have no
    draftable action, and a `waiting_reply` whose thread carries no
    `rfc_message_id` has no addressable obligation either. Both report
    `action: null`, which is the difference between "nothing has been done yet"
    (`execution: null`) and "there is nothing to do here".

    Nothing new decides this. The action type is `execution.models.ActionType`,
    the mapping is `execution.proposer._map`, and the identity is
    `execution.obligation.reply_obligation_key` — the same three the write path
    uses, so a card cannot offer an action the proposer would refuse to create.
    """
    svc = DecisionService(db)
    obligations = await svc.obligations(brief)
    decisions = list(brief.decisions or [])
    if not decisions:
        return _serialize(brief, obligations=obligations, freshness=freshness)

    # WHETHER EACH DECISION'S REASONING IS REPRODUCIBLE AT ITS OWN SNAPSHOT.
    #
    # One query for the whole brief. A decision publishes the values the engines
    # produced during its pass, while its `reasoning_signal_ids` pointer resolves
    # to a stored conclusion — and until observations existed, that pointer led to
    # the conclusion's FIRST detection, so live decision 4373a0b3 reports 11.85
    # days / 0.8951 against a cited signal recording 7.16 / 0.7386.
    #
    # Reported, never repaired: an unverifiable decision is still served, still
    # ranked, still actionable — it simply stops claiming provenance it cannot
    # produce. Substituting a neighbouring observation would recreate the defect
    # one layer up. See `DecisionService.provenance`.
    provenance = await svc.provenance(brief)

    # ONE derivation, shared with the proposer: `decision_action` runs the same
    # `_map` the write path runs, so the read path and the write path cannot
    # disagree about which obligation a decision represents — or about whether
    # it has one at all.
    actions = [decision_action(d) for d in decisions]
    found = await ExecutionRepository(db).get_by_obligation_keys(
        brief.organization_id, [a.obligation_key for a in actions if a]
    )

    enriched = []
    for d, a in zip(decisions, actions):
        row = {
            **d,
            "execution": (
                execution_summary(found.get(a.obligation_key)) if a else None
            ),
            "action": a.to_dict() if a else None,
            # Always present, never conditional — same rule the decision fields
            # follow: a client must be able to tell "this reasoning could not be
            # verified" from "this build does not report verification".
            "provenance": provenance.get(str(d.get("decision_id"))),
        }
        # The founder-facing projection, built LAST so it can read `action` and
        # answer "can Navis do it?" from the same value the card renders a
        # button from. Pure and additive — every stored field above is
        # untouched and stays available to the trace view.
        row["explanation"] = explain(row)
        enriched.append(row)
    return _serialize(brief, enriched, obligations=obligations, freshness=freshness)


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
    """What the founder opens in the morning.

    Builds one if none exists yet, and REBUILDS one that no longer describes the
    organization: a brief is keyed on the state watermark it was ranked at, so a
    watermark that has moved is a cache key that has changed. See
    `orchestration.brief.brief_for_today` — the comparison is one indexed read,
    the rebuild reuses the existing pipeline, and the result reports which
    happened rather than leaving the founder to guess.
    """
    brief, freshness = await brief_for_today(db, p.org_id)
    return await _with_executions(db, brief, freshness)


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


@router.get(
    "/decisions/history",
    dependencies=[Depends(rate_limit("read", "rate_limit_read_per_window"))],
)
async def decisions_history(
    limit: int = DEFAULT_EPISODE_LIMIT,
    p: Principal = Depends(require_capability(Capability.DECISION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """What has happened to the decisions this organization has made.

    Decision -> Human choice -> Execution -> Outcome -> Learning, folded per
    REASONING CONCLUSION rather than per decision id — see
    `orchestration.decision_history` for why a decision id cannot key a history.

    Lives on this router for the same reason `/today` does: the arrow stays
    API -> Orchestration -> Decision, and the join across Execution, Observation
    and Learning belongs to the composition root, not to any of them.

    STRICTLY READ-ONLY, and deliberately unlike `/today`: it never calls
    `build_brief`, so opening the Decisions page cannot rebuild a brief, fold
    State or re-project Intelligence.
    """
    return await decision_history(db, p.org_id, limit=max(1, min(limit, 200)))


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
    identifier space, so removed decisions still carry their dismissal history.

    THE EVENT NOW CARRIES LINEAGE, which is what makes the sentence above true
    rather than aspirational. `DECISION_DISMISSED` used to publish the dismissal
    id, the ref, the reason and the user — and nothing that names WHAT was
    dismissed in the pipeline's own identifiers. Observation dedupes on
    `event_id` and Learning attributes on `reasoning_signal_ids`, so an event
    without either could not have been consumed even once a handler existed.

    Both are resolved from data that is already stored: the dismissal id is
    deterministic (uuid5 over org|ref|user) and doubles as the idempotency key,
    and the reasoning lineage is read off the stored decision via
    `DecisionService.find_decision`. Nothing is guessed — a ref that resolves to
    no decision publishes no lineage rather than a neighbouring decision's, and
    the dismissal is still persisted either way."""
    dismissal_id = uuid.uuid5(
        _DISMISSAL_NAMESPACE, f"{ctx.org_id}|{decision_ref}|{ctx.user.id}"
    )
    decision = await DecisionService(db).find_decision(ctx.org_id, decision_ref)
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
            # THE IDEMPOTENCY KEY. Deterministic in (org, ref, user), so a
            # re-clicked dismiss produces the same id, Observation's inbox claim
            # fails, and no second outcome is recorded.
            "event_id": str(dismissal_id),
            "dismissal_id": str(dismissal_id),
            "decision_ref": decision_ref,
            # The founder's stated grounds. Carried through to the outcome
            # payload verbatim by `ObservationService.record_from_execution`,
            # which already preserves `reason` for the same purpose on a
            # rejected action.
            "reason": body.reason,
            "user_id": str(ctx.user.id),
            # ── Lineage, read off the stored decision. Absent (null / empty)
            # when the ref resolves to no decision — an honest gap rather than
            # an invented attribution.
            "decision_id": (decision or {}).get("decision_id"),
            "reasoning_signal_ids": list((decision or {}).get("reasoning_signal_ids") or ()),
            "reasoning_version": (decision or {}).get("reasoning_version"),
            "state_watermark": (decision or {}).get("state_watermark"),
        },
    ))
    return {"status": "dismissed", "dismissal_id": str(dismissal_id)}
