"""Execution HTTP surface — approval workflow only. NO synchronous provider calls.

Approve/reject/cancel flip state + write the outbox; the actual side effect runs
out-of-band in the worker (enqueued via Celery). The request returns in milliseconds.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.execution.actions import (
    ActionPreconditionFailed,
    ActionSpec,
    UnknownAction,
    awaiting_draft,
    require_spec,
)
from navis.execution.config import MAX_EXECUTION_ATTEMPTS
from navis.execution.drafts import (
    DraftNotApprovable,
    DraftNotFound,
    DraftService,
    InsufficientContext,
)
from navis.execution.models import ActionType, ExecutionOrigin, ExecutionRequest
from navis.execution.providers.reply import NEEDS_DRAFT_MESSAGE, requires_draft
from navis.execution.service import ExecutionNotFound, ExecutionService
from navis.execution.status import ExecutionStatus, InvalidTransition
from navis.identity.authz import Principal, require_permission
from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity.permissions import Permission
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.organization.service import OrganizationService
from navis.platform.database import get_session
from navis.platform.queue import celery_app

router = APIRouter(prefix="/actions", tags=["execution"])


class ActionRequestBody(BaseModel):
    decision_id: uuid.UUID
    action_type: ActionType
    payload: dict[str, Any] = {}
    # reasoning_signals.signal_id values ONLY — see execution/models.py. Raw
    # captured events belong in `source_signal_event_ids`; mixing them makes
    # every downstream reference unresolvable.
    reasoning_signal_ids: list[str] = []
    source_signal_event_ids: list[str] = []
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
        # Two distinct namespaces, never merged: reasoning_signals.signal_id
        # here, signal_events.id below.
        "reasoning_signal_ids": r.reasoning_signal_ids,
        "source_signal_event_ids": r.source_signal_event_ids or [],
        "reasoning_version": r.reasoning_version,
        "state_watermark": r.state_watermark,
        "provider_reference": r.provider_reference,
        "attempts": r.attempts,
        # The card renders a real failure reason instead of guessing at one.
        # Truncated because `last_error` can carry a full provider traceback.
        "last_error": (r.last_error or None) and r.last_error[:500],
        "origin": r.origin,
        # True while the action is a placeholder Navis proposed but has not
        # written yet. The card renders "needs a draft" instead of Approve,
        # and `approve` refuses it — see providers/reply.py.
        #
        # Spec-driven: an action that does not support drafts can never be a
        # placeholder, and a new draftable connector reports this correctly
        # without editing the serializer.
        "requires_draft": awaiting_draft(r.action_type, r.payload),
    }


def _enqueue_execution() -> None:
    try:
        celery_app.send_task("navis.execution.worker.execute_approved_actions")
    except Exception:  # broker may be down in dev; the beat schedule still picks it up
        pass


async def _run_inline_execution_pass() -> None:
    """Run one execution pass in the FastAPI process (no Celery needed).

    Guarded by `settings.execution_inline_fallback` — always True in dev,
    should be False in production when a real Celery worker is running.
    Fires as `asyncio.create_task` from the approve handler so it does
    NOT block the HTTP response; the frontend picks up state via the
    poll loop.

    Claims are FOR UPDATE SKIP LOCKED so this races safely with a real
    Celery worker if both are running — whoever gets the row wins,
    the other pass is a no-op.
    """
    # Lazy import to avoid touching the worker module at HTTP-router
    # import time (worker.py has module-level Celery @task decorators
    # that register with celery_app; safe but keeps the router lean).
    from navis.execution.worker import _execute

    try:
        await _execute()
    except Exception:  # noqa: BLE001 — inline path must never crash the process
        from navis.platform.monitoring import get_logger

        get_logger("execution.inline").exception(
            "inline execution pass failed (row stays APPROVED; "
            "Celery beat will retry)"
        )


async def _require_org(db: AsyncSession, eid: uuid.UUID, org_id: uuid.UUID) -> ExecutionRequest:
    req = await db.get(ExecutionRequest, eid)
    if req is None or req.organization_id != org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "execution request not found")
    return req


async def _require_action_capability(
    db: AsyncSession, org_id: uuid.UUID, action_type: str
) -> ActionSpec:
    """The org must hold the CONNECTOR capability this action consumes.

    `require_capability(Capability.EXECUTION)` on the route is a blanket gate:
    "may this org execute anything at all?". It does not answer "may this org
    send Slack?" — so an org whose SLACK entitlement was explicitly revoked, or
    whose `kill:gmail` switch is on, could still run a send_slack / send_email
    action, because EXECUTION was the only thing anyone checked. The kill switch
    on gmail stopped Navis READING mail and did nothing about SENDING it.

    Both gates apply: EXECUTION (the route) and the spec's capability (here).
    Resolution is per-org through the same EntitlementService every other gate
    uses, so org isolation is inherited rather than re-implemented.

    Enforced on APPROVE, which is the last boundary before the worker can claim
    the row — so it covers every producer, including the chat surface that
    creates executions without going through this router.
    """
    try:
        spec = require_spec(action_type)
    except UnknownAction as exc:
        # A row whose action type is no longer registered (a connector removed, a
        # legacy type) must not be approvable. It cannot execute either — the
        # worker refuses to dispatch it — so refusing here just says so honestly
        # instead of accepting an approval that can only dead-letter.
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    if not await OrganizationService(db).can_use(org_id, spec.capability):
        raise HTTPException(
            status.HTTP_402_PAYMENT_REQUIRED,
            f"action '{action_type}' requires the '{spec.capability.value}' "
            "capability, which is not enabled for this organization",
        )
    return spec


@router.post("", status_code=201, dependencies=[Depends(require_capability(Capability.EXECUTION))])
async def request_action(
    body: ActionRequestBody,
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    req, created = await ExecutionService(db).request(
        p.org_id, body.decision_id, body.action_type.value, body.payload,
        reasoning_signal_ids=body.reasoning_signal_ids,
        source_signal_event_ids=body.source_signal_event_ids,
        reasoning_version=body.reasoning_version,
        state_watermark=body.state_watermark,
    )
    return {"created": created, **_serialize(req)}


@router.get("")
async def list_pending(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    """The Action Center's queue: PROACTIVE actions awaiting review.

    Origin-filtered on purpose. Chat-originated proposals are reviewed in the
    conversation that produced them — the founder is already looking at the
    card. Returning them here too put one row on two surfaces, each able to
    approve it, and filled the "things Navis noticed" queue with things the
    founder had just typed.

    Chat does not need this endpoint: it holds the execution_id it created and
    polls `GET /actions/{id}` directly.
    """
    rows = (
        await db.scalars(
            select(ExecutionRequest)
            .where(
                ExecutionRequest.organization_id == ctx.org_id,
                ExecutionRequest.status == ExecutionStatus.REQUESTED.value,
                ExecutionRequest.origin == ExecutionOrigin.PROACTIVE.value,
            )
            .order_by(ExecutionRequest.created_at.desc())
            # Bounded response: the approval queue is human-scale; a runaway proposer must
            # not turn this endpoint into an unbounded scan + payload.
            .limit(500)
        )
    ).all()
    return [_serialize(r) for r in rows]


@router.get("/{execution_id}")
async def get_execution(
    execution_id: uuid.UUID,
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Fetch a single execution row by id — used by the AskConsole
    MeetingProposalCard to poll status transitions after approval
    (REQUESTED → APPROVED → EXECUTING → EXECUTED | FAILED).

    Org-scoped: the row is returned only if it belongs to the
    authenticated principal's org (404 otherwise). Never exposes
    another tenant's audit record.
    """
    req = await _require_org(db, execution_id, p.org_id)
    return _serialize(req)


@router.post("/{execution_id}/approve", dependencies=[Depends(require_capability(Capability.EXECUTION))])
async def approve(
    execution_id: uuid.UUID,
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    row = await _require_org(db, execution_id, p.org_id)
    # Per-action capability, on top of the route's blanket EXECUTION gate. Runs
    # BEFORE the draft is frozen and before any state changes, so an org that may
    # not use this connector cannot get a row into APPROVED at all.
    spec = await _require_action_capability(db, p.org_id, row.action_type)
    # CONNECTOR READINESS, for EVERY action that declares it.
    #
    # This call used to live INSIDE the draft branch below, which made
    # `ActionSpec.preflight` reachable only for a draftable action that still
    # needed a draft. `reply_to_email` was the only spec that declared one, so
    # the nesting was invisible — and it silently disqualified every other
    # action from ever having one: a `schedule_meeting` preflight registered
    # tomorrow would simply never run, and the connector author would have no
    # way to tell from the spec that their readiness check was dead code.
    #
    # ORDERING IS THE POINT AND IT IS UNCHANGED FOR DRAFTS. Authorization
    # (`_require_action_capability`, above) is asked FIRST and independently:
    # possession of a credential must never imply permission to use it, so
    # entitlements decide before anything resolves a token. Readiness is asked
    # SECOND, still before the draft is frozen and before any state transition,
    # so a founder is never asked to review a message that cannot be sent.
    #
    # Provider-neutral: the spec supplies the callable and the connector writes
    # the message. This handler still does not know that Google exists.
    if spec.preflight is not None:
        try:
            await spec.preflight(db, p.org_id)
        except ActionPreconditionFailed as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    # An action Navis has proposed but not yet WRITTEN cannot be approved —
    # there is nothing to send. The provider also refuses it (providers/reply.py),
    # which is the hard guarantee; this check exists so the user gets an honest
    # 409 instead of an approval that silently dead-letters seconds later.
    #
    # Spec-driven, not `action_type == REPLY_TO_EMAIL`: a future draftable action
    # (a Notion page, a Jira issue) inherits this path by declaring
    # supports_draft, without editing the generic approve handler.
    if spec.supports_draft and requires_draft(row.payload):
        # A proactive reply is approved through the DRAFT, which is the only
        # thing a human can actually have reviewed. `DraftService.approve`
        # re-checks staleness at this instant and merges the reviewed text into
        # the payload — which is what clears `requires_draft` and lets the
        # execution proceed.
        #
        # The connector pre-flight that used to run here now runs above, for
        # every action rather than only this one. It still runs BEFORE the draft
        # is frozen, which is the property that mattered: without it the founder
        # reviews, edits, approves — and the send dead-letters minutes later on
        # a 403 they had no way to anticipate.
        try:
            await DraftService(db).approve(row, approver_id=p.subject_id)
        except DraftNotFound:
            raise HTTPException(status.HTTP_409_CONFLICT, NEEDS_DRAFT_MESSAGE)
        except DraftNotApprovable as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
        if requires_draft(row.payload):  # belt and braces
            raise HTTPException(status.HTTP_409_CONFLICT, NEEDS_DRAFT_MESSAGE)
    # Snapshot BEFORE the service mutates the identity-mapped row. Only a real
    # REQUESTED -> APPROVED transition creates new work; a repeat approve on an
    # already APPROVED/EXECUTING/EXECUTED row is a no-op read (see
    # ExecutionService._APPROVE_IDEMPOTENT_FROM) and must not enqueue a worker
    # pass. Without this guard every stray double-click queued a Celery task
    # plus an inline pass that could only ever find nothing to claim.
    was_pending = row.status == ExecutionStatus.REQUESTED.value
    try:
        req = await ExecutionService(db).approve(execution_id, p.subject_id)
    except InvalidTransition as exc:
        # Reachable only from REJECTED / CANCELLED — approving something a human
        # explicitly declined is a genuine conflict and stays a 409.
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    except ExecutionNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    if was_pending:
        _enqueue_execution()  # side effect runs out-of-band in the worker
        # Dev-mode inline fallback: fire an in-process execution pass so
        # meetings actually get created on Google even without a running
        # Celery worker. The task runs AFTER we return (asyncio.create_task
        # does not block the response), so the poll loop on the frontend
        # will catch REQUESTED → APPROVED → EXECUTING → EXECUTED as the
        # pass advances the row. Safe to run alongside a real worker —
        # both use FOR UPDATE SKIP LOCKED to race for the row.
        from navis.platform.config import settings as _settings

        if _settings.execution_inline_fallback:
            import asyncio

            asyncio.create_task(_run_inline_execution_pass())
    return _serialize(req)


@router.post("/{execution_id}/retry", dependencies=[Depends(require_capability(Capability.EXECUTION))])
async def retry(
    execution_id: uuid.UUID,
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Re-drive a FAILED execution.

    A FAILED row with attempts remaining is ALREADY re-claimable by the worker
    (`repository.claim_executable`) — but only on the next scheduled pass. The
    card's Retry button needs something to happen now, and `approve` cannot
    serve: FAILED -> APPROVED is not an edge in the status machine, so calling
    it 409s. Rather than widen the state machine (which would let a rejected
    row be resurrected), this endpoint changes NO state and just asks the
    worker to look again.

    Idempotent and safe to spam: the claim is FOR UPDATE SKIP LOCKED, and a row
    that has exhausted its attempts is left alone with a clear 409.
    """
    req = await _require_org(db, execution_id, p.org_id)
    if req.status != ExecutionStatus.FAILED.value:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"only failed executions can be retried (this one is {req.status})",
        )
    if (req.attempts or 0) >= MAX_EXECUTION_ATTEMPTS:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"execution exhausted its {MAX_EXECUTION_ATTEMPTS} attempts; "
            "it is in the dead-letter queue and needs an operator replay",
        )
    _enqueue_execution()
    from navis.platform.config import settings as _settings

    if _settings.execution_inline_fallback:
        import asyncio

        asyncio.create_task(_run_inline_execution_pass())
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


# ── Draft surface (Phase 2B) ───────────────────────────────────────────
#
# The document a human reviews before a proactive email is sent. Deliberately
# separate endpoints from the execution lifecycle: editing a draft is not an
# execution state transition, and the ExecutionRequest identity must survive any
# number of edits (see execution/drafts/models.py).


class DraftEditBody(BaseModel):
    """Partial edit — only the fields present are changed."""

    subject: str | None = Field(None, max_length=500)
    body: str | None = Field(None, max_length=20_000)
    recipients: list[str] | None = Field(None, max_length=20)


async def _sender_name(db: AsyncSession, org_id: uuid.UUID) -> str | None:
    """The founder's display name for the sign-off, or None.

    None is a real answer, not a failure: the draft then ends with a bare
    "Best," rather than an invented name. Never guessed from an email address.

    Resolved through the SAME integration boundary the sender uses, so the
    sign-off names the account the mail will actually go out from. Picking the
    newest Gmail row instead (as this did) signs a multi-account workspace's
    draft with one person's name and sends it from another's address.

    Goes through `resolve_integration` rather than importing the Google resolver:
    this is a generic HTTP handler, and connector authentication does not belong
    in it. The provider name comes from the ActionSpec's registered provider, so
    a future draftable connector reaches its own credential the same way.

    The read-only fallback is deliberate: an org that has not granted
    gmail.send must still be able to GENERATE and review drafts — the approve
    gate is what stops it, and stopping it here would mean losing the sign-off
    name for every org that has not reconsented yet.
    """
    from navis.execution.integrations import IntegrationUnavailable, resolve_integration
    from navis.identity.models import Integration, IntegrationProvider, User

    try:
        row = await resolve_integration(db, org_id, "gmail")
    except IntegrationUnavailable:
        row = await db.scalar(
            select(Integration)
            .where(
                Integration.organization_id == org_id,
                Integration.provider == IntegrationProvider.GMAIL,
                Integration.is_active.is_(True),
            )
            .order_by(Integration.created_at.desc())
        )
    if row is None or row.user_id is None:
        return None
    user = await db.get(User, row.user_id)
    name = (getattr(user, "full_name", None) or "").strip() if user else ""
    return name or None


# `_gmail_send_ready` lived here and was a Gmail-shaped credential question
# inlined in the generic approve handler — the one place Execution knew that a
# provider was Google. It now lives behind `ActionSpec.preflight` as
# `_gmail_send_preflight` (execution/actions.py), asking the same resolver and
# raising `ActionPreconditionFailed` with the same message. Deleted rather than
# left unused: a generic-looking helper that hardcodes one connector is exactly
# what the next connector would copy.


def _draft_response(req: ExecutionRequest, draft) -> dict[str, Any]:
    return {"action": _serialize(req), "draft": draft.to_dict()}


@router.get("/{execution_id}/draft", dependencies=[Depends(require_capability(Capability.EXECUTION))])
async def get_draft(
    execution_id: uuid.UUID,
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The draft for a proactive action, generating one on first open.

    Lazy generation means a candidate the founder never looks at costs nothing,
    and a draft is always written against the freshest thread Navis has. The
    read also re-checks staleness, so opening a card that the world has moved
    past shows the warning immediately rather than at approve time.
    """
    req = await _require_org(db, execution_id, p.org_id)
    try:
        draft = await DraftService(db).get_or_generate(
            req, sender_name=await _sender_name(db, p.org_id)
        )
    except InsufficientContext as exc:
        # Item 11: when there is not enough grounded material, Navis asks
        # rather than producing a confident email.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    await db.commit()
    return _draft_response(req, draft)


@router.put("/{execution_id}/draft", dependencies=[Depends(require_capability(Capability.EXECUTION))])
async def edit_draft(
    execution_id: uuid.UUID,
    body: DraftEditBody,
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Save the founder's edits.

    Updates a document. It does NOT create a second ExecutionRequest, does not
    recompute the idempotency key, and does not touch the execution row — the
    action keeps its identity, lineage and board position across any number of
    edits.
    """
    req = await _require_org(db, execution_id, p.org_id)
    try:
        draft = await DraftService(db).edit(
            req,
            subject=body.subject,
            body=body.body,
            recipients=body.recipients,
            editor_id=p.subject_id,
        )
    except DraftNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no draft for this action yet")
    except DraftNotApprovable as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    await db.commit()
    return _draft_response(req, draft)


@router.post(
    "/{execution_id}/draft/regenerate",
    dependencies=[Depends(require_capability(Capability.EXECUTION))],
)
async def regenerate_draft(
    execution_id: uuid.UUID,
    force: bool = False,
    p: Principal = Depends(require_permission(Permission.EXECUTE_ACTIONS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Rewrite the draft against the CURRENT thread.

    The recovery path for a stale draft. `force` is required to overwrite a
    draft the founder has edited — silently discarding their own words would be
    the most infuriating thing this feature could do, so it takes an explicit
    second ask.
    """
    req = await _require_org(db, execution_id, p.org_id)
    try:
        draft = await DraftService(db).generate(
            req, sender_name=await _sender_name(db, p.org_id), force=force
        )
    except DraftNotApprovable as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    except InsufficientContext as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    await db.commit()
    return _draft_response(req, draft)
