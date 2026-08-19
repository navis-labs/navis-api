"""ExecutionService — request -> approve/reject/cancel -> execute, persist-first.

This service NEVER calls a provider and NEVER reads Memory/State/Signals. It:
  * creates idempotent ExecutionRequests from a Decision (lineage attached);
  * enforces the status machine on every transition;
  * writes each domain event to the outbox IN THE SAME TRANSACTION as the state
    change (the relay worker publishes later).

The actual side effect runs in the worker (providers + retry), not here.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from navis.execution.actions import require_spec
from navis.execution.contract import FailureKind, OutcomeReference, ProviderResult
from navis.execution.idempotency import (
    execution_id,
    idempotency_key,
    obligation_idempotency_key,
)
from navis.execution.obligation import reply_obligation_key
from navis.execution.models import ExecutionOrigin, ExecutionRequest
from navis.execution.status import ExecutionStatus, assert_transition
from navis.platform.events import EventType
from navis.platform.monitoring import get_logger

logger = get_logger("execution")

_EVENT_FOR_STATUS = {
    ExecutionStatus.REQUESTED: EventType.ACTION_REQUESTED,
    ExecutionStatus.APPROVED: EventType.ACTION_APPROVED,
    ExecutionStatus.REJECTED: EventType.ACTION_REJECTED,
    ExecutionStatus.EXECUTING: EventType.ACTION_EXECUTING,
    ExecutionStatus.EXECUTED: EventType.ACTION_EXECUTED,
    ExecutionStatus.FAILED: EventType.ACTION_FAILED,
    ExecutionStatus.CANCELLED: EventType.ACTION_CANCELLED,
}


class ExecutionNotFound(Exception):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionService:
    def __init__(self, db=None, *, repo=None, clock=_utcnow) -> None:
        if repo is not None:
            self.repo = repo
        else:
            from navis.execution.repository import ExecutionRepository

            self.repo = ExecutionRepository(db)
        self._now = clock

    # ---- intake (from the Decision->Execution bridge) ----

    async def request(
        self,
        org_id: uuid.UUID,
        decision_id: uuid.UUID,
        action_type: str,
        payload: dict[str, Any],
        *,
        reasoning_signal_ids: list[str],
        reasoning_version: int | None,
        state_watermark: str | None,
        # Raw signal_events.id values the caller was acting on. Optional and
        # keyword-only so every existing call site is unchanged; see the model
        # for why this is not folded into reasoning_signal_ids.
        source_signal_event_ids: list[str] | None = None,
        origin: ExecutionOrigin = ExecutionOrigin.CHAT,
    ) -> tuple[ExecutionRequest, bool]:
        """`origin` decides which surface REVIEWS this, not how it executes.

        It deliberately does NOT feed the idempotency key: the same decision +
        payload is the same action whoever noticed it, and letting origin split
        the key would let one meeting be booked twice — once from chat, once
        from the proactive queue.

        Raises `UnknownAction` for an action_type with no ActionSpec.
        """
        # FAIL CLOSED AT INTAKE. Every producer — the execution router, chat, the
        # proactive proposer — funnels through here, so this is the one place
        # that can refuse an action Navis does not know how to perform.
        #
        # Before this, an unknown action_type wrote a REQUESTED row, appeared on
        # the Action Center, accepted a human approval, and only died in the
        # worker as "no provider for <type>" plus a dead letter. It never reached
        # a provider — but failing AFTER a human has approved it is the wrong
        # boundary: it spends the founder's attention on something that could
        # never have worked, and it makes the DLQ the discovery mechanism for a
        # typo. The spec registry knows the answer before the row exists.
        require_spec(action_type)
        # A reply names the message it answers, and THAT is its identity. Using
        # the payload hash instead let one unanswered email become a new action
        # every time the state watermark moved — see execution/obligation.py.
        #
        # `decision_id` still rides along on the row for lineage and audit; it
        # simply no longer decides what counts as the same action.
        obligation = reply_obligation_key(action_type, payload)
        if obligation is not None:
            key = obligation_idempotency_key(org_id, obligation)
        else:
            key = idempotency_key(org_id, decision_id, action_type, payload)
        eid = execution_id(org_id, key)
        values = dict(
            id=eid,
            organization_id=org_id,
            decision_id=decision_id,
            reasoning_signal_ids=reasoning_signal_ids,
            source_signal_event_ids=source_signal_event_ids or [],
            reasoning_version=reasoning_version,
            state_watermark=state_watermark,
            action_type=action_type,
            status=ExecutionStatus.REQUESTED.value,
            idempotency_key=key,
            obligation_key=obligation,
            payload=payload,
            provider_reference={},
            attempts=0,
            origin=origin.value,
        )
        req, created = await self.repo.upsert_request(values)
        if created:
            await self._emit(req, ExecutionStatus.REQUESTED)
            await self.repo.flush()
        return req, created

    # ---- human approval lifecycle ----

    # Approving something already approved/running/done asked for an outcome that
    # has already been achieved — so it is a no-op, not a conflict. This is what
    # `status.py` means by "double-approval ... no-ops instead of double side
    # effects": the guard prevents the second SIDE EFFECT, it should not surface
    # an error for a request that succeeded.
    #
    # Without this a user double-clicking Approve sees "invalid execution
    # transition executed -> approved (409)" even though their meeting WAS
    # scheduled — and the same 409 appears on a stale action list, a retried
    # request, or a race with the execution worker.
    #
    # REJECTED / CANCELLED are deliberately excluded: approving something a human
    # explicitly declined is a real conflict and must still fail loudly.
    _APPROVE_IDEMPOTENT_FROM = {
        ExecutionStatus.APPROVED,
        ExecutionStatus.EXECUTING,
        ExecutionStatus.EXECUTED,
    }

    # The same reasoning applies to the other two human verbs: re-issuing an
    # intent that already holds is a no-op, re-issuing a CONFLICTING intent is
    # a real 409. So `reject` on a REJECTED row and `cancel` on a CANCELLED row
    # return the row unchanged, while `reject` on an EXECUTED row still raises
    # (the side effect happened; pretending otherwise would be a lie).
    _REJECT_IDEMPOTENT_FROM = {ExecutionStatus.REJECTED}
    _CANCEL_IDEMPOTENT_FROM = {ExecutionStatus.CANCELLED}

    async def approve(self, eid: uuid.UUID, approver_id: uuid.UUID) -> ExecutionRequest:
        req = await self._require(eid)
        if ExecutionStatus(req.status) in self._APPROVE_IDEMPOTENT_FROM:
            # Already approved (and possibly already run). Return the current
            # state unchanged: no re-approval, no second audit entry, and above
            # all no second side effect.
            return req
        self._transition(req, ExecutionStatus.APPROVED)
        req.approved_by = approver_id
        req.approved_at = self._now()
        await self._emit(req, ExecutionStatus.APPROVED)
        await self.repo.flush()
        return req

    async def reject(self, eid: uuid.UUID, approver_id: uuid.UUID, reason: str | None = None) -> ExecutionRequest:
        req = await self._require(eid)
        if ExecutionStatus(req.status) in self._REJECT_IDEMPOTENT_FROM:
            return req
        self._transition(req, ExecutionStatus.REJECTED)
        req.approved_by = approver_id
        await self._emit(req, ExecutionStatus.REJECTED, extra={"reason": reason})
        await self.repo.flush()
        return req

    async def cancel(self, eid: uuid.UUID) -> ExecutionRequest:
        req = await self._require(eid)
        if ExecutionStatus(req.status) in self._CANCEL_IDEMPOTENT_FROM:
            return req
        self._transition(req, ExecutionStatus.CANCELLED)
        await self._emit(req, ExecutionStatus.CANCELLED)
        await self.repo.flush()
        return req

    # ---- worker-side execution lifecycle ----

    async def begin_execution(self, req: ExecutionRequest) -> None:
        self._transition(req, ExecutionStatus.EXECUTING)
        req.attempts = (req.attempts or 0) + 1
        await self._emit(req, ExecutionStatus.EXECUTING)
        await self.repo.flush()

    async def complete_execution(self, req: ExecutionRequest, result: ProviderResult) -> None:
        self._transition(req, ExecutionStatus.EXECUTED)
        req.executed_at = self._now()
        req.provider_reference = result.reference.to_dict() if result.reference else {}
        await self._emit(req, ExecutionStatus.EXECUTED)
        await self.repo.flush()

    async def fail_execution(
        self, req: ExecutionRequest, error: str, *, kind: FailureKind | None = None
    ) -> None:
        """`kind` is WHY, as a stable value; `error` stays the human sentence.

        Optional and keyword-only so every existing call site is unchanged. An
        omitted kind records no `failure_kind` rather than guessing one — a wrong
        classification would be worse than an absent one for anything reading
        these as evidence.
        """
        self._transition(req, ExecutionStatus.FAILED)
        req.last_error = error[:2000]
        await self._emit(
            req,
            ExecutionStatus.FAILED,
            extra={
                "error": error[:500],
                "failure_kind": kind.value if kind is not None else None,
            },
        )
        await self.repo.flush()

    # ---- internals ----

    async def _require(self, eid: uuid.UUID) -> ExecutionRequest:
        req = await self.repo.get(eid)
        if req is None:
            raise ExecutionNotFound(str(eid))
        return req

    @staticmethod
    def _transition(req: ExecutionRequest, to: ExecutionStatus) -> None:
        assert_transition(ExecutionStatus(req.status), to)
        req.status = to.value

    async def _emit(self, req: ExecutionRequest, status: ExecutionStatus, extra: dict | None = None) -> None:
        payload: dict[str, Any] = {
            "execution_id": str(req.id),
            "org_id": str(req.organization_id),
            "decision_id": str(req.decision_id),
            "action_type": req.action_type,
            "status": status.value,
            "reasoning_signal_ids": req.reasoning_signal_ids,
            # Carried alongside, never merged: Observation/Learning key their
            # aggregates on reasoning ids, so the two must stay distinguishable
            # all the way down the outbox.
            "source_signal_event_ids": req.source_signal_event_ids or [],
            "reasoning_version": req.reasoning_version,
            "state_watermark": req.state_watermark,
        }
        if status == ExecutionStatus.EXECUTED:
            # The hook Observation V2 will consume (no Execution change needed later).
            payload["outcome"] = OutcomeReference(
                execution_id=str(req.id),
                org_id=str(req.organization_id),
                decision_id=str(req.decision_id),
                action_type=req.action_type,
                reasoning_signal_ids=req.reasoning_signal_ids,
                reasoning_version=req.reasoning_version,
                state_watermark=req.state_watermark,
                provider_reference=req.provider_reference,
            ).to_dict()
        if extra:
            payload.update(extra)
        await self.repo.add_event(
            req.id, req.organization_id, _EVENT_FOR_STATUS[status].value, payload
        )
