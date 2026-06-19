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

from navis.execution.contract import OutcomeReference, ProviderResult
from navis.execution.idempotency import execution_id, idempotency_key
from navis.execution.models import ExecutionRequest
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
    ) -> tuple[ExecutionRequest, bool]:
        key = idempotency_key(org_id, decision_id, action_type, payload)
        eid = execution_id(org_id, key)
        values = dict(
            id=eid,
            organization_id=org_id,
            decision_id=decision_id,
            reasoning_signal_ids=reasoning_signal_ids,
            reasoning_version=reasoning_version,
            state_watermark=state_watermark,
            action_type=action_type,
            status=ExecutionStatus.REQUESTED.value,
            idempotency_key=key,
            payload=payload,
            provider_reference={},
            attempts=0,
        )
        req, created = await self.repo.upsert_request(values)
        if created:
            await self._emit(req, ExecutionStatus.REQUESTED)
            await self.repo.flush()
        return req, created

    # ---- human approval lifecycle ----

    async def approve(self, eid: uuid.UUID, approver_id: uuid.UUID) -> ExecutionRequest:
        req = await self._require(eid)
        self._transition(req, ExecutionStatus.APPROVED)
        req.approved_by = approver_id
        req.approved_at = self._now()
        await self._emit(req, ExecutionStatus.APPROVED)
        await self.repo.flush()
        return req

    async def reject(self, eid: uuid.UUID, approver_id: uuid.UUID, reason: str | None = None) -> ExecutionRequest:
        req = await self._require(eid)
        self._transition(req, ExecutionStatus.REJECTED)
        req.approved_by = approver_id
        await self._emit(req, ExecutionStatus.REJECTED, extra={"reason": reason})
        await self.repo.flush()
        return req

    async def cancel(self, eid: uuid.UUID) -> ExecutionRequest:
        req = await self._require(eid)
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

    async def fail_execution(self, req: ExecutionRequest, error: str) -> None:
        self._transition(req, ExecutionStatus.FAILED)
        req.last_error = error[:2000]
        await self._emit(req, ExecutionStatus.FAILED, extra={"error": error[:500]})
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
