"""Execution service: request -> approve -> execute, with an audit trail."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from navis.execution.base import ActionHandler, NoopHandler
from navis.execution.models import Action, ActionStatus, ActionType
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger

logger = get_logger("execution")

# Phase-1 registry: all no-ops. Swap in real handlers per type without touching
# the approval flow.
_HANDLERS: dict[ActionType, ActionHandler] = {
    ActionType.SEND_EMAIL: NoopHandler("send_email"),
    ActionType.CREATE_TASK: NoopHandler("create_task"),
    ActionType.SCHEDULE_MEETING: NoopHandler("schedule_meeting"),
    ActionType.CREATE_WORKFLOW: NoopHandler("create_workflow"),
}


class ExecutionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def request(
        self, org_id: uuid.UUID, action_type: ActionType, params: dict, origin_ref: str | None
    ) -> Action:
        action = Action(
            organization_id=org_id,
            type=action_type,
            status=ActionStatus.REQUESTED,
            params=params,
            origin_ref=origin_ref,
        )
        self.db.add(action)
        await self.db.flush()
        await event_bus.publish(
            DomainEvent(
                type=EventType.ACTION_REQUESTED,
                org_id=org_id,
                source="execution",
                payload={"action_id": str(action.id), "type": action_type.value},
            )
        )
        return action

    async def approve_and_execute(self, action: Action, approver_id: uuid.UUID) -> Action:
        action.status = ActionStatus.APPROVED
        action.approved_by = approver_id
        handler = _HANDLERS.get(action.type)
        try:
            result = await handler.execute(action.params) if handler else {"error": "no handler"}
            action.result = result
            action.status = ActionStatus.EXECUTED
            action.executed_at = datetime.now(timezone.utc)
        except Exception as exc:  # pragma: no cover - defensive
            action.status = ActionStatus.FAILED
            action.result = {"error": str(exc)}
            logger.exception("action %s failed", action.id)

        await self.db.flush()
        await event_bus.publish(
            DomainEvent(
                type=EventType.ACTION_EXECUTED,
                org_id=action.organization_id,
                source="execution",
                payload={"action_id": str(action.id), "status": action.status.value},
            )
        )
        return action
