"""Task provider (create_task). Tasks are internal, so the default is a real
deterministic create (no external network) — returns a stable internal task id."""
from __future__ import annotations

import hashlib
from typing import Any

from navis.execution.contract import ProviderReference, ProviderResult
from navis.execution.providers.base import Provider, ProviderValidationError


class TasksProvider(Provider):
    name = "tasks"
    action_type = "create_task"

    async def validate(self, payload: dict[str, Any]) -> None:
        if not payload.get("title"):
            raise ProviderValidationError("create_task requires 'title'")

    async def execute(self, payload: dict[str, Any]) -> ProviderResult:
        task_id = "task-" + hashlib.sha256(
            f"{payload.get('title')}|{payload.get('decision_id')}".encode()
        ).hexdigest()[:16]
        return ProviderResult(ok=True, reference=ProviderReference("tasks", task_id))

    async def status(self, reference: dict[str, Any]) -> str:
        return "open"
