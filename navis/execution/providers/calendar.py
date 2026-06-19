"""Calendar provider (schedule_meeting). Dry-run by default."""
from __future__ import annotations

import hashlib
from typing import Any, Awaitable, Callable

from navis.execution.contract import ProviderReference, ProviderResult
from navis.execution.providers.base import Provider, ProviderValidationError

Sender = Callable[[dict[str, Any]], Awaitable[str]]


class CalendarProvider(Provider):
    name = "calendar"
    action_type = "schedule_meeting"

    def __init__(self, sender: Sender | None = None) -> None:
        self._sender = sender

    async def validate(self, payload: dict[str, Any]) -> None:
        if not payload.get("title"):
            raise ProviderValidationError("schedule_meeting requires 'title'")
        if not payload.get("start"):
            raise ProviderValidationError("schedule_meeting requires 'start'")

    async def execute(self, payload: dict[str, Any]) -> ProviderResult:
        if self._sender is None:
            synthetic = "dryrun-" + hashlib.sha256(
                f"{payload.get('title')}|{payload.get('start')}".encode()
            ).hexdigest()[:16]
            return ProviderResult(
                ok=True, reference=ProviderReference("calendar", synthetic, {"dry_run": True})
            )
        event_id = await self._sender(payload)
        return ProviderResult(ok=True, reference=ProviderReference("calendar", event_id))

    async def status(self, reference: dict[str, Any]) -> str:
        return "scheduled"
