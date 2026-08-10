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


class RescheduleMeetingProvider(Provider):
    """Move an existing event to a new time.

    Separate from CalendarProvider because the payloads and the failure modes
    differ: a reschedule REQUIRES an existing `event_id`, and "the event is
    gone" is a terminal error rather than something to retry.
    """

    name = "calendar"
    action_type = "reschedule_meeting"

    def __init__(self, sender: Sender | None = None) -> None:
        self._sender = sender

    async def validate(self, payload: dict[str, Any]) -> None:
        if not payload.get("event_id"):
            raise ProviderValidationError("reschedule_meeting requires 'event_id'")
        if not payload.get("start"):
            raise ProviderValidationError("reschedule_meeting requires 'start'")

    async def execute(self, payload: dict[str, Any]) -> ProviderResult:
        if self._sender is None:
            return ProviderResult(
                ok=True,
                reference=ProviderReference(
                    "calendar", str(payload.get("event_id")), {"dry_run": True}
                ),
            )
        event_id = await self._sender(payload)
        return ProviderResult(ok=True, reference=ProviderReference("calendar", event_id))

    async def status(self, reference: dict[str, Any]) -> str:
        return "rescheduled"


class CancelMeetingProvider(Provider):
    """Cancel an existing event. Idempotent — 'already gone' is success."""

    name = "calendar"
    action_type = "cancel_meeting"

    def __init__(self, sender: Sender | None = None) -> None:
        self._sender = sender

    async def validate(self, payload: dict[str, Any]) -> None:
        if not payload.get("event_id"):
            raise ProviderValidationError("cancel_meeting requires 'event_id'")

    async def execute(self, payload: dict[str, Any]) -> ProviderResult:
        if self._sender is None:
            return ProviderResult(
                ok=True,
                reference=ProviderReference(
                    "calendar", str(payload.get("event_id")), {"dry_run": True}
                ),
            )
        event_id = await self._sender(payload)
        return ProviderResult(ok=True, reference=ProviderReference("calendar", event_id))

    async def status(self, reference: dict[str, Any]) -> str:
        return "cancelled"
