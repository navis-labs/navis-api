"""Slack provider (send_slack). Dry-run by default."""
from __future__ import annotations

import hashlib
from typing import Any, Awaitable, Callable

from navis.execution.contract import ProviderReference, ProviderResult
from navis.execution.providers.base import Provider, ProviderValidationError

Sender = Callable[[dict[str, Any]], Awaitable[str]]


class SlackProvider(Provider):
    name = "slack"
    action_type = "send_slack"

    def __init__(self, sender: Sender | None = None) -> None:
        self._sender = sender

    async def validate(self, payload: dict[str, Any]) -> None:
        if not payload.get("channel"):
            raise ProviderValidationError("send_slack requires 'channel'")

    async def execute(self, payload: dict[str, Any]) -> ProviderResult:
        if self._sender is None:
            synthetic = "dryrun-" + hashlib.sha256(
                f"{payload.get('channel')}|{payload.get('text')}".encode()
            ).hexdigest()[:16]
            return ProviderResult(
                ok=True, reference=ProviderReference("slack", synthetic, {"dry_run": True})
            )
        ts = await self._sender(payload)
        return ProviderResult(ok=True, reference=ProviderReference("slack", ts))

    async def status(self, reference: dict[str, Any]) -> str:
        return "sent"
