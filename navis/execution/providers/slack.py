"""Slack provider (send_slack). Dry-run by default; real sender injected in production.

IDEMPOTENCY — READ THIS BEFORE RELYING ON IT
--------------------------------------------
`send_slack` is AT-LEAST-ONCE. That is a property of Slack, not a shortcut.

The other two live connectors do better, because their APIs allow it:

  * Calendar is EXACTLY-ONCE — `events.insert` accepts a client-generated event
    id, so a replay 409s instead of booking a second meeting.
  * Gmail is at-least-once with a NARROW window — we stamp our own Message-ID and
    search `rfc822msgid:` before sending, so most replays find the earlier send.

Slack offers neither. `chat.postMessage` takes no client-supplied idempotency
key, returns a server-generated `ts`, and there is no supported way to search for
"did I already post this". So the honest statement is:

    a replay of the same ExecutionRequest posts a SECOND message.

What still protects the common cases:

  * the worker's already-executed guard — a row carrying a provider reference is
    completed, never re-run, which covers a crash between the Slack call and the
    result commit ONCE the reference was written;
  * one attempt per pass, and a bounded retry budget.

The UNCLOSED window is the same one every non-idempotent API has: the provider
succeeded, the worker died before recording the reference, stale-EXECUTING
recovery pushed the row back to FAILED, and the next pass posts again. Closing it
needs an idempotency key Slack does not provide. Documented rather than papered
over — a comment claiming exactly-once here would be a lie the on-call engineer
would find out about at the worst moment.
"""
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
        if not str(payload.get("channel") or "").strip():
            raise ProviderValidationError("send_slack requires 'channel'")
        # A message with no text is never a thing a human meant to post. This is
        # the same defect class as the blank-email bug the Gmail provider fixed:
        # validate() only checked `channel`, so an approved action with an empty
        # (or whitespace-only) body would have posted a blank message into a real
        # channel. Validation lives HERE, the last gate before Slack, so any
        # future producer of a send_slack payload inherits the guarantee.
        #
        # ProviderValidationError is non-retryable by contract, so a malformed row
        # fails once, dead-letters with this message, and never reaches Slack.
        if not str(payload.get("text") or "").strip():
            raise ProviderValidationError(
                "send_slack requires non-empty 'text' — refusing to post a blank "
                "message"
            )

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
