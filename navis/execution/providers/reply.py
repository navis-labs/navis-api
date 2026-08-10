"""Reply-to-thread provider (reply_to_email).

Separate from `send_email` because a reply is not a fresh message: it must land
in an existing Gmail thread (`reply_thread_id`) and reference the message it
answers, and it is proposed PROACTIVELY — by Navis, from a reasoning signal —
rather than dictated by the user in chat.

That second difference is the load-bearing one. A proactive reply is authored in
two steps that can be days apart:

  1. Navis notices someone is waiting and proposes "Reply to John". At this
     point there is no body — Navis has not written the email, and inventing one
     would be fabrication.
  2. (Phase 2) A drafting stage writes a grounded body, and the founder edits
     and approves it.

Between those two steps the candidate must be visible but NOT sendable. That is
what `requires_draft` encodes, and this provider is the hard guarantee: it
refuses to execute anything without a real body, so a bug in the UI, a stray
API call, or a future caller cannot turn a placeholder into a sent email.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from navis.execution.contract import ProviderReference, ProviderResult
from navis.execution.providers.base import Provider, ProviderValidationError

Sender = Callable[[dict[str, Any]], Awaitable[str]]

NEEDS_DRAFT_MESSAGE = (
    "This reply has no draft yet, so there is nothing to send. Navis proposed "
    "who to reply to and why; writing the message is not implemented yet."
)


def requires_draft(payload: dict[str, Any] | None) -> bool:
    """True when a payload is a placeholder that must not be executed.

    Checks the explicit flag AND the actual body, so an payload that merely
    forgot the flag is still caught. Used by the approve endpoint (to refuse
    cleanly with a 409) and by the provider (to refuse absolutely).
    """
    if not payload:
        return True
    if payload.get("requires_draft"):
        return True
    return not str(payload.get("body") or "").strip()


class ReplyToEmailProvider(Provider):
    name = "gmail"
    action_type = "reply_to_email"

    def __init__(self, sender: Sender | None = None) -> None:
        self._sender = sender  # None => dry-run

    async def validate(self, payload: dict[str, Any]) -> None:
        if not payload.get("to"):
            raise ProviderValidationError("reply_to_email requires 'to'")
        if not payload.get("reply_thread_id"):
            raise ProviderValidationError(
                "reply_to_email requires 'reply_thread_id' — a reply must land "
                "in the thread it answers"
            )
        if not str(payload.get("subject") or "").strip():
            raise ProviderValidationError("reply_to_email requires 'subject'")
        if requires_draft(payload):
            # Non-retryable by contract: a draftless reply fails once with this
            # message and dead-letters. It is never dispatched and never retried
            # into existence.
            raise ProviderValidationError(NEEDS_DRAFT_MESSAGE)

    async def execute(self, payload: dict[str, Any]) -> ProviderResult:
        if self._sender is None:
            import hashlib

            synthetic = "dryrun-" + hashlib.sha256(
                f"{payload.get('reply_thread_id')}|{payload.get('subject')}".encode()
            ).hexdigest()[:16]
            return ProviderResult(
                ok=True,
                reference=ProviderReference("gmail", synthetic, {"dry_run": True}),
            )
        message_id = await self._sender(payload)
        return ProviderResult(ok=True, reference=ProviderReference("gmail", message_id))

    async def status(self, reference: dict[str, Any]) -> str:
        return "sent"
