"""Gmail send provider. Dry-run by default; real send is a documented seam.

A real send would POST to gmail.users.messages.send with a write scope using the
same Google transport pattern as signals (separate write-scoped integration). Until
that sender is injected, execute() is a safe dry-run returning a synthetic message id.
"""
from __future__ import annotations

import hashlib
from typing import Any, Awaitable, Callable

from navis.execution.contract import ProviderReference, ProviderResult
from navis.execution.providers.base import Provider, ProviderValidationError

# A real sender: payload -> external message id. Injected in production.
Sender = Callable[[dict[str, Any]], Awaitable[str]]


class GmailProvider(Provider):
    name = "gmail"
    action_type = "send_email"

    def __init__(self, sender: Sender | None = None) -> None:
        self._sender = sender  # None => dry-run

    async def validate(self, payload: dict[str, Any]) -> None:
        if not payload.get("to"):
            raise ProviderValidationError("send_email requires 'to'")
        if not payload.get("subject"):
            raise ProviderValidationError("send_email requires 'subject'")
        # An email with no body is never a thing a human meant to send. The
        # proactive proposer used to emit exactly that — `{"to": <them>,
        # "subject": <the decision title>, "body": ""}` — so one click on a
        # "Commitment slipping: I'll get you the deck by Friday" card sent that
        # sentence, as a subject line, with an empty body, to the person who
        # said it. Validation lives HERE (not only at the proposer) because
        # this is the last gate before Google: any future producer of a
        # send_email payload inherits the guarantee for free.
        #
        # ProviderValidationError is non-retryable by contract, so a malformed
        # row fails once, dead-letters with this message, and never reaches
        # Gmail — not on this attempt and not on any retry.
        if not str(payload.get("body") or "").strip():
            raise ProviderValidationError(
                "send_email requires a non-empty 'body' — refusing to send a "
                "blank email"
            )

    async def execute(self, payload: dict[str, Any]) -> ProviderResult:
        if self._sender is None:
            synthetic = "dryrun-" + hashlib.sha256(
                f"{payload.get('to')}|{payload.get('subject')}".encode()
            ).hexdigest()[:16]
            return ProviderResult(
                ok=True,
                reference=ProviderReference("gmail", synthetic, {"dry_run": True}),
            )
        message_id = await self._sender(payload)
        return ProviderResult(ok=True, reference=ProviderReference("gmail", message_id))

    async def status(self, reference: dict[str, Any]) -> str:
        return "sent"


def build_gmail_sender(transport) -> Sender:
    """Production sender: POST a message via the Google transport (WRITE scope).

    Reuses signals' GoogleTransport pattern (read-only import — Signals unchanged).
    The integration must carry a `gmail.send` scope; until that is provisioned and
    this sender injected into GmailProvider, execution stays dry-run.
    """
    import base64
    from email.message import EmailMessage

    _BASE = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"

    async def _send(payload: dict[str, Any]) -> str:
        msg = EmailMessage()
        msg["To"] = payload["to"]
        msg["Subject"] = payload.get("subject", "")
        msg.set_content(payload.get("body", ""))
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        resp = await transport.request("POST", _BASE, json={"raw": raw})
        return resp.get("id", "")

    return _send
