"""Production senders for CalendarProvider + GmailProvider.

Each sender is a `Callable[[payload], Awaitable[str]]` that:
  1. Reads `payload['__ctx__']['org_id']` (threaded in by the execution worker
     — see navis.execution.worker.execute_pass).
  2. Opens a DB session and resolves the org's active Integration row
     (google_calendar / gmail) — same pattern as navis.orchestration.sync.
  3. Decrypts the stored tokens, builds a TokenManager + GoogleTransport,
     and executes the concrete API call through the existing signals-side
     API clients (CalendarClient.create_event / GmailClient.send_message).
  4. Persists any refreshed token back to the DB in the same request path
     so quota isn't burned on repeat refreshes.
  5. Returns the provider's external id (Google event id / Gmail message id)
     which the CalendarProvider / GmailProvider wraps into a
     ProviderReference for the audit trail.

The senders never touch Execution's own tables — they only wrap external
provider calls. Failure raises; the worker's retry policy handles the rest
(429 / 5xx / auth-refresh) via GoogleTransport.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Any
from uuid import UUID

from sqlalchemy import select

from navis.identity.models import Integration, IntegrationProvider
from navis.platform.database import session_scope
from navis.platform.monitoring import get_logger
from navis.signals.calendar import CalendarClient
from navis.signals.crypto import token_cipher
from navis.signals.gmail import GmailClient
from navis.signals.google.oauth import TokenManager
from navis.signals.google.transport import GoogleAPIError, GoogleTransport

logger = get_logger("execution.providers.senders")


# ── Context extraction ─────────────────────────────────────────────────────


class SenderContextError(Exception):
    """Raised when the execution context is missing required org scoping.

    Propagates to the worker as a non-retryable error — a missing org_id
    means the caller didn't route through ExecutionService properly and no
    number of retries will fix it.
    """


def _org_id_from_payload(payload: dict[str, Any]) -> UUID:
    ctx = payload.get("__ctx__")
    if not isinstance(ctx, dict):
        raise SenderContextError("execution context missing (__ctx__)")
    raw = ctx.get("org_id")
    if not raw:
        raise SenderContextError("execution context missing org_id")
    try:
        return UUID(raw)
    except (TypeError, ValueError) as exc:
        raise SenderContextError(f"invalid org_id in __ctx__: {raw}") from exc


async def _resolve_integration(
    db, org_id: UUID, provider: IntegrationProvider
) -> Integration:
    integration = await db.scalar(
        select(Integration)
        .where(
            Integration.organization_id == org_id,
            Integration.provider == provider,
            Integration.is_active.is_(True),
        )
        .order_by(Integration.created_at.desc())
    )
    if integration is None:
        raise RuntimeError(
            f"no active {provider.value} integration for org {org_id}"
        )
    if not integration.access_token or not integration.refresh_token:
        raise RuntimeError(
            f"{provider.value} integration missing tokens; reconnect required"
        )
    return integration


def _decrypt_tokens_inplace(integration: Integration) -> None:
    """TokenManager expects PLAINTEXT tokens; DB stores them encrypted."""
    if integration.access_token:
        integration.access_token = token_cipher.decrypt(integration.access_token)
    if integration.refresh_token:
        integration.refresh_token = token_cipher.decrypt(integration.refresh_token)


async def _persist_refreshed_token(
    db, integration: Integration, token_manager: TokenManager
) -> None:
    last = token_manager.last_token
    if last is None or not last.refreshed:
        return
    integration.access_token = token_cipher.encrypt(last.access_token)
    integration.token_expires_at = last.expires_at
    if last.refresh_token is not None:
        integration.refresh_token = token_cipher.encrypt(last.refresh_token)
    await db.commit()


# ── Calendar sender ────────────────────────────────────────────────────────


def _iso_end(payload: dict[str, Any]) -> str:
    """Derive an end timestamp when the payload only supplies start + duration.

    ExecutionRequest payloads originate from the reasoning engine or the
    MeetingScheduleDrawer; both usually supply `end` directly, but the
    CalendarProvider only requires `start`, so we tolerate either shape.
    """
    end = payload.get("end")
    if isinstance(end, str) and end.strip():
        return end
    start = payload.get("start")
    if not isinstance(start, str):
        raise RuntimeError("meeting payload missing both 'end' and parsable 'start'")
    try:
        dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"meeting payload has invalid 'start': {start}") from exc
    duration_min = int(payload.get("duration_minutes") or 30)
    return (dt + timedelta(minutes=duration_min)).isoformat()


async def calendar_sender(payload: dict[str, Any]) -> str:
    """Create a real Google Calendar event for the org.

    Payload contract (matches CalendarProvider.validate + the frontend's
    MeetingScheduleDrawer):
      * title (required)            — event summary
      * start (required)            — ISO 8601 with offset
      * end                         — ISO 8601; derived from start + duration_minutes if absent
      * attendees                   — list[str]  emails
      * description                 — str
      * location                    — str
      * time_zone                   — IANA (default UTC)
      * conference                  — bool (attach Google Meet link)
      * send_google_invite          — bool (send native Google email to attendees)
    """
    org_id = _org_id_from_payload(payload)
    async with session_scope() as db:
        integration = await _resolve_integration(
            db, org_id, IntegrationProvider.GOOGLE_CALENDAR
        )
        _decrypt_tokens_inplace(integration)
        token_manager = TokenManager(integration)
        transport = GoogleTransport(
            token_manager.token_provider(),
            on_auth_failure=token_manager.invalidate,
        )
        try:
            client = CalendarClient(transport)
            event = await client.create_event(
                summary=payload["title"],
                start_iso=payload["start"],
                end_iso=_iso_end(payload),
                attendees=payload.get("attendees") or None,
                description=payload.get("description"),
                location=payload.get("location"),
                time_zone=payload.get("time_zone") or "UTC",
                send_updates=(
                    "all" if payload.get("send_google_invite", True) else "none"
                ),
                conference=bool(payload.get("conference", False)),
            )
        except GoogleAPIError as exc:
            # Auth errors are non-retryable at the sender level — surface
            # a clear message so the DLQ + audit both explain "needs
            # reconnect", not a generic 401.
            if exc.is_auth_error:
                raise RuntimeError(
                    "Google Calendar rejected the request; reconnect required"
                ) from exc
            raise
        finally:
            await _persist_refreshed_token(db, integration, token_manager)
            await transport.aclose()

    event_id = str(event.get("id") or "")
    if not event_id:
        raise RuntimeError("calendar create_event returned no event id")
    return event_id


# ── Gmail sender ───────────────────────────────────────────────────────────


def _encode_reply_message(
    *,
    from_email: str,
    to_emails: list[str],
    subject: str,
    body: str,
    in_reply_to_message_id: str | None = None,
) -> str:
    """Build an RFC 2822 message, base64url-encoded for Gmail messages.send."""
    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = ", ".join(to_emails)
    msg["Subject"] = subject
    if in_reply_to_message_id:
        msg["In-Reply-To"] = in_reply_to_message_id
        msg["References"] = in_reply_to_message_id
    msg.set_content(body)
    return base64.urlsafe_b64encode(msg.as_bytes()).rstrip(b"=").decode("ascii")


def _normalise_recipients(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [r.strip() for r in raw.split(",") if r.strip()]
    if isinstance(raw, list):
        return [str(r).strip() for r in raw if r]
    return []


async def gmail_sender(payload: dict[str, Any]) -> str:
    """Send a real outbound Gmail message on behalf of the org.

    Payload contract (matches GmailProvider.validate):
      * to (required)         — string or list[str] of recipient emails
      * subject (required)    — string
      * body                  — string
      * reply_thread_id       — Gmail threadId to append the message to (reply flow)
      * in_reply_to_message_id — RFC 2822 Message-ID of the source message
    """
    org_id = _org_id_from_payload(payload)
    recipients = _normalise_recipients(payload.get("to"))
    if not recipients:
        raise RuntimeError("gmail send: no recipients after normalisation")

    async with session_scope() as db:
        integration = await _resolve_integration(
            db, org_id, IntegrationProvider.GMAIL
        )
        _decrypt_tokens_inplace(integration)
        token_manager = TokenManager(integration)
        transport = GoogleTransport(
            token_manager.token_provider(),
            on_auth_failure=token_manager.invalidate,
        )
        try:
            from_email = integration.external_account or ""
            if not from_email:
                raise RuntimeError(
                    "gmail integration missing external_account; reconnect required"
                )
            raw = _encode_reply_message(
                from_email=from_email,
                to_emails=recipients,
                subject=str(payload.get("subject") or ""),
                body=str(payload.get("body") or ""),
                in_reply_to_message_id=payload.get("in_reply_to_message_id"),
            )
            client = GmailClient(transport)
            sent = await client.send_message(
                raw_rfc822_base64url=raw,
                thread_id=payload.get("reply_thread_id"),
            )
        except GoogleAPIError as exc:
            if exc.is_auth_error:
                raise RuntimeError(
                    "Gmail rejected the request; reconnect required"
                ) from exc
            raise
        finally:
            await _persist_refreshed_token(db, integration, token_manager)
            await transport.aclose()

    message_id = str(sent.get("id") or "")
    if not message_id:
        raise RuntimeError("gmail send returned no message id")
    return message_id


