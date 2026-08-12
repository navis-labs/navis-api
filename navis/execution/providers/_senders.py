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


from navis.identity.models import Integration
from navis.platform.database import session_scope
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.calendar import CalendarClient
from navis.signals.calendar.client import conference_info
from navis.signals.crypto import token_cipher
from navis.signals.gmail import GmailClient
from navis.signals.google.oauth import TokenManager
from navis.signals.google.resolve import (
    resolve_calendar_integration,
    resolve_gmail_send_integration,
)
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


def _execution_id_from_payload(payload: dict[str, Any]) -> UUID:
    ctx = payload.get("__ctx__")
    raw = ctx.get("execution_id") if isinstance(ctx, dict) else None
    if not raw:
        raise SenderContextError("execution context missing execution_id")
    try:
        return UUID(raw)
    except (TypeError, ValueError) as exc:
        raise SenderContextError(f"invalid execution_id in __ctx__: {raw}") from exc


def _google_event_id(execution_id: UUID) -> str:
    """Deterministic Google Calendar event id for an ExecutionRequest.

    THIS is what makes "no duplicate Google Calendar events can ever occur"
    true rather than aspirational. Google enforces id uniqueness per calendar,
    so a client-supplied id turns events.insert into an idempotent operation:
    every replay of the same ExecutionRequest — worker retry after a timeout
    where Google actually succeeded, a stale-EXECUTING recovery that re-drives
    the row, two workers racing, a manual dead-letter replay — targets the SAME
    id and the second insert comes back 409 instead of booking a second meeting.

    Google's id alphabet is base32hex: `a-v` and `0-9`, length 5-1024. A UUID's
    hex digits (0-9a-f) are a strict subset, and the "navis" prefix is too
    (n, a, v, i, s all fall in a-v), so the result is always valid.
    """
    return f"navis{execution_id.hex}"


# `_resolve_integration` (newest active row for a provider, wins regardless of
# what it can do) lived here and was deleted with its last caller. Both Google
# senders now go through navis.signals.google.resolve, which selects on
# CAPABILITY first. A generic newest-wins helper is exactly the shape that
# reintroduces this bug the next time a sender needs an integration, so it is
# gone rather than merely unused.


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


async def calendar_sender(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Create a real Google Calendar event for the org.

    Returns `(event_id, conference_extra)`. The second element feeds
    `ProviderReference.extra`, carrying `meet_url` / `conference_status` when a
    conference was requested and `{}` otherwise — the reschedule and cancel
    senders still return a bare id, since neither creates a conference.

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

    EXACTLY-ONCE on the Google side: the event id is derived from the
    ExecutionRequest id (see `_google_event_id`), so a replay returns 409 and we
    report the pre-existing event rather than creating a duplicate.
    """
    org_id = _org_id_from_payload(payload)
    event_id = _google_event_id(_execution_id_from_payload(payload))
    async with session_scope() as db:
        # Shared resolver: finds the org's Calendar grant, materialises the
        # per-provider row when a sibling Google grant already carries the
        # scope, and fails loud BEFORE any Google call when the grant is
        # read-only. That last case is the v1→v2 scope migration path: legacy
        # grants keep read features working (freebusy, listing) but writing
        # requires reconsent. The Celery worker maps the raise to FAILED +
        # dead-letter; the audit message says exactly what the user must do.
        integration = await resolve_calendar_integration(
            db, org_id, require_write=True
        )
        await db.commit()  # persist any self-heal before decrypting in place
        _decrypt_tokens_inplace(integration)
        token_manager = TokenManager(integration)
        transport = GoogleTransport(
            token_manager.token_provider(),
            on_auth_failure=token_manager.invalidate,
        )
        try:
            client = CalendarClient(transport)
            try:
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
                    event_id=event_id,
                )
            except GoogleAPIError as exc:
                if not exc.is_conflict:
                    raise
                # 409 — this exact ExecutionRequest already produced an event on
                # this calendar. The desired end state HOLDS, so this is a
                # success, not a failure. Confirm the event is really there
                # (rather than trusting the status code) and return its id, so
                # the audit row still carries a real provider reference.
                logger.info(
                    "calendar create replayed (409) execution_event_id=%s org=%s",
                    event_id,
                    org_id,
                )
                event = await client.get_event(event_id)
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

    created_id = str(event.get("id") or "")
    if not created_id:
        raise RuntimeError("calendar create_event returned no event id")
    # The event object carries the Meet link; returning only the id threw it
    # away. `conference_info` yields {} for a non-conference event, so the
    # reference shape is unchanged when no Meet was requested.
    return created_id, conference_info(event)


# ── Calendar reschedule / cancel senders ───────────────────────────────────


async def _open_calendar(db, org_id: UUID) -> tuple[CalendarClient, Any, Integration]:
    """Resolve + decrypt the org's Calendar integration and build a client.

    Shared by the reschedule / cancel senders so the token lifecycle (decrypt,
    refresh, persist) lives in exactly one place per verb.
    """
    integration = await resolve_calendar_integration(db, org_id, require_write=True)
    await db.commit()  # persist any self-heal before decrypting in place
    _decrypt_tokens_inplace(integration)
    token_manager = TokenManager(integration)
    transport = GoogleTransport(
        token_manager.token_provider(), on_auth_failure=token_manager.invalidate
    )
    return CalendarClient(transport), (transport, token_manager), integration


async def calendar_reschedule_sender(payload: dict[str, Any]) -> str:
    """Move an EXISTING event to a new time.

    Payload contract (see `RescheduleMeetingProvider.validate`):
      * event_id (required)  — the Google event id being moved
      * start (required)     — new ISO 8601 start
      * end / duration_minutes
      * time_zone            — IANA (default UTC)

    PATCH is naturally idempotent: replaying the same new start/end converges on
    the same event, so a worker retry can never produce a second meeting.
    """
    org_id = _org_id_from_payload(payload)
    event_id = str(payload.get("event_id") or "").strip()
    if not event_id:
        raise RuntimeError("reschedule_meeting payload missing 'event_id'")
    async with session_scope() as db:
        client, (transport, token_manager), integration = await _open_calendar(db, org_id)
        try:
            event = await client.patch_event(
                event_id,
                start_iso=payload["start"],
                end_iso=_iso_end(payload),
                time_zone=payload.get("time_zone") or "UTC",
                send_updates=(
                    "all" if payload.get("send_google_invite", True) else "none"
                ),
            )
        except GoogleAPIError as exc:
            if exc.is_auth_error:
                raise RuntimeError(
                    "Google Calendar rejected the request; reconnect required"
                ) from exc
            if exc.is_not_found or exc.is_gone:
                raise RuntimeError(
                    f"calendar event {event_id} no longer exists; nothing to reschedule"
                ) from exc
            raise
        finally:
            await _persist_refreshed_token(db, integration, token_manager)
            await transport.aclose()
    return str(event.get("id") or event_id)


async def calendar_cancel_sender(payload: dict[str, Any]) -> str:
    """Cancel an EXISTING event.

    Payload contract (see `CancelMeetingProvider.validate`):
      * event_id (required) — the Google event id being cancelled

    Idempotent by construction: `CalendarClient.delete_event` swallows 404/410
    because "already gone" IS the requested end state.
    """
    org_id = _org_id_from_payload(payload)
    event_id = str(payload.get("event_id") or "").strip()
    if not event_id:
        raise RuntimeError("cancel_meeting payload missing 'event_id'")
    async with session_scope() as db:
        client, (transport, token_manager), integration = await _open_calendar(db, org_id)
        try:
            await client.delete_event(
                event_id,
                send_updates=(
                    "all" if payload.get("send_google_invite", True) else "none"
                ),
            )
        except GoogleAPIError as exc:
            if exc.is_auth_error:
                raise RuntimeError(
                    "Google Calendar rejected the request; reconnect required"
                ) from exc
            raise
        finally:
            await _persist_refreshed_token(db, integration, token_manager)
            await transport.aclose()
    return event_id


# ── Gmail sender ───────────────────────────────────────────────────────────


def navis_message_id(execution_id: UUID) -> str:
    """Deterministic RFC 2822 Message-ID for one ExecutionRequest.

    THE DUPLICATE-SEND GUARD. Gmail's `messages.send` has no client-side
    idempotency key — unlike Calendar's `events.insert`, which accepts a
    client-generated event id and 409s on replay. So the strongest available
    guard is to stamp our OWN Message-ID and, before every send, ask Gmail whether a
    message with that id already exists (`rfc822msgid:` search).

    Same execution => same Message-ID => a replay finds the earlier send instead
    of making a second one. See `_already_sent` for the honest limits.
    """
    return f"<navis-{execution_id.hex}@navis.local>"


def _encode_reply_message(
    *,
    from_email: str,
    to_emails: list[str],
    subject: str,
    body: str,
    in_reply_to_message_id: str | None = None,
    message_id: str | None = None,
    references: str | None = None,
) -> str:
    """Build an RFC 2822 message, base64url-encoded for Gmail messages.send."""
    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = ", ".join(to_emails)
    msg["Subject"] = subject
    if message_id:
        msg["Message-ID"] = message_id
    if in_reply_to_message_id:
        msg["In-Reply-To"] = in_reply_to_message_id
        # References carries the whole ancestry when we have it, so mail clients
        # thread the reply under the original rather than starting a sibling.
        msg["References"] = references or in_reply_to_message_id
    msg.set_content(body)
    return base64.urlsafe_b64encode(msg.as_bytes()).rstrip(b"=").decode("ascii")


async def _already_sent(client: GmailClient, message_id: str) -> str | None:
    """The Gmail id of a previous send of this exact execution, if any.

    HONEST LIMITS — this narrows the duplicate window, it does not close it:

      * Gmail indexes a sent message for search within moments, but that is not
        a documented synchronous guarantee. A retry inside that window can still
        double-send.
      * A search failure returns None (we proceed to send) rather than blocking
        a legitimate send on a transient error. Failing open is the right trade
        for a founder's outbound mail; failing closed would strand real replies.

    So Gmail remains AT-LEAST-ONCE. Calendar is exactly-once because Google
    supports it there; claiming parity for Gmail would be a lie.
    """
    try:
        found = await client.list_messages(query=f"rfc822msgid:{message_id.strip('<>')}")
    except Exception:  # noqa: BLE001 — see docstring: fail open, do not strand mail
        logger.warning("gmail duplicate pre-check failed for %s; sending anyway", message_id)
        return None
    items = found.get("messages") or []
    if not items:
        return None
    existing = str(items[0].get("id") or "") or None
    if existing:
        logger.warning(
            "gmail send SKIPPED: %s was already sent as message %s", message_id, existing
        )
        metrics.incr("gmail.send.duplicate_prevented")
    return existing


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
    execution_id = _execution_id_from_payload(payload)
    recipients = _normalise_recipients(payload.get("to"))
    if not recipients:
        raise RuntimeError("gmail send: no recipients after normalisation")
    # Final gate before a real send. The provider validates too; this is the
    # last line, and it runs against the payload that will actually be encoded.
    if not str(payload.get("body") or "").strip():
        raise RuntimeError("gmail send: refusing to send an empty body")
    our_message_id = navis_message_id(execution_id)

    async with session_scope() as db:
        # Capability-aware resolution, mirroring calendar_sender's use of
        # resolve_calendar_integration. This must NOT be the generic
        # newest-row-wins `_resolve_integration`: on a workspace with two
        # connected Google accounts that picks the most recent grant and then
        # discovers it cannot send, while a send-capable grant sits one row
        # down. The resolver filters on capability FIRST and raises the same
        # actionable message when nothing can send — so an org still connected
        # under the pre-v4 read-only grant gets one clear failure instead of an
        # opaque 403 from inside a worker retry.
        integration = await resolve_gmail_send_integration(db, org_id)
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
            client = GmailClient(transport)
            # Has this exact execution already been sent? A retry after a
            # timeout that actually delivered would otherwise send twice.
            already = await _already_sent(client, our_message_id)
            if already:
                return already
            raw = _encode_reply_message(
                from_email=from_email,
                to_emails=recipients,
                subject=str(payload.get("subject") or ""),
                body=str(payload.get("body") or ""),
                in_reply_to_message_id=payload.get("in_reply_to_message_id"),
                message_id=our_message_id,
                references=payload.get("references"),
            )
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


