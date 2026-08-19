"""Production senders for CalendarProvider + GmailProvider.

Each sender is a `Callable[[payload], Awaitable[str]]` that runs in THREE
PHASES, and the phase boundaries are the point:

  1. Read `payload['__ctx__']['org_id']` (threaded in by the execution worker
     from the PERSISTED row — see navis.execution.worker.execute_pass).
  2. SHORT SESSION: `_load_grant` resolves the org's credential through the
     generic integration boundary and returns a DETACHED `_Grant` snapshot.
     The session is closed before this returns.
  3. NO SESSION: build a TokenManager + GoogleTransport over the snapshot and
     make the external call through the existing signals-side API clients
     (CalendarClient.create_event / GmailClient.send_message / SlackClient).
  4. SHORT SESSION, only if Google rotated a token: `_persist_refreshed_token`
     writes it back with a targeted UPDATE so quota isn't burned re-refreshing.
  5. Return the provider's external id (Google event id / Gmail message id /
     Slack ts), which the Provider wraps into a ProviderReference.

WHY THE PHASES ARE NOT NEGOTIABLE
---------------------------------
Steps 2 and 4 used to be one session wrapped around step 3, which produced two
separate production defects — a database connection held idle-in-transaction for
the entire external call, and PLAINTEXT OAUTH TOKENS COMMITTED BACK TO THE
`integrations` TABLE. Both are described in full on `_Grant`. A snapshot that is
not an ORM row cannot be flushed, which is what makes the second one impossible
rather than merely fixed.

The senders never touch Execution's own tables — they only wrap external
provider calls. Failure raises; the worker classifies it (permanent vs
retryable) and the retry policy handles transient 429 / 5xx via GoogleTransport.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Any
from uuid import UUID


from sqlalchemy import select, update

from navis.execution.integrations import IntegrationUnavailable, resolve_integration
from navis.execution.providers.base import ProviderError
from navis.identity.models import Integration, IntegrationProvider
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


class SenderContextError(ProviderError):
    """Raised when the execution context is missing required org scoping.

    A missing or malformed `org_id` means the caller did not route through
    `ExecutionService`, and no number of retries will grow one — so this is
    non-retryable, which is what the base class is now told.

    It previously subclassed `Exception` while its docstring claimed exactly
    this. The worker classifies by TYPE: a bare Exception lands in the transient
    branch, so a request with no organization scope was re-driven the full five
    attempts, each one re-deriving the same impossibility. Retrying a request
    that cannot name its tenant is also the last thing anyone wants near the
    credential boundary.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


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


# ── The Google credential adapter ──────────────────────────────────────────
#
# THIS MODULE IS THE ONLY PLACE IN EXECUTION THAT KNOWS GOOGLE AUTHENTICATION
# EXISTS. The generic layer asks `navis.execution.integrations` for a credential
# by provider name; the boundary dispatches here.
#
# These functions do NOT reimplement anything. They wrap the existing, already
# hardened Signals resolvers (capability-first selection, deterministic ordering,
# self-healing calendar row) and translate their Google-specific exceptions into
# the one provider-neutral error the generic layer understands. Building a second
# Gmail credential system would be the worst possible outcome of this task.


async def resolve_gmail_credential(db, org_id: UUID) -> Integration:
    """Gmail SEND credential for one organization. Registered as provider `gmail`.

    Org scope comes from the argument and nowhere else — there is no fallback to
    a newest-row, global, or process-cached grant.
    """
    from navis.signals.google.resolve import (
        GmailNotConnected,
        GmailSendScopeInsufficient,
    )

    try:
        return await resolve_gmail_send_integration(db, org_id)
    except GmailNotConnected as exc:
        raise IntegrationUnavailable(
            "Google is not connected for this organization, so Navis cannot "
            "send mail on its behalf.",
            provider="gmail",
        ) from exc
    except GmailSendScopeInsufficient as exc:
        # Connected, but the grant cannot send — the pre-v4 read-only scope set.
        # `reconsent_required` is what makes the UI say "reconnect to grant send"
        # instead of the wrong "connect Google".
        raise IntegrationUnavailable(
            "Navis can read this organization's Gmail but is not allowed to send "
            "from it. Reconnect Google Workspace to grant send access.",
            provider="gmail",
            reconsent_required=True,
        ) from exc


async def resolve_slack_credential(db, org_id: UUID) -> Integration:
    """Slack WRITE credential for one organization. Registered as provider `slack`.

    Deliberately does NOT reuse a Google-shaped resolver: Slack bot tokens are
    long-lived with no refresh cycle (see signals/slack/oauth.py), so there is no
    TokenManager, no expiry check and no self-healing sibling row. One active
    SLACK integration per workspace, selected by organization.

    Org scope comes from the argument and nowhere else. Deterministic when a
    workspace somehow holds two active rows: newest install wins, ties broken by
    id, so the selection can never depend on planner order.
    """
    from navis.signals.slack.oauth import slack_write_granted

    rows = (
        await db.scalars(
            select(Integration).where(
                Integration.organization_id == org_id,
                Integration.provider == IntegrationProvider.SLACK,
                Integration.is_active.is_(True),
            )
        )
    ).all()
    if not rows:
        raise IntegrationUnavailable(
            "Slack is not connected for this organization.", provider="slack"
        )

    # Capability FIRST, recency second — the same ordering bug Gmail had, avoided
    # here by construction: a newer read-only install must never outrank an older
    # one that can actually post.
    capable = [r for r in rows if slack_write_granted(r.scopes)]
    if not capable:
        raise IntegrationUnavailable(
            "Navis can read this organization's Slack but is not allowed to post "
            "to it. Reconnect Slack to grant the chat:write permission.",
            provider="slack",
            reconsent_required=True,
        )
    usable = [r for r in capable if r.access_token]
    if not usable:
        raise IntegrationUnavailable(
            "the Slack install has no stored bot token; reconnect required",
            provider="slack",
            reconsent_required=True,
        )
    return sorted(
        usable,
        key=lambda r: (-(r.created_at.timestamp() if r.created_at else 0.0), str(r.id)),
    )[0]


async def slack_sender(payload: dict[str, Any]) -> str:
    """Post a real Slack message on behalf of the org. Returns the message `ts`.

    Payload contract (matches SlackProvider.validate):
      * channel (required) — channel id (C…/G…) or name the bot can post to
      * text (required)    — the message body
      * thread_ts          — reply inside an existing thread

    AT-LEAST-ONCE, and that is a property of Slack, not a shortcut here. See
    SlackProvider for the full statement of the window.
    """
    from navis.signals.slack.client import SlackAPIError, SlackClient, SlackReconsentRequired

    org_id = _org_id_from_payload(payload)
    channel = str(payload.get("channel") or "").strip()
    text = str(payload.get("text") or "").strip()
    if not channel or not text:
        # Last line before the network, mirroring gmail_sender's empty-body guard.
        raise RuntimeError("slack post: refusing to send without a channel and text")

    # Slack already had the right shape (copy the credential out, then close);
    # it now goes through the SAME loader as the Google senders so there is one
    # credential-loading path to audit rather than one per connector. Slack has
    # no refresh cycle, so there is no token to persist afterwards.
    grant = await _load_grant(org_id, "slack")
    token = token_cipher.decrypt(grant.access_token) or ""

    # The Slack call runs with NO session open.
    client = SlackClient(token)
    try:
        body = await client.chat_post_message(
            channel=channel, text=text, thread_ts=payload.get("thread_ts")
        )
    except SlackReconsentRequired as exc:
        # invalid_auth / token_revoked / account_inactive — the install is dead.
        # Surfacing it as IntegrationUnavailable makes it PERMANENT (the worker
        # dead-letters it immediately with a reconnect instruction) instead of
        # retrying against a token that will never work again. `exc.error` is a
        # Slack error code, never a token.
        raise IntegrationUnavailable(
            f"Slack rejected the request ({exc.error}); reconnect Slack to continue.",
            provider="slack",
            reconsent_required=True,
        ) from exc
    except SlackAPIError as exc:
        if exc.retryable:
            raise
        # A permanent business error (channel_not_found, not_in_channel,
        # missing_scope). Re-raised as a plain RuntimeError carrying the Slack
        # error CODE only — the founder needs to know Navis was never invited to
        # that channel; nobody needs the token.
        raise RuntimeError(f"slack post failed: {exc.error}") from exc
    finally:
        await client.aclose()

    ts = str(body.get("ts") or "")
    if not ts:
        raise RuntimeError("slack chat.postMessage returned no message ts")
    return ts


async def resolve_calendar_credential(db, org_id: UUID) -> Integration:
    """Google Calendar WRITE credential for one organization. Provider `calendar`."""
    from navis.signals.google.resolve import (
        CalendarNotConnected,
        CalendarScopeInsufficient,
    )

    try:
        return await resolve_calendar_integration(db, org_id, require_write=True)
    except CalendarNotConnected as exc:
        raise IntegrationUnavailable(
            "Google Calendar is not connected for this organization.",
            provider="calendar",
        ) from exc
    except CalendarScopeInsufficient as exc:
        raise IntegrationUnavailable(
            "Google Calendar is connected without the write scope Navis needs to "
            "create or change events. Reconnect Google Workspace to grant it.",
            provider="calendar",
            reconsent_required=True,
        ) from exc


def _reconsent(provider: str, system: str) -> IntegrationUnavailable:
    """Google rejected the credential itself — PERMANENT, needs a human.

    WHY THIS IS NOT A RuntimeError. Each of these call sites already carried the
    comment "auth errors are non-retryable at the sender level", and each raised
    a plain `RuntimeError`. The worker classifies exceptions by TYPE: it treats
    `ProviderValidationError` and `IntegrationUnavailable` as permanent and
    everything else as transient. A `RuntimeError` therefore landed in the
    transient branch, so the stated intent was never delivered — a dead grant
    burned the whole retry budget across five passes re-discovering that the
    token is still dead, and only then produced the dead letter that tells the
    founder to reconnect.

    401 escaping the transport is already proof of permanence: `GoogleTransport`
    forces a token refresh and retries ONCE on 401 (see `_on_auth_failure`), so
    reaching here means the refresh did not fix it. Only 401 is treated this
    way — Google also returns 403 for `rateLimitExceeded`, which is genuinely
    transient, so widening this to 403 would turn a throttle into a fake
    reconnect prompt.

    `reconsent_required=True` is what makes the UI say "reconnect" rather than
    "connect". The message names the system and the remedy and carries no token.
    """
    return IntegrationUnavailable(
        f"{system} rejected this organization's credential and refreshing it did "
        f"not help. Reconnect {system} to continue.",
        provider=provider,
        reconsent_required=True,
    )


@dataclass(frozen=True, slots=True)
class _Grant:
    """A DETACHED snapshot of one organization's credential.

    THE TWO PROBLEMS THIS SOLVES, BOTH OF WHICH WERE REAL
    -----------------------------------------------------
    1. PLAINTEXT TOKENS WERE BEING WRITTEN BACK TO THE DATABASE.

       The senders held the live `Integration` ORM row and called a
       `_decrypt_tokens_inplace` helper that assigned the DECRYPTED tokens onto
       it. That marks the row dirty. `session_scope` COMMITS on exit
       (platform/database/session.py), and the re-encrypt step only ran when a
       refresh had happened — so on the ordinary path, where the access token
       was still valid, the exit commit wrote the real access and refresh tokens
       to `integrations` IN PLAINTEXT.

       It then stayed silent forever: `token_cipher.decrypt` returns an
       untagged value as-is (the legacy-plaintext path), so everything kept
       working while encryption-at-rest was gone. The only trace was a
       `signals.token.legacy_plaintext_seen` metric — and had the
       `signals_reject_legacy_plaintext` sunset switch ever been turned on,
       every affected integration would have broken at once.

       The decryption was never needed. `TokenManager.get_token` decrypts what
       it is given (signals/google/oauth.py), so handing it the stored
       CIPHERTEXT is both correct and what it already expects. This snapshot
       therefore carries the tokens exactly as stored — encrypted — and the
       in-place helper is gone rather than merely unused.

    2. A DATABASE SESSION WAS HELD OPEN ACROSS THE EXTERNAL API CALL.

       The worker is explicit that Phase B must hold no transaction while it
       talks to Google (see worker._execute_one). The senders defeated that by
       opening their own session and keeping it for the whole conversation —
       `gmail_sender` across BOTH round trips with no intervening commit, so the
       connection sat idle-in-transaction for the full duration including the
       transport's internal backoff.

       Being a plain frozen dataclass rather than an ORM row, this cannot be
       flushed, cannot be expired, and cannot re-open a session. The session is
       closed before the first external byte and a NEW short one is opened only
       if there is a refreshed token to store.
    """

    integration_id: UUID
    external_account: str | None
    # Stored form — ciphertext. Named to match what `TokenManager` duck-types so
    # the snapshot can be handed to it directly.
    access_token: str | None
    refresh_token: str | None
    token_expires_at: datetime | None


async def _load_grant(org_id: UUID, provider: str) -> _Grant:
    """Resolve one org's credential and DETACH it. Session closed on return.

    The single credential-loading path for every executable connector, so the
    "no ORM object outlives the session" rule is enforced in one place rather
    than re-argued in each sender. Goes through `resolve_integration`, so
    organization scoping and the per-provider resolver registry are unchanged.
    """
    async with session_scope() as db:
        integration = await resolve_integration(db, org_id, provider)
        # The calendar resolver may materialise a healing row; commit it while
        # the tokens on that row are still ciphertext. Nothing below mutates the
        # ORM object, so this commit can only ever persist the heal.
        await db.commit()
        return _Grant(
            integration_id=integration.id,
            external_account=integration.external_account,
            access_token=integration.access_token,
            refresh_token=integration.refresh_token,
            token_expires_at=integration.token_expires_at,
        )


async def _persist_refreshed_token(grant: _Grant, token_manager: TokenManager) -> None:
    """Store a token Google rotated during this call, in its own short session.

    A targeted UPDATE by primary key rather than a mutation of an ORM row: there
    is no row to mutate any more, and an UPDATE cannot accidentally carry some
    OTHER pending change to the integration along with it.

    Non-fatal. The external side effect has already happened by the time this
    runs, and failing to cache a refreshed token must never turn a sent email
    into a failed execution — the next call simply refreshes again.
    """
    last = token_manager.last_token
    if last is None or not last.refreshed:
        return
    values: dict[str, Any] = {
        "access_token": token_cipher.encrypt(last.access_token),
        "token_expires_at": last.expires_at,
    }
    if last.refresh_token is not None:
        values["refresh_token"] = token_cipher.encrypt(last.refresh_token)
    try:
        async with session_scope() as db:
            await db.execute(
                update(Integration)
                .where(Integration.id == grant.integration_id)
                .values(**values)
            )
    except Exception:  # noqa: BLE001 — see docstring
        logger.exception(
            "could not persist refreshed token for integration=%s "
            "(the external call already succeeded)",
            grant.integration_id,
        )


def _google_transport(grant: _Grant) -> tuple[GoogleTransport, TokenManager]:
    """A transport bound to a detached grant. No session, no ORM row."""
    token_manager = TokenManager(grant)
    return (
        GoogleTransport(
            token_manager.token_provider(),
            on_auth_failure=token_manager.invalidate,
        ),
        token_manager,
    )


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
    # Shared resolver: finds the org's Calendar grant, materialises the
    # per-provider row when a sibling Google grant already carries the scope, and
    # fails loud BEFORE any Google call when the grant is read-only. That last
    # case is the v1→v2 scope migration path: legacy grants keep read features
    # working (freebusy, listing) but writing requires reconsent. The worker maps
    # the raise to FAILED + dead-letter; the message says what the user must do.
    #
    # Returns a DETACHED snapshot — the session is closed before the Google call
    # below, so no transaction is held across it and no ORM row can be flushed.
    grant = await _load_grant(org_id, "calendar")
    transport, token_manager = _google_transport(grant)
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
            # 409 — this exact ExecutionRequest already produced an event on this
            # calendar. The desired end state HOLDS, so this is a success, not a
            # failure. Confirm the event is really there (rather than trusting
            # the status code) and return its id, so the audit row still carries
            # a real provider reference.
            logger.info(
                "calendar create replayed (409) execution_event_id=%s org=%s",
                event_id,
                org_id,
            )
            event = await client.get_event(event_id)
    except GoogleAPIError as exc:
        # Auth errors are non-retryable at the sender level, and `_reconsent` is
        # what actually makes that true: the DLQ + audit explain "needs
        # reconnect" after ONE attempt rather than after five.
        if exc.is_auth_error:
            raise _reconsent("calendar", "Google Calendar") from exc
        raise
    finally:
        await _persist_refreshed_token(grant, token_manager)
        await transport.aclose()

    created_id = str(event.get("id") or "")
    if not created_id:
        raise RuntimeError("calendar create_event returned no event id")
    # The event object carries the Meet link; returning only the id threw it
    # away. `conference_info` yields {} for a non-conference event, so the
    # reference shape is unchanged when no Meet was requested.
    return created_id, conference_info(event)


# ── Calendar reschedule / cancel senders ───────────────────────────────────


async def _open_calendar(org_id: UUID) -> tuple[CalendarClient, GoogleTransport, TokenManager, _Grant]:
    """Resolve the org's Calendar grant and build a client. NO SESSION HELD.

    Shared by the reschedule / cancel senders so the credential lifecycle lives
    in exactly one place per verb. Returns after the resolving session has
    closed, so the caller's Google call runs with no transaction open.
    """
    grant = await _load_grant(org_id, "calendar")
    transport, token_manager = _google_transport(grant)
    return CalendarClient(transport), transport, token_manager, grant


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
    client, transport, token_manager, grant = await _open_calendar(org_id)
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
            raise _reconsent("calendar", "Google Calendar") from exc
        if exc.is_not_found or exc.is_gone:
            # TERMINAL, and now said in the vocabulary the worker reads. A
            # deleted event does not come back, so retrying this four more times
            # only delays the dead letter that explains it. Cancel needs no
            # equivalent: `delete_event` swallows 404/410 because "already gone"
            # IS the requested end state.
            raise ProviderError(
                f"calendar event {event_id} no longer exists; nothing to "
                "reschedule",
                retryable=False,
            ) from exc
        raise
    finally:
        await _persist_refreshed_token(grant, token_manager)
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
    client, transport, token_manager, grant = await _open_calendar(org_id)
    try:
        await client.delete_event(
            event_id,
            send_updates=(
                "all" if payload.get("send_google_invite", True) else "none"
            ),
        )
    except GoogleAPIError as exc:
        if exc.is_auth_error:
            raise _reconsent("calendar", "Google Calendar") from exc
        raise
    finally:
        await _persist_refreshed_token(grant, token_manager)
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

    # Capability-aware resolution, mirroring calendar_sender. This must NOT be a
    # generic newest-row-wins lookup: on a workspace with two connected Google
    # accounts that picks the most recent grant and then discovers it cannot
    # send, while a send-capable grant sits one row down. The resolver filters on
    # capability FIRST and raises the same actionable message when nothing can
    # send — so an org still connected under the pre-v4 read-only grant gets one
    # clear failure instead of an opaque 403 from inside a worker retry.
    #
    # The session closes HERE. Both Gmail round trips below — the duplicate
    # pre-check and the send — previously ran inside it with no intervening
    # commit, holding a connection idle-in-transaction for the whole exchange.
    grant = await _load_grant(org_id, "gmail")
    from_email = grant.external_account or ""
    if not from_email:
        # No From: address means no send is possible under this grant, ever.
        # Permanent for the same reason a rejected token is.
        raise IntegrationUnavailable(
            "the connected Gmail account has no address on record, so Navis "
            "cannot set a From: header. Reconnect Google Workspace.",
            provider="gmail",
            reconsent_required=True,
        )
    transport, token_manager = _google_transport(grant)
    try:
        client = GmailClient(transport)
        # Has this exact execution already been sent? A retry after a timeout
        # that actually delivered would otherwise send twice.
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
            raise _reconsent("gmail", "Gmail") from exc
        raise
    finally:
        await _persist_refreshed_token(grant, token_manager)
        await transport.aclose()

    message_id = str(sent.get("id") or "")
    if not message_id:
        raise RuntimeError("gmail send returned no message id")
    return message_id


