"""Signals HTTP surface.

  GET  /integrations/google/connect    start OAuth (authenticated)
  GET  /integrations/google/callback   OAuth redirect target -> upserts Integration
  POST /webhooks/google/pubsub         Gmail push notification -> enqueue sync (deduped)
  POST /integrations/seed              non-production only: inject a synthetic SignalEvent

Push handling is deliberately thin: the webhook validates + enqueues a Celery task. The
heavy sync runs out-of-band so the webhook returns in milliseconds.

OAuth state is cryptographically random, stored server-side in Redis (10-min TTL,
single-use), and the callback resolves org/user ONLY from that store — never from the
callback query string (CSRF + authorization-code binding per the OAuth spec).
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from redis.exceptions import RedisError
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity.models import Integration, IntegrationProvider
from navis.platform.cache import redis_client
from navis.platform.config import settings
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.platform.queue import celery_app
from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalEventType,
    SignalSource,
)
from navis.signals.crypto import token_cipher
from navis.signals.models import SignalDeadLetter, SignalEventRecord
from navis.signals.google import oauth as google_oauth
from navis.signals.google.oauth import calendar_write_granted
from navis.signals.google.pubsub_auth import (
    PubSubAuthError,
    pubsub_auth_configured,
    verify_pubsub_request,
)
from navis.signals.reliability.slack_health import record_webhook_outcome
from navis.signals.slack.signature import SlackSignatureError, verify_slack_signature
from navis.signals.service import SignalIngestionService

from navis.organization.contract import Capability
from navis.organization.guards import require_capability

logger = get_logger("signals.router")
router = APIRouter(tags=["signals"])

# ---- OAuth state (CSRF + code binding): random, server-side, single-use, expiring ----
_OAUTH_STATE_PREFIX = "oauth:state:"
_OAUTH_STATE_TTL = 600  # 10 minutes
# ---- Pub/Sub redelivery dedup ----
_PUBSUB_DEDUPE_PREFIX = "pubsub:dedupe:"
_PUBSUB_DEDUPE_TTL = 600  # 10 minutes; tune via redelivery window
# ---- Minimum scopes an integration MUST be granted to function (read-only by design) ----
_REQUIRED_SCOPES: dict[IntegrationProvider, tuple[str, ...]] = {
    IntegrationProvider.GMAIL: ("https://www.googleapis.com/auth/gmail.readonly",),
}

# ---- Per-provider activation scopes -----------------------------------------
# The Google consent flow requests Gmail + Calendar + Drive at once, but the
# callback historically only wrote a `gmail` Integration row. That silently
# starved the Calendar and Drive sync loops (orchestration/sync.py dispatches
# per Integration.provider) and made /workspace/calendar/available-slots
# always 409 because there was no `google_calendar` row to look up.
#
# We now upsert one row per Google-family provider whose scope the user
# actually granted, sharing the same tokens and external_account. Missing
# ALL of a provider's activation scopes means we skip that provider's row —
# no fake rows.
#
# Calendar accepts either the new `calendar.events` (v2 grant used by the
# Ask Navis scheduling flow) or the legacy `calendar.readonly` — an org
# that connected under v1 keeps its read-only Calendar integration active
# until the user reconnects to grant write. The sender-side scope check
# in _senders.calendar_sender surfaces reconsent BEFORE any Google call
# when write is required, so a stale read-only row never silently 403s.
_GOOGLE_FAMILY_ACTIVATION_SCOPES: tuple[tuple[IntegrationProvider, tuple[str, ...]], ...] = (
    (IntegrationProvider.GMAIL,            ("https://www.googleapis.com/auth/gmail.readonly",)),
    (
        IntegrationProvider.GOOGLE_CALENDAR,
        (
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/calendar.readonly",
        ),
    ),
    (IntegrationProvider.DRIVE,            ("https://www.googleapis.com/auth/drive.metadata.readonly",)),
)


def _first_activation_scope_present(
    granted_set: set[str], activation_scopes: tuple[str, ...]
) -> str | None:
    """Return the first activation scope actually granted, or None if none are.

    Used at OAuth-callback / backfill time to decide whether to create a
    per-provider Integration row. Keeping the "found" scope lets us log which
    variant was accepted (useful for tracking the v1→v2 Calendar migration).
    """
    for scope in activation_scopes:
        if scope in granted_set:
            return scope
    return None


async def _create_oauth_state(*, org_id: uuid.UUID, user_id: uuid.UUID, provider: str) -> str:
    """Mint a 256-bit random state and bind (org, user, provider, created_at) to it in
    Redis with a TTL. The state string itself carries NO tenant data — it's an opaque,
    unguessable handle, so the callback can't be fed attacker-chosen attribution."""
    state = secrets.token_urlsafe(32)
    payload = json.dumps({
        "org_id": str(org_id),
        "user_id": str(user_id),
        "provider": provider,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    try:
        await redis_client.set(f"{_OAUTH_STATE_PREFIX}{state}", payload, ex=_OAUTH_STATE_TTL)
    except RedisError as exc:  # fail CLOSED: no CSRF token store -> refuse to start OAuth
        logger.error("oauth state store unavailable: %r", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "oauth temporarily unavailable")
    return state


async def _consume_oauth_state(state: str | None) -> dict[str, Any]:
    """Atomically fetch-and-delete the state (GETDEL) — single-use, so replays of a used
    or expired state miss and are rejected. Returns the bound attribution dict."""
    if not state:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "missing oauth state")
    try:
        raw = await redis_client.getdel(f"{_OAUTH_STATE_PREFIX}{state}")
    except RedisError as exc:  # fail CLOSED: can't validate CSRF -> reject
        logger.error("oauth state validate failed: %r", exc)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid oauth state")
    if raw is None:
        # missing / expired (TTL) / already-used (single-use) — indistinguishable on purpose
        metrics.incr("signals.oauth.state_rejected")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid, expired, or used oauth state")
    return json.loads(raw)


def _email_from_id_token(id_token: str | None) -> str | None:
    """Extract the verified email from Google's id_token. The token came directly from
    Google's token endpoint over TLS in the authorization-code flow, so it is trusted; we
    read the `email` claim from the JWT payload to identify the connected account."""
    if not id_token:
        return None
    try:
        payload_b64 = id_token.split(".")[1]
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
        return claims.get("email")
    except (IndexError, binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _missing_scopes(provider: IntegrationProvider, granted: str | None) -> list[str]:
    have = set((granted or "").split())
    return [s for s in _REQUIRED_SCOPES.get(provider, ()) if s not in have]


async def _backfill_google_family(
    db: AsyncSession,
    *,
    org_id: uuid.UUID,
) -> list[str]:
    """One-shot self-heal for orgs that connected Google BEFORE the callback
    started writing separate Calendar/Drive rows. Reads the Gmail row's
    granted scopes and clones it (same tokens, same account) into whichever
    Google-family provider rows are missing.

    Idempotent (ON CONFLICT DO NOTHING — never clobbers a row that already
    exists), and shares tokens on purpose because a single Google OAuth
    grant covers every scope in the request. Returns the provider values
    that were freshly inserted so the caller can log / metric.
    """
    gmail_row = await db.scalar(
        select(Integration).where(
            Integration.organization_id == org_id,
            Integration.provider == IntegrationProvider.GMAIL,
            Integration.is_active.is_(True),
        )
    )
    if gmail_row is None:
        return []
    granted_set = set((gmail_row.scopes or "").split())
    inserted: list[str] = []
    for family_provider, activation_scopes in _GOOGLE_FAMILY_ACTIVATION_SCOPES:
        if family_provider is IntegrationProvider.GMAIL:
            continue  # Gmail row is the source of truth we're copying from.
        if _first_activation_scope_present(granted_set, activation_scopes) is None:
            continue  # scope not granted → nothing to clone (correct: user
            # never authorised this service, so we shouldn't fake a row).
        stmt = (
            pg_insert(Integration)
            .values(
                organization_id=org_id,
                user_id=gmail_row.user_id,
                provider=family_provider.value,
                external_account=gmail_row.external_account,
                access_token=gmail_row.access_token,
                refresh_token=gmail_row.refresh_token,
                token_expires_at=gmail_row.token_expires_at,
                scopes=gmail_row.scopes,
                is_active=True,
                sync_state={},
            )
            .on_conflict_do_nothing(constraint="uq_org_provider_account")
            .returning(Integration.id)
        )
        result = await db.execute(stmt)
        if result.scalar() is not None:
            inserted.append(family_provider.value)
    return inserted


@router.get(
    "/integrations/google/connect",
    dependencies=[Depends(require_capability(Capability.GMAIL))],
)
async def google_connect(ctx: CurrentContext = Depends(get_current_context)) -> dict[str, str]:
    state = await _create_oauth_state(
        org_id=ctx.org_id, user_id=ctx.user.id, provider=IntegrationProvider.GMAIL.value
    )
    return {"authorization_url": google_oauth.authorization_url(state)}


def _frontend_return_url(*, ok: bool, provider: str, reason: str | None = None) -> str:
    """Where the browser lands after the OAuth code exchange finishes.

    Success -> `${frontend_url}/dashboard/account?connected=<provider>`
    Failure -> `${frontend_url}/dashboard/account?error=<code>`

    Keeping the return URL out of the OAuth state store (which stays purely for
    CSRF binding) means the callback never has to trust anything the client
    sent — the destination is a server-side setting.
    """
    base = settings.frontend_url.rstrip("/")
    if ok:
        return f"{base}/dashboard/account?connected={quote(provider)}"
    return f"{base}/dashboard/account?error={quote(reason or 'oauth_failed')}"


@router.get("/integrations/google/callback")
async def google_callback(
    code: str,
    state: str,
    db: AsyncSession = Depends(get_session),
) -> RedirectResponse:
    provider = IntegrationProvider.GMAIL

    # 1) Validate state FIRST and resolve attribution ONLY from the server-side store.
    # State errors bounce the browser back with an error code — never a raw 400 page,
    # or the founder is stuck on a JSON blob with no way home.
    try:
        bound = await _consume_oauth_state(state)
    except HTTPException:
        return RedirectResponse(
            _frontend_return_url(ok=False, provider=provider.value, reason="invalid_state"),
            status_code=status.HTTP_303_SEE_OTHER,
        )
    org_id, user_id = uuid.UUID(bound["org_id"]), uuid.UUID(bound["user_id"])

    # 2) Exchange the code (bound to this validated state).
    try:
        tokens = await google_oauth.exchange_code(code)
    except Exception as exc:  # noqa: BLE001 — surface any exchange failure to the UI
        logger.warning("oauth code exchange failed org=%s user=%s err=%r", org_id, user_id, exc)
        metrics.incr("signals.oauth.exchange_failed", labels={"provider": provider.value})
        return RedirectResponse(
            _frontend_return_url(ok=False, provider=provider.value, reason="exchange_failed"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    # 3) Reject incomplete grants — never activate an integration missing required scopes.
    granted = tokens.get("scope")
    missing = _missing_scopes(provider, granted)
    if missing:
        metrics.incr("signals.oauth.scope_rejected", labels={"provider": provider.value})
        logger.warning(
            "oauth scope rejected org=%s user=%s provider=%s missing=%s granted=%s",
            org_id, user_id, provider.value, missing, granted,
        )
        return RedirectResponse(
            _frontend_return_url(ok=False, provider=provider.value, reason="missing_scopes"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    # 4) Identify the connected account (the dedup key) from the trusted id_token.
    email = _email_from_id_token(tokens.get("id_token"))
    if not email:
        logger.warning("oauth callback missing account email org=%s user=%s", org_id, user_id)
        return RedirectResponse(
            _frontend_return_url(ok=False, provider=provider.value, reason="no_email_claim"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    # 5) UPSERT one row PER Google-family provider whose scope was granted.
    # Uniqueness enforced at the DB on (org, provider, external_account) — a
    # reconnect (same account) UPDATES each row: no duplicate integrations, no
    # duplicate watches/sync streams/signals. Race-safe (ON CONFLICT, not
    # read-then-insert). Sharing tokens across rows is correct because a single
    # Google OAuth grant covers every scope in the request; each provider row
    # just makes its service independently addressable to the sync orchestrator
    # and the per-provider workspace endpoints.
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(tokens.get("expires_in", 3600)))
    enc_access = token_cipher.encrypt(tokens.get("access_token"))
    enc_refresh = token_cipher.encrypt(tokens.get("refresh_token"))
    granted_set = set((granted or "").split())

    activated: list[str] = []
    for family_provider, activation_scopes in _GOOGLE_FAMILY_ACTIVATION_SCOPES:
        if _first_activation_scope_present(granted_set, activation_scopes) is None:
            continue  # user didn't grant this service — skip; no fake row.
        stmt = (
            pg_insert(Integration)
            .values(
                organization_id=org_id,
                user_id=user_id,
                provider=family_provider.value,
                external_account=email,
                access_token=enc_access,
                refresh_token=enc_refresh,
                token_expires_at=expires_at,
                scopes=granted,
                is_active=True,
                sync_state={},
            )
            .on_conflict_do_update(
                constraint="uq_org_provider_account",
                set_={
                    "user_id": user_id,
                    "access_token": enc_access,
                    # Google omits a refresh_token on re-consent — keep the stored one if so.
                    "refresh_token": func.coalesce(enc_refresh, Integration.refresh_token),
                    "token_expires_at": expires_at,
                    "scopes": granted,
                    "is_active": True,
                    "updated_at": func.now(),
                },
            )
            .returning(Integration.id)
        )
        await db.execute(stmt)
        activated.append(family_provider.value)
        metrics.incr(
            "signals.integration.connected", labels={"provider": family_provider.value}
        )

    logger.info(
        "google integrations upserted org=%s user=%s account=%s providers=%s",
        org_id, user_id, email, activated,
    )
    return RedirectResponse(
        _frontend_return_url(ok=True, provider=provider.value),
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/webhooks/google/pubsub")
async def google_pubsub_webhook(request: Request) -> dict[str, str]:
    """Gmail push. Pub/Sub posts {message:{data:base64(json)}}. We decode the target
    mailbox + historyId and enqueue an incremental sync (deduped), then ack 200."""
    # Primary auth: Google's signed OIDC token (rejects forged requests).
    if pubsub_auth_configured():
        try:
            verify_pubsub_request(request.headers.get("Authorization"))
        except PubSubAuthError as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"pubsub auth failed: {exc}")
    else:
        token = request.query_params.get("token")
        expected = os.environ.get("GOOGLE_WEBHOOK_VERIFICATION_TOKEN")
        if expected and token != expected:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad webhook token")

    body = await request.json()
    message = (body or {}).get("message", {})
    raw = message.get("data")
    decoded: dict[str, Any] = {}
    if raw:
        try:
            decoded = json.loads(base64.b64decode(raw).decode("utf-8"))
        except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("undecodable pubsub message id=%s", message.get("messageId"))
            return {"status": "ignored"}  # ack so Pub/Sub stops retrying a poison message

    email = decoded.get("emailAddress")
    history_id = str(decoded.get("historyId")) if decoded.get("historyId") else None

    # Dedup redelivery storms: Pub/Sub is at-least-once and resends the same (email,
    # historyId). One enqueue per (email, historyId) within the TTL window, multi-worker
    # safe via Redis SET NX. Fails OPEN (process) if Redis is down — a re-sync is
    # idempotent, so a possible duplicate beats dropping a notification.
    if await _is_duplicate_pubsub(email, history_id):
        metrics.incr("signals.pubsub.deduped")
        return {"status": "duplicate"}

    celery_app.send_task(
        "navis.orchestration.sync.sync_google_account",
        kwargs={
            "provider": IntegrationProvider.GMAIL.value,
            "email": email,
            "history_id": history_id,
        },
    )
    return {"status": "accepted"}


async def _is_duplicate_pubsub(email: str | None, history_id: str | None) -> bool:
    if not email or not history_id:
        return False  # nothing to key on -> process (rare; sync is idempotent anyway)
    key = f"{_PUBSUB_DEDUPE_PREFIX}{email}:{history_id}"
    try:
        was_new = await redis_client.set(key, "1", nx=True, ex=_PUBSUB_DEDUPE_TTL)
    except RedisError:  # fail OPEN: process rather than drop a webhook
        metrics.incr("signals.pubsub.dedupe_redis_error")
        return False
    return not was_new  # already present within TTL -> duplicate


@router.post("/webhooks/slack/events")
async def slack_events_webhook(request: Request) -> Any:
    """Slack Events API push. Verify HMAC + timestamp, dedupe the redelivery, enqueue, ACK —
    NEVER process inline (Slack needs <3s; we target <500ms). The Celery processor does the
    Integration lookup + mapping + ingest out of band."""
    body = await request.body()
    hour = int(time.time()) // 3600
    # Best-effort team for health attribution ONLY (untrusted until verified; spoofing here can
    # at worst cause a false ops alert, never a security or data effect).
    team_hint = _best_effort_team(body)
    try:
        verify_slack_signature(
            body=body,
            timestamp=request.headers.get("X-Slack-Request-Timestamp"),
            signature=request.headers.get("X-Slack-Signature"),
        )
    except SlackSignatureError as exc:
        metrics.incr("slack.webhook.rejected", labels={"reason": exc.reason})
        await record_webhook_outcome(team_hint, ok=False, hour=hour)  # a real-Slack 401 burst -> auto-disable risk
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"slack signature failed: {exc}")

    try:
        envelope = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        await record_webhook_outcome(team_hint, ok=True, hour=hour)  # verified Slack; we ack (2xx)
        return {"status": "ignored"}  # ack so Slack stops retrying an undecodable body

    await record_webhook_outcome(envelope.get("team_id"), ok=True, hour=hour)  # verified + ACKed

    if envelope.get("type") == "url_verification":  # one-time endpoint handshake (also signed)
        return {"challenge": envelope.get("challenge")}

    # At-least-once: Slack resends the SAME event_id on its retries -> suppress before enqueue.
    if await _is_duplicate_slack(envelope.get("team_id"), envelope.get("event_id")):
        metrics.incr("slack.webhook.deduped")
        return {"status": "duplicate"}

    celery_app.send_task(
        "navis.orchestration.sync.process_slack_event",
        kwargs={"team_id": envelope.get("team_id"), "envelope": envelope},
    )
    metrics.incr("slack.webhook.accepted")
    return {"status": "accepted"}


def _best_effort_team(body: bytes) -> str | None:
    try:
        return (json.loads(body.decode("utf-8")) or {}).get("team_id")
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return None


async def _is_duplicate_slack(team_id: str | None, event_id: str | None) -> bool:
    if not team_id or not event_id:
        return False  # nothing to key on -> process (mapping is idempotent anyway)
    key = f"slack:dedupe:{team_id}:{event_id}"
    try:
        was_new = await redis_client.set(key, "1", nx=True, ex=_PUBSUB_DEDUPE_TTL)
    except RedisError:  # fail OPEN: process rather than drop a webhook (ingest is idempotent)
        metrics.incr("slack.webhook.dedupe_redis_error")
        return False
    return not was_new


@router.get("/integrations")
async def list_integrations(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """List the Integration rows for the current org — never the tokens.

    The response also carries per-integration ingest counts:
      * ``ingested_today``      — signal_events since UTC midnight
      * ``ingested_last_hour``  — signal_events in the trailing 60 min
      * ``dead_letter_count``   — unresolved DLQ rows

    The frontend Sources rail merges this with a static provider registry
    (Gmail/Calendar/Drive/Slack) so unconnected providers still get a
    Connect card. Because Google OAuth returns a single Integration row
    that grants Gmail + Calendar + Drive at once, the frontend treats
    the Gmail provider row as authoritative for the whole Google family.
    """
    # Time boundaries, computed once so both counter columns share the
    # same "now". UTC everywhere — signal_events.captured_at is stored
    # tz-aware and comparing to naive now() would raise.
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    last_hour = now - timedelta(hours=1)

    # Fetch integrations + per-integration counters in parallel-ish
    # (SQLAlchemy still runs them sequentially on one connection, but
    # keeping them in this function means one round-trip per query, not
    # one per integration). Indexes used: ix_signal_integration for
    # signal_events, ix_dlq_integration for signal_dead_letters —
    # both are tenant-scoped when combined with the org filter.
    rows_result = await db.execute(
        select(Integration)
        .where(Integration.organization_id == ctx.org_id)
        .order_by(Integration.provider, Integration.updated_at.desc())
    )
    # Third counter (no filter) = all-time ingestion for this integration.
    # The Account page's "Signals contributed · today / total" chip reads it
    # so the total isn't a hardcoded frontend mock (it was 1247 in
    # mockConnections). Same query, same index scan, one extra aggregate —
    # no additional round-trip.
    ingest_counts_result = await db.execute(
        select(
            SignalEventRecord.integration_id,
            func.count()
                .filter(SignalEventRecord.captured_at >= today_start)
                .label("ingested_today"),
            func.count()
                .filter(SignalEventRecord.captured_at >= last_hour)
                .label("ingested_last_hour"),
            func.count().label("ingested_total"),
        )
        .where(SignalEventRecord.organization_id == ctx.org_id)
        .where(SignalEventRecord.integration_id.is_not(None))
        .group_by(SignalEventRecord.integration_id)
    )
    dlq_counts_result = await db.execute(
        select(
            SignalDeadLetter.integration_id,
            func.count().label("dead_letter_count"),
        )
        .where(SignalDeadLetter.organization_id == ctx.org_id)
        .where(SignalDeadLetter.resolved_at.is_(None))
        .where(SignalDeadLetter.integration_id.is_not(None))
        .group_by(SignalDeadLetter.integration_id)
    )

    ingest_by_integration: dict[uuid.UUID, tuple[int, int, int]] = {
        integration_id: (int(today or 0), int(last_hour_count or 0), int(total or 0))
        for integration_id, today, last_hour_count, total in ingest_counts_result.all()
    }
    dlq_by_integration: dict[uuid.UUID, int] = {
        integration_id: int(count or 0)
        for integration_id, count in dlq_counts_result.all()
    }

    integrations: list[dict[str, Any]] = []
    for r in rows_result.scalars().all():
        # sync_state is provider-specific; expose only a small set of
        # widely-supported keys to the UI. Anything else stays on the
        # server (avoids accidentally leaking a cursor token that's
        # opaque but sensitive).
        state = r.sync_state or {}
        last_synced_at = state.get("last_synced_at") if isinstance(state, dict) else None
        ingested_today, ingested_last_hour, ingested_total = ingest_by_integration.get(
            r.id, (0, 0, 0)
        )
        provider_value = r.provider.value if hasattr(r.provider, "value") else str(r.provider)
        # Capability flags derived from the stored grant. Frontend uses
        # these to gate features that need a specific write scope without
        # having to parse the raw scope string. Currently only Calendar
        # write matters — Gmail send is deliberately not requested by
        # this iteration (see CLAUDE.md scope narrowing).
        calendar_can_write = (
            provider_value == IntegrationProvider.GOOGLE_CALENDAR.value
            and calendar_write_granted(r.scopes)
        )
        integrations.append({
            "integration_id":     str(r.id),
            "provider":           provider_value,
            "external_account":   r.external_account,
            "is_active":          r.is_active,
            "connected_at":       r.created_at.isoformat() if r.created_at else None,
            "updated_at":         r.updated_at.isoformat() if r.updated_at else None,
            "token_expires_at":   r.token_expires_at.isoformat() if r.token_expires_at else None,
            # Space-separated scope string — the frontend can split for display.
            "scopes":             r.scopes,
            # True when this integration carries a Calendar write scope.
            # False on legacy `calendar.readonly` grants — the frontend
            # surfaces a reconsent prompt before allowing scheduling.
            "calendar_write_scope": calendar_can_write,
            # Opportunistic; may be None until the sync engine writes it.
            "last_synced_at":     last_synced_at,
            # Per-connector ingestion counters — zero when the connector
            # has never produced a signal (or when the events are older
            # than the time window we count). `ingested_total` is all-time
            # for this integration (no time filter).
            "ingested_today":     ingested_today,
            "ingested_last_hour": ingested_last_hour,
            "ingested_total":     ingested_total,
            "dead_letter_count":  dlq_by_integration.get(r.id, 0),
        })
    return {"integrations": integrations}


class SeedSignal(BaseModel):
    source: SignalSource = SignalSource.GMAIL
    event_type: SignalEventType = SignalEventType.EMAIL_RECEIVED
    external_id: str
    external_version: str = "1"
    # user_id and integration_id are both optional now. When omitted:
    #   * user_id defaults to the caller's user id (from the auth ctx).
    #   * integration_id is looked up from the org's active integrations
    #     matching `source`. If none exists, we fall back to NULL — the
    #     signal_events FK is defined as ON DELETE SET NULL so a null
    #     integration_id is a first-class value, NOT a bug.
    # Previously we required both fields and callers were passing random
    # UUIDs, which tripped the integrations.id FK and 500'd the endpoint.
    user_id: uuid.UUID | None = None
    integration_id: uuid.UUID | None = None
    payload: dict[str, Any] = {}


# Signal source → provider mapping used by seed to auto-resolve the org's
# integration row when the caller didn't specify one.
_SEED_SOURCE_TO_PROVIDER: dict[SignalSource, IntegrationProvider] = {
    SignalSource.GMAIL: IntegrationProvider.GMAIL,
    SignalSource.CALENDAR: IntegrationProvider.GOOGLE_CALENDAR,
    SignalSource.DRIVE: IntegrationProvider.DRIVE,
    SignalSource.SLACK: IntegrationProvider.SLACK,
}


@router.post(
    "/integrations/seed",
    status_code=201,
    dependencies=[Depends(require_capability(Capability.GMAIL))],
)
async def seed_signal(
    body: SeedSignal,
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """NON-PRODUCTION ONLY: inject a synthetic, contract-valid SignalEvent through the
    funnel. Disabled in production (404, no info leak) so it can't be used for signal
    poisoning."""
    if settings.is_production:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")

    # Resolve user_id — default to the calling principal.
    user_id = body.user_id or ctx.user.id

    # Resolve integration_id. Order of preference:
    #   1. Caller-supplied value — but only if it references a real row
    #      owned by this org (prevents cross-tenant poisoning and also
    #      the "random UUID -> FK violation -> 500" footgun).
    #   2. Any active integration in this org matching `source`.
    #   3. A synthetic dev-seed row created on the fly. Necessary because
    #      SignalContext requires a non-null integration_id (see
    #      signals/base.py::SignalContext.__post_init__) — seeding
    #      without one would raise SignalValidationError.
    integration_id: uuid.UUID | None = None
    if body.integration_id is not None:
        row = await db.scalar(
            select(Integration).where(
                Integration.id == body.integration_id,
                Integration.organization_id == ctx.org_id,
            )
        )
        if row is not None:
            integration_id = row.id
    if integration_id is None:
        provider = _SEED_SOURCE_TO_PROVIDER.get(body.source, IntegrationProvider.GMAIL)
        # Prefer a real, active Google integration if one exists.
        row = await db.scalar(
            select(Integration).where(
                Integration.organization_id == ctx.org_id,
                Integration.provider == provider,
                Integration.is_active.is_(True),
            )
        )
        if row is None:
            # No real integration → look for the sticky dev-seed row we
            # may have created on a previous seed call. The unique
            # constraint (org_id, provider, external_account) means we
            # can't just INSERT a second one; we have to UPSERT.
            row = await db.scalar(
                select(Integration).where(
                    Integration.organization_id == ctx.org_id,
                    Integration.provider == provider,
                    Integration.external_account == "dev-seed",
                )
            )
        if row is None:
            # First seed for this (org, provider). Create the synthetic
            # dev-only row. Marked with external_account="dev-seed" so
            # it's easy to spot / prune later. is_active=False so
            # periodic sync tasks won't try to hit Google with fake
            # tokens.
            row = Integration(
                organization_id=ctx.org_id,
                user_id=user_id,
                provider=provider,
                external_account="dev-seed",
                access_token=None,
                refresh_token=None,
                token_expires_at=None,
                scopes=None,
                is_active=False,
                sync_state={},
            )
            db.add(row)
            await db.flush()  # obtain row.id without committing
        integration_id = row.id

    event = SignalEvent.create(
        SignalContext(org_id=ctx.org_id, user_id=user_id, integration_id=integration_id),
        source=body.source,
        event_type=body.event_type,
        external_id=body.external_id,
        external_version=body.external_version,
        payload=body.payload,
    )
    try:
        inserted = await SignalIngestionService(db).ingest(event)
    except Exception as exc:  # ensure a failed insert doesn't poison the session
        await db.rollback()
        logger.exception("seed signal ingest failed org=%s src=%s", ctx.org_id, body.source.value)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"seed ingestion failed: {exc.__class__.__name__}: {str(exc)[:400]}",
        )
    # Fire the bus event as a background task so the seed request
    # returns immediately. Subscribers (memory, state, brief, onboarding
    # advance) can take seconds to complete their DB work; awaiting them
    # inline turned the seed endpoint into a 14-second call. The outbox
    # row is durable so a real Celery relay would eventually re-publish;
    # this background dispatch just makes local dev pipelines advance
    # without a worker running.
    if inserted:
        import asyncio

        from navis.platform.events import DomainEvent, event_bus
        from navis.signals.service import event_type_for

        domain_event = DomainEvent(
            type=event_type_for(event),
            org_id=ctx.org_id,
            source="signals",
            payload={
                "event_id": str(event.event_id),
                "source": event.source.value,
                "event_type": event.event_type.value,
                "external_id": event.external_id,
                "external_version": event.external_version,
                "user_id": str(event.user_id),
                "integration_id": str(event.integration_id),
            },
        )

        async def _fire() -> None:
            try:
                await event_bus.publish(domain_event)
            except Exception:
                logger.exception("seed background publish failed org=%s", ctx.org_id)

        asyncio.create_task(_fire())

    return {
        "ingested": inserted,
        "event_id": str(event.event_id),
        "duplicate": not inserted,
        "integration_id": str(integration_id) if integration_id else None,
    }


@router.delete("/integrations/slack/disconnect")
async def slack_disconnect(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Soft-disconnect all Slack integrations for this org.

    Sets `is_active=False` — we do NOT delete the Integration row so:
      * historical SignalEvents still resolve their integration provenance,
      * a re-connect UPSERTs into the same row (no duplicate history),
      * the Celery sync task's `is_active` filter stops touching the account
        immediately without a redeploy.
    Tokens are left encrypted in place; re-connect overwrites them. Broadcasts
    INTEGRATION_DISCONNECTED so onboarding/capability gates recompute."""
    from sqlalchemy import update

    stmt = (
        update(Integration)
        .where(
            Integration.organization_id == ctx.org_id,
            Integration.provider == IntegrationProvider.SLACK,
            Integration.is_active.is_(True),
        )
        .values(is_active=False, updated_at=func.now())
        .returning(Integration.id)
    )
    result = await db.execute(stmt)
    deactivated = [row[0] for row in result.all()]
    await db.commit()

    if deactivated:
        from navis.platform.events import DomainEvent, EventType, event_bus

        for integration_id in deactivated:
            await event_bus.publish(DomainEvent(
                type=EventType.INTEGRATION_DISCONNECTED,
                org_id=ctx.org_id,
                source="signals.router",
                payload={
                    "integration_id": str(integration_id),
                    "provider": IntegrationProvider.SLACK.value,
                },
            ))
        metrics.incr(
            "signals.integration.disconnected",
            labels={"provider": IntegrationProvider.SLACK.value},
            value=len(deactivated),
        )
        logger.info(
            "slack integrations disconnected org=%s count=%d", ctx.org_id, len(deactivated)
        )

    return {"disconnected": True, "count": len(deactivated)}


@router.delete("/integrations/google/disconnect")
async def google_disconnect(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Soft-disconnect all Google-family integrations for this org.

    Mirrors the Slack disconnect contract:
      * flips is_active=False on every Google-family row (Gmail /
        Google Calendar / Drive) so the sync engine's is_active filter
        stops touching the account without a redeploy;
      * revokes the refresh token at Google (best-effort — failure is
        logged but does NOT block the local disconnect, because our DB
        row is already dead);
      * clears access_token / refresh_token bytes so a subsequent leak
        or dump doesn't expose credentials for a connection the user
        no longer wants;
      * clears sync_state cursors so a reconnect starts a clean window
        instead of resuming a historyId the user thought they wiped;
      * publishes INTEGRATION_DISCONNECTED per row so onboarding /
        capability gates recompute and dependents (memory, brief,
        actions) recompute their pipelines.

    The Integration row itself is NOT deleted — historical SignalEvents
    reference it via FK, and a re-connect UPSERTs into the same row to
    avoid duplicate history / dangling references. Same rationale as
    Slack.
    """
    from sqlalchemy import update

    from navis.signals.crypto import token_cipher
    from navis.signals.google.oauth import revoke_token as revoke_google_token

    GOOGLE_FAMILY = (
        IntegrationProvider.GMAIL,
        IntegrationProvider.GOOGLE_CALENDAR,
        IntegrationProvider.DRIVE,
    )

    # Collect refresh tokens BEFORE zeroing them — we need the plaintext
    # to revoke at Google's endpoint. Best-effort decrypt; a failure here
    # (rotated-out key, corrupted token) still lets the local disconnect
    # proceed. The user's ground truth is our DB, not Google's grant.
    rows = (
        await db.scalars(
            select(Integration).where(
                Integration.organization_id == ctx.org_id,
                Integration.provider.in_(GOOGLE_FAMILY),
                Integration.is_active.is_(True),
            )
        )
    ).all()

    if not rows:
        return {"disconnected": True, "count": 0, "revoked": 0}

    refresh_tokens: list[str] = []
    for row in rows:
        try:
            plaintext = token_cipher.decrypt(row.refresh_token)
        except Exception:
            logger.exception(
                "could not decrypt refresh_token for revoke integration=%s (continuing)",
                row.id,
            )
            plaintext = None
        if plaintext:
            refresh_tokens.append(plaintext)

    # Flip is_active + zero tokens + clear cursors, atomically per row.
    # Using a bulk update means the sync worker sees the flip in the
    # same transaction the API caller commits — no race window where
    # a scheduled sync fires on a "disconnected" row.
    stmt = (
        update(Integration)
        .where(
            Integration.organization_id == ctx.org_id,
            Integration.provider.in_(GOOGLE_FAMILY),
            Integration.is_active.is_(True),
        )
        .values(
            is_active=False,
            access_token=None,
            refresh_token=None,
            sync_state={},
            updated_at=func.now(),
        )
        .returning(Integration.id, Integration.provider)
    )
    result = await db.execute(stmt)
    deactivated = list(result.all())
    await db.commit()

    # Revoke at Google — best-effort, deduplicated by token. Multiple
    # Google-family rows on a single account will share one refresh
    # token, so revoking once kills the whole grant. Dedup avoids a
    # second call that would return HTTP 400 "already invalid".
    revoked_count = 0
    seen: set[str] = set()
    for token in refresh_tokens:
        if token in seen:
            continue
        seen.add(token)
        ok = await revoke_google_token(token)
        if ok:
            revoked_count += 1

    # Broadcast per-row so downstream subscribers (onboarding gate,
    # capability recomputer, cache-invalidation SSE fanout) all fire.
    if deactivated:
        from navis.platform.events import DomainEvent, EventType, event_bus

        for integration_id, provider in deactivated:
            await event_bus.publish(
                DomainEvent(
                    type=EventType.INTEGRATION_DISCONNECTED,
                    org_id=ctx.org_id,
                    source="signals.router",
                    payload={
                        "integration_id": str(integration_id),
                        "provider": provider.value,
                    },
                )
            )
        metrics.incr(
            "signals.integration.disconnected",
            labels={"provider": "google"},
            value=len(deactivated),
        )
        logger.info(
            "google integrations disconnected org=%s count=%d revoked=%d",
            ctx.org_id,
            len(deactivated),
            revoked_count,
        )

    return {
        "disconnected": True,
        "count": len(deactivated),
        "revoked": revoked_count,
    }


@router.post(
    "/integrations/google/sync",
    dependencies=[Depends(require_capability(Capability.GMAIL))],
)
async def sync_google_integration(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Manually drive one sync pass over the org's active Google integrations.

    Normally sync runs out-of-band via `navis.orchestration.sync.sync_all_sources`
    (Celery beat) + Gmail Pub/Sub push. Neither is running in local dev unless the
    operator has provisioned a broker + a Google Cloud subscription — so without
    this endpoint a fresh Gmail-connected demo has no way to ingest anything and
    the whole pipeline (memory -> state -> brief -> execution) stays cold.

    This route runs the same `sync_integration` orchestrator the beat task uses,
    just inline. Safe to call repeatedly — each integration takes an advisory
    lock, so concurrent presses are single-flighted rather than racing.
    """
    from navis.orchestration.sync import sync_integration

    # Self-heal legacy orgs that connected Google before the callback started
    # writing per-provider rows: if the Gmail row grants calendar/drive
    # scopes but the matching rows are missing, create them from the Gmail
    # row's tokens. Idempotent — no-op once the rows exist.
    backfilled = await _backfill_google_family(db, org_id=ctx.org_id)
    if backfilled:
        await db.commit()
        logger.info(
            "google family backfilled org=%s providers=%s", ctx.org_id, backfilled
        )
        for provider_value in backfilled:
            metrics.incr(
                "signals.integration.backfilled", labels={"provider": provider_value}
            )

    rows = (
        await db.scalars(
            select(Integration).where(
                Integration.organization_id == ctx.org_id,
                Integration.is_active.is_(True),
                Integration.provider.in_(
                    (
                        IntegrationProvider.GMAIL,
                        IntegrationProvider.GOOGLE_CALENDAR,
                        IntegrationProvider.DRIVE,
                    )
                ),
            )
        )
    ).all()

    if not rows:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "no active Google integration to sync — connect one first",
        )

    results: list[dict[str, Any]] = []
    total_ingested = 0
    for integration in rows:
        try:
            outcome = await sync_integration(integration)
        except Exception as exc:  # keep the endpoint's contract simple: never 500 mid-loop
            logger.exception("manual sync failed integration=%s", integration.id)
            results.append(
                {
                    "integration_id": str(integration.id),
                    "provider": integration.provider.value,
                    "error": str(exc),
                }
            )
            continue
        outcome = {**outcome, "integration_id": str(integration.id)}
        total_ingested += int(outcome.get("ingested") or 0)
        results.append(outcome)

    # Drain both outboxes in the background so the just-ingested signals
    # get published to the bus (advancing onboarding stage / lighting up
    # capabilities) even without a Celery worker running. Same rationale
    # as the seed endpoint — real Celery relay would eventually catch up
    # anyway; this just makes local dev demos feel live.
    #
    # Order matters: draining signals runs the in-process memory subscriber,
    # which STAGES memory outbox pointers in its own commit. The memory relay
    # then publishes those, which is what advances the onboarding funnel from
    # `first_signal` to `memory_ready` (and further). Skipping the memory pass
    # was the reason the pipeline stalled at Building Memory in dev.
    #
    # We also drain memory unconditionally: earlier syncs may have staged
    # pointers that never published because that session's Celery relay
    # wasn't running — a fresh press should clear the backlog.
    import asyncio

    async def _drain() -> None:
        try:
            from navis.memory.outbox import relay_memory_pass
            from navis.memory.repository import MemoryRepository
            from navis.platform.database import session_scope
            from navis.signals.outbox import relay_signals_pass
            from navis.signals.repository import SignalRepository

            if total_ingested > 0:
                async with session_scope() as bg_db:
                    await relay_signals_pass(SignalRepository(bg_db))
            async with session_scope() as bg_db:
                await relay_memory_pass(MemoryRepository(bg_db))
        except Exception:
            logger.exception("sync background outbox drain failed org=%s", ctx.org_id)

    asyncio.create_task(_drain())

    return {"synced": len(results), "total_ingested": total_ingested, "results": results}
