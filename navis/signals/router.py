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
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from redis.exceptions import RedisError
from sqlalchemy import func
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
from navis.signals.google import oauth as google_oauth
from navis.signals.google.pubsub_auth import (
    PubSubAuthError,
    pubsub_auth_configured,
    verify_pubsub_request,
)
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


@router.get(
    "/integrations/google/connect",
    dependencies=[Depends(require_capability(Capability.GMAIL))],
)
async def google_connect(ctx: CurrentContext = Depends(get_current_context)) -> dict[str, str]:
    state = await _create_oauth_state(
        org_id=ctx.org_id, user_id=ctx.user.id, provider=IntegrationProvider.GMAIL.value
    )
    return {"authorization_url": google_oauth.authorization_url(state)}


@router.get("/integrations/google/callback")
async def google_callback(
    code: str,
    state: str,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    # 1) Validate state FIRST and resolve attribution ONLY from the server-side store.
    bound = await _consume_oauth_state(state)
    org_id, user_id = uuid.UUID(bound["org_id"]), uuid.UUID(bound["user_id"])
    provider = IntegrationProvider.GMAIL

    # 2) Exchange the code (bound to this validated state).
    tokens = await google_oauth.exchange_code(code)

    # 3) Reject incomplete grants — never activate an integration missing required scopes.
    granted = tokens.get("scope")
    missing = _missing_scopes(provider, granted)
    if missing:
        metrics.incr("signals.oauth.scope_rejected", labels={"provider": provider.value})
        logger.warning(
            "oauth scope rejected org=%s user=%s provider=%s missing=%s granted=%s",
            org_id, user_id, provider.value, missing, granted,
        )
        raise HTTPException(status.HTTP_403_FORBIDDEN, "required permissions were not granted")

    # 4) Identify the connected account (the dedup key) from the trusted id_token.
    email = _email_from_id_token(tokens.get("id_token"))
    if not email:
        logger.warning("oauth callback missing account email org=%s user=%s", org_id, user_id)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "could not determine the Google account")

    # 5) UPSERT — uniqueness enforced at the DB on (org, provider, external_account). A
    # reconnect (same account) UPDATES the existing row: no duplicate integrations, no
    # duplicate watches/sync streams/signals. Race-safe (ON CONFLICT, not read-then-insert).
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(tokens.get("expires_in", 3600)))
    enc_access = token_cipher.encrypt(tokens.get("access_token"))
    enc_refresh = token_cipher.encrypt(tokens.get("refresh_token"))
    stmt = (
        pg_insert(Integration)
        .values(
            organization_id=org_id,
            user_id=user_id,
            provider=provider.value,
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
    integration_id = (await db.execute(stmt)).scalar_one()
    metrics.incr("signals.integration.connected", labels={"provider": provider.value})
    logger.info("google integration upserted org=%s user=%s account=%s", org_id, user_id, email)
    return {"integration_id": str(integration_id), "status": "connected"}


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


class SeedSignal(BaseModel):
    source: SignalSource = SignalSource.GMAIL
    event_type: SignalEventType = SignalEventType.EMAIL_RECEIVED
    external_id: str
    external_version: str = "1"
    user_id: uuid.UUID
    integration_id: uuid.UUID
    payload: dict[str, Any] = {}


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
    event = SignalEvent.create(
        SignalContext(org_id=ctx.org_id, user_id=body.user_id, integration_id=body.integration_id),
        source=body.source,
        event_type=body.event_type,
        external_id=body.external_id,
        external_version=body.external_version,
        payload=body.payload,
    )
    inserted = await SignalIngestionService(db).ingest(event)
    return {"ingested": inserted, "event_id": str(event.event_id), "duplicate": not inserted}
