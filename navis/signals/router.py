"""Signals HTTP surface.

  GET  /integrations/google/connect    start OAuth (authenticated)
  GET  /integrations/google/callback   OAuth redirect target -> creates Integration
  POST /webhooks/google/pubsub         Gmail push notification -> enqueue sync
  POST /integrations/seed              dev-only: inject a synthetic SignalEvent

Push handling is deliberately thin: the webhook validates + enqueues a Celery
task. The heavy sync runs out-of-band so the webhook returns in milliseconds.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity.models import Integration, IntegrationProvider
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger
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


@router.get(
    "/integrations/google/connect",
    dependencies=[Depends(require_capability(Capability.GMAIL))],
)
async def google_connect(ctx: CurrentContext = Depends(get_current_context)) -> dict[str, str]:
    # State carries org + user so the callback can attribute the integration.
    state = f"{ctx.org_id}:{ctx.user.id}"
    return {"authorization_url": google_oauth.authorization_url(state)}


@router.get("/integrations/google/callback")
async def google_callback(
    code: str,
    state: str,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        org_id_s, user_id_s = state.split(":", 1)
        org_id, user_id = uuid.UUID(org_id_s), uuid.UUID(user_id_s)
    except (ValueError, AttributeError):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid state")

    tokens = await google_oauth.exchange_code(code)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(tokens.get("expires_in", 3600)))
    integration = Integration(
        organization_id=org_id,
        user_id=user_id,
        provider=IntegrationProvider.GMAIL,
        # Encrypt at rest (no-op passthrough if SIGNALS_TOKEN_KEYS unset).
        access_token=token_cipher.encrypt(tokens.get("access_token")),
        refresh_token=token_cipher.encrypt(tokens.get("refresh_token")),
        token_expires_at=expires_at,
        scopes=tokens.get("scope"),
        is_active=True,
        sync_state={},
    )
    db.add(integration)
    await db.flush()
    logger.info("google integration connected org=%s user=%s", org_id, user_id)
    return {"integration_id": str(integration.id), "status": "connected"}


@router.post("/webhooks/google/pubsub")
async def google_pubsub_webhook(request: Request) -> dict[str, str]:
    """Gmail push. Pub/Sub posts {message:{data:base64(json)}}. We decode the
    target mailbox + historyId and enqueue an incremental sync, then ack 200."""
    # Primary auth: Google's signed OIDC token (rejects forged requests).
    if pubsub_auth_configured():
        try:
            verify_pubsub_request(request.headers.get("Authorization"))
        except PubSubAuthError as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"pubsub auth failed: {exc}")
    else:
        # Fallback: optional shared secret in the query string.
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

    # Enqueue out-of-band; never sync inside the webhook.
    celery_app.send_task(
        "navis.orchestration.sync.sync_google_account",
        kwargs={
            "provider": IntegrationProvider.GMAIL.value,
            "email": decoded.get("emailAddress"),
            "history_id": str(decoded.get("historyId")) if decoded.get("historyId") else None,
        },
    )
    return {"status": "accepted"}


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
    """Dev-only: inject a synthetic, contract-valid SignalEvent through the funnel."""
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
