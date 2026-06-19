"""SyncOrchestrator — drive each connected integration through one sync pass.

Owns the side-effecting glue the signals domain intentionally avoids:
  * resolve a fresh access token (TokenManager) and persist it (encrypted) if refreshed;
  * hold a per-integration PostgreSQL advisory lock so concurrent syncs (push +
    beat + manual) can't race the cursor;
  * read/advance the cursor in `Integration.sync_state` in the SAME session;
  * create/renew Gmail watches and persist watch state.

Signals capture; orchestration persists cursors/tokens/watch. Clean seam.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from sqlalchemy import select

from navis.identity.models import Integration, IntegrationProvider
from navis.platform.database import session_scope
from navis.platform.monitoring import get_logger
from navis.platform.queue import celery_app
from navis.signals.base import SignalContext, WorkspaceConnector
from navis.signals.calendar import CalendarClient, CalendarConnector
from navis.signals.crypto import token_cipher
from navis.signals.drive import DriveClient, DriveConnector
from navis.signals.gmail import GmailClient, GmailConnector
from navis.signals.google.oauth import TokenManager
from navis.signals.google.transport import GoogleTransport
from navis.signals.locking import integration_advisory_lock
from navis.signals.sync import SignalSyncService

logger = get_logger("orchestration.sync")


def _gmail_client(t: GoogleTransport) -> Any:
    return GmailClient(t)


def _calendar_client(t: GoogleTransport) -> Any:
    return CalendarClient(t)


def _drive_client(t: GoogleTransport) -> Any:
    return DriveClient(t)


_REGISTRY: dict[IntegrationProvider, tuple[WorkspaceConnector, Any]] = {
    IntegrationProvider.GMAIL: (GmailConnector(), _gmail_client),
    IntegrationProvider.GOOGLE_CALENDAR: (CalendarConnector(), _calendar_client),
    IntegrationProvider.DRIVE: (DriveConnector(), _drive_client),
}


def _build_client(integration: Integration):
    entry = _REGISTRY.get(integration.provider)
    if entry is None:
        return None, None, None
    connector, client_factory = entry
    token_manager = TokenManager(integration)
    transport = GoogleTransport(token_manager.token_provider())
    client = client_factory(transport)
    return connector, client, (transport, token_manager)


async def sync_integration(integration: Integration) -> dict:
    connector, client, handles = _build_client(integration)
    if connector is None:
        return {"skipped": integration.provider.value}
    transport, token_manager = handles

    if not integration.user_id:
        logger.error("integration %s has no user_id; cannot emit tenant-safe signals", integration.id)
        return {"error": "missing user_id"}

    try:
        async with session_scope() as db:
            # Single-flight: skip if another worker holds this integration.
            async with integration_advisory_lock(db, integration.id) as acquired:
                if not acquired:
                    return {"skipped": "locked"}

                fresh = await db.get(Integration, integration.id)
                if fresh is None or not fresh.is_active:
                    return {"skipped": "inactive"}

                ctx = SignalContext(
                    org_id=fresh.organization_id, user_id=fresh.user_id, integration_id=fresh.id
                )
                cursor = (fresh.sync_state or {}).get("cursor") or {}

                outcome = await SignalSyncService(db).sync(ctx, connector, client, cursor)

                # Persist cursor + refreshed token in the SAME transaction.
                if not outcome.failed:
                    state = dict(fresh.sync_state or {})
                    state["cursor"] = outcome.cursor
                    fresh.sync_state = state
                    if token_manager.last_token and token_manager.last_token.refreshed:
                        fresh.access_token = token_cipher.encrypt(
                            token_manager.last_token.access_token
                        )
                        fresh.token_expires_at = token_manager.last_token.expires_at
    finally:
        await transport.aclose()

    logger.info(
        "synced %s integration=%s ingested=%d dup=%d dlq=%d",
        integration.provider.value,
        integration.id,
        outcome.ingested,
        outcome.duplicates,
        outcome.dead_lettered,
    )
    return {
        "provider": integration.provider.value,
        "ingested": outcome.ingested,
        "duplicates": outcome.duplicates,
        "dead_lettered": outcome.dead_lettered,
    }


async def ensure_watch_for_integration(integration: Integration) -> dict:
    """Create or renew the push watch and persist its state in sync_state['watch']."""
    from navis.signals.reliability import WatchRecoveryService

    connector, client, handles = _build_client(integration)
    if connector is None or not integration.user_id:
        return {"skipped": True}
    transport, _ = handles
    ctx = SignalContext(integration.organization_id, integration.user_id, integration.id)
    try:
        current = (integration.sync_state or {}).get("watch")
        new_state = await WatchRecoveryService().ensure_watch(ctx, connector, client, current)
        if not new_state:
            return {"renewed": False}
        async with session_scope() as db:
            fresh = await db.get(Integration, integration.id)
            if fresh is not None:
                s = dict(fresh.sync_state or {})
                s["watch"] = new_state
                fresh.sync_state = s
        return {"renewed": True}
    finally:
        await transport.aclose()


async def _dispatch_all() -> int:
    """Fan-out: enqueue ONE per-integration task per active integration (paginated)."""
    dispatched, after, page = 0, None, 1000
    while True:
        async with session_scope() as db:
            stmt = (
                select(Integration.id)
                .where(Integration.is_active.is_(True))
                .order_by(Integration.id)
                .limit(page)
            )
            if after is not None:
                stmt = stmt.where(Integration.id > after)
            ids = (await db.scalars(stmt)).all()
        if not ids:
            break
        for iid in ids:
            sync_one_integration.delay(str(iid))
        dispatched += len(ids)
        after = ids[-1]
        if len(ids) < page:
            break
    return dispatched


async def _sync_by_id(integration_id: uuid.UUID) -> dict:
    async with session_scope() as db:
        integration = await db.get(Integration, integration_id)
    if integration is None or not integration.is_active:
        return {"skipped": "inactive"}
    return await sync_integration(integration)


async def _renew_all_watches() -> dict:
    async with session_scope() as db:
        integrations = (
            await db.scalars(
                select(Integration).where(
                    Integration.is_active.is_(True),
                    Integration.provider == IntegrationProvider.GMAIL,
                )
            )
        ).all()
    renewed = 0
    for integ in integrations:
        try:
            if (await ensure_watch_for_integration(integ)).get("renewed"):
                renewed += 1
        except Exception:
            logger.exception("watch renewal failed integration=%s", integ.id)
    return {"checked": len(integrations), "renewed": renewed}


async def _sync_account(provider: str, email: str | None, history_id: str | None) -> dict:
    """Webhook-triggered: sync the specific mailbox the push notification named."""
    async with session_scope() as db:
        stmt = select(Integration).where(
            Integration.provider == IntegrationProvider(provider),
            Integration.is_active.is_(True),
        )
        if email:
            stmt = stmt.where(Integration.external_account == email)
        integration = await db.scalar(stmt)
    if integration is None:
        logger.warning("pubsub sync: no integration for %s/%s", provider, email)
        return {"matched": False}
    return await sync_integration(integration)


@celery_app.task(name="navis.orchestration.sync.sync_one_integration")
def sync_one_integration(integration_id: str) -> dict:
    return asyncio.run(_sync_by_id(uuid.UUID(integration_id)))


@celery_app.task(name="navis.orchestration.sync.sync_all_sources")
def sync_all_sources() -> dict:
    return {"dispatched": asyncio.run(_dispatch_all())}


@celery_app.task(name="navis.orchestration.sync.renew_watches")
def renew_watches() -> dict:
    return asyncio.run(_renew_all_watches())


@celery_app.task(name="navis.orchestration.sync.sync_google_account")
def sync_google_account(provider: str, email: str | None = None, history_id: str | None = None) -> dict:
    return asyncio.run(_sync_account(provider, email, history_id))
