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

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from navis.identity.models import Integration, IntegrationAuthState, IntegrationProvider
from navis.orchestration.runtime_policy import scheduled_work_allowed
from navis.organization.contract import Capability
from navis.platform.database import session_scope
from navis.platform.monitoring import get_logger, org_context
from navis.platform.monitoring.metrics import metrics
from navis.platform.queue import celery_app, run_async
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

# Which entitlement a connector's ingestion consumes (T-14). Separate from
# `_REGISTRY` on purpose: that maps a provider to the code that talks to it, this
# maps a provider to the permission to do so at all, and Slack has the second
# without the first (its capture is webhook-driven, not polled).
#
# A provider absent from this map CANNOT BE SYNCED — `_capability_for` returns
# None and `sync_integration` refuses. That is the fail-closed direction: an
# unmapped connector is one nobody has decided the entitlement for yet, and
# guessing would let a future connector run outside the plan that pays for it.
# GOOGLE_MEET maps to CALENDAR because Meet records are derived from calendar
# events rather than fetched from a separate grant.
_CAPABILITY: dict[IntegrationProvider, Capability] = {
    IntegrationProvider.GMAIL: Capability.GMAIL,
    IntegrationProvider.GOOGLE_CALENDAR: Capability.CALENDAR,
    IntegrationProvider.GOOGLE_MEET: Capability.CALENDAR,
    IntegrationProvider.DRIVE: Capability.DRIVE,
    IntegrationProvider.SLACK: Capability.SLACK,
}


def _capability_for(provider: IntegrationProvider) -> Capability | None:
    return _CAPABILITY.get(provider)


def _utcnow() -> datetime:
    """Wall clock, timezone-aware. One reading per sync pass.

    Taken once and reused for `last_attempt_at` and `last_success_at` so the two
    cannot disagree by the duration of a database round trip — a success stamped
    microseconds before the attempt that produced it reads as a clock bug.
    """
    return datetime.now(timezone.utc)


#: Matches `_MAX_FAILURE_REASON` in `navis.signals.sync`, and for the same
#: reason: the column is VARCHAR(500), so truncating at 300 keeps the cut ours
#: and visible rather than letting the database raise on an over-long string.
_MAX_FAILURE_REASON = 300


def _bounded_reason(outcome: Any) -> str:
    """A short, storable description of a NON-auth sync failure.

    Deliberately coarse. The exception never reaches this frame — the sync engine
    dead-letters it and returns a flat outcome — so this reports the shape of the
    failure and points at the DLQ, which holds the stack. Writing something more
    specific here would mean re-plumbing exceptions through a boundary that
    exists to stop exactly that.
    """
    if outcome.dead_lettered:
        return f"sync failed; {outcome.dead_lettered} event(s) dead-lettered"[:_MAX_FAILURE_REASON]
    return "sync failed; see dead-letter queue"[:_MAX_FAILURE_REASON]


def _build_client(integration: Integration):
    entry = _REGISTRY.get(integration.provider)
    if entry is None:
        return None, None, None
    connector, client_factory = entry
    token_manager = TokenManager(integration)
    transport = GoogleTransport(
        token_manager.token_provider(), on_auth_failure=token_manager.invalidate
    )
    client = client_factory(transport)
    return connector, client, (transport, token_manager)


async def sync_integration(integration: Integration) -> dict:
    connector, client, handles = _build_client(integration)
    if connector is None or handles is None:
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
                if fresh.user_id is None:  # checked on the freshly-read row, not the stale arg
                    logger.error("integration %s has no user_id; cannot emit tenant-safe signals", fresh.id)
                    return {"error": "missing user_id"}

                # ── ENTITLEMENT GATE (T-14) ──────────────────────────────────
                # Asked HERE rather than in the beat task because all three sync
                # producers converge on this function — the scheduled fan-out, the
                # Pub/Sub push, and the manual /integrations/google/sync route — so
                # one check covers every producer instead of three that can drift.
                # Both inputs come from the FRESHLY RE-READ row inside the advisory
                # lock, so neither the tenant nor the connector is taken from the
                # caller's stale argument.
                capability = _capability_for(fresh.provider)
                if capability is None:
                    logger.error(
                        "no capability mapped for provider %s; refusing to sync integration %s",
                        fresh.provider, fresh.id,
                    )
                    return {"skipped": "no_capability_mapping"}
                if not await scheduled_work_allowed(
                    db, fresh.organization_id, capability, task="sync_integration"
                ):
                    return {"skipped": "not_entitled"}

                # ── AUTH HEALTH GATE ─────────────────────────────────────────
                # A grant that needs re-consent will be refused identically on
                # every future attempt, so attempting it is pure waste: it burns
                # a worker, a provider rate-limit slot and a DLQ row to learn
                # what the row already records. Checked on the FRESHLY RE-READ
                # row inside the lock, so a re-consent that landed moments ago is
                # honoured rather than raced.
                #
                # AFTER THE ENTITLEMENT GATE, deliberately. Entitlement is the
                # more fundamental refusal — a tenant whose plan does not cover
                # Gmail should be told that, not told its token needs
                # re-authorising, and a person who re-consents in response to the
                # wrong diagnosis has wasted a trip through an OAuth flow that
                # was never going to help.
                #
                # `_dispatch_all` already excludes these from the scheduled
                # fan-out. This is the second gate, and it is the one that covers
                # the other two producers: the Pub/Sub push and the manual sync
                # route do not go through the fan-out, so without it a dead
                # account would still be retried on every notification.
                #
                # RECOVERY IS NOT BLOCKED BY THIS. Nothing here writes
                # `auth_state`, and a new OAuth grant clears it — so the user
                # re-consents, the callback sets `ok`, and the very next pass
                # runs normally with no backlog to drain.
                if fresh.auth_state != IntegrationAuthState.OK.value:
                    metrics.incr(
                        "sync.skipped.auth",
                        labels={"provider": fresh.provider.value, "auth_state": fresh.auth_state},
                    )
                    return {"skipped": "auth_" + fresh.auth_state}

                # Tenant log context, bound from the FRESHLY RE-READ row — never
                # from the caller's argument and never from a task payload (T-21).
                # Everything below, including the connector's own logging, now
                # carries org_id as a structured field.
                with org_context(fresh.organization_id):
                    ctx = SignalContext(
                        org_id=fresh.organization_id, user_id=fresh.user_id, integration_id=fresh.id
                    )
                    cursor = (fresh.sync_state or {}).get("cursor") or {}

                    outcome = await SignalSyncService(db).sync(ctx, connector, client, cursor)

                    # ── HEALTH, RECORDED ON EVERY PATH ───────────────────────
                    # In the SAME transaction as the cursor, and — unlike the
                    # cursor — written whether the pass succeeded or not. That
                    # asymmetry is the whole point: a failure moves no cursor, so
                    # a health field kept beside the cursor would only ever
                    # record successes and could never say a source is broken.
                    #
                    # `last_attempt_at` FIRST and unconditionally: it is the
                    # fact that separates "healthy and quiet" from "the
                    # scheduler stopped visiting this row".
                    now = _utcnow()
                    fresh.last_attempt_at = now

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
                            # Google rotated the refresh token -> persist the NEW one, else the
                            # next refresh uses a dead token and the integration silently breaks.
                            if token_manager.last_token.refresh_token:
                                fresh.refresh_token = token_cipher.encrypt(
                                    token_manager.last_token.refresh_token
                                )
                        # A PASS THAT REACHED THE PROVIDER IS PROOF THE GRANT
                        # WORKS, so it clears the failure record.
                        #
                        # NOTE WHAT THIS DOES *NOT* DO. It cannot rescue a row
                        # already marked `reconsent_required`, because the gate
                        # above refuses that row before the sync runs — so this
                        # branch is only ever reached by a row that was already
                        # `ok`. That is deliberate and it is the whole retry-storm
                        # design: `reconsent_required` is TERMINAL until a new
                        # OAuth grant, and the callback in `signals/router.py` is
                        # the single path back. Writing `auth_state` here anyway
                        # keeps the invariant "a successful pass means ok" true at
                        # the row level rather than relying on it being true by
                        # construction, which is the kind of assumption that stops
                        # holding when somebody later adds a way in.
                        fresh.last_success_at = now
                        fresh.auth_state = IntegrationAuthState.OK.value
                        fresh.last_failure_reason = None
                    elif outcome.auth_failed:
                        # THE CREDENTIAL IS DEAD, and this is the branch that
                        # used to not exist — the exception was dead-lettered and
                        # the row left claiming to be connected.
                        #
                        # `is_active` IS NOT TOUCHED. The user still wants Gmail
                        # connected; Google just rejected the refresh token.
                        # Flipping intent here is what made "disconnected by the
                        # user" and "needs re-authorising" indistinguishable, and
                        # it would also hide the row from the reconnect UI that
                        # is supposed to fix it.
                        #
                        # NEITHER THE REFRESH TOKEN NOR THE CURSOR IS DELETED.
                        # The token may be revived by re-consent, and the cursor
                        # is where ingestion must resume — discarding it would
                        # turn every re-consent into a full historical resync.
                        fresh.auth_state = IntegrationAuthState.RECONSENT_REQUIRED.value
                        fresh.last_failure_reason = outcome.auth_failure_reason
                        logger.warning(
                            "integration %s (%s) needs re-consent: %s",
                            fresh.id, fresh.provider.value, outcome.auth_failure_reason,
                        )
                        metrics.incr(
                            "sync.auth_failed", labels={"provider": fresh.provider.value}
                        )
                    else:
                        # A TRANSIENT FAILURE — a timeout, a 500, a connector
                        # bug. `auth_state` is deliberately LEFT ALONE: the
                        # credential was never shown to be bad, and demoting a
                        # working grant on a network blip would stop syncing an
                        # account that is fine. The reason is still recorded, so
                        # an operator can see that attempts are failing even
                        # while the grant itself is healthy.
                        fresh.last_failure_reason = _bounded_reason(outcome)
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
    if connector is None or handles is None or not integration.user_id:
        return {"skipped": True}

    # A watch is a standing instruction to Google to push into this deployment on
    # the org's behalf, so it needs the same entitlement the ingestion does — and
    # renewing one for a suspended tenant would keep that push alive indefinitely.
    capability = _capability_for(integration.provider)
    async with session_scope() as db:
        if capability is None or not await scheduled_work_allowed(
            db, integration.organization_id, capability, task="renew_watch"
        ):
            return {"skipped": True, "reason": "not_entitled"}

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
    """Fan-out: enqueue ONE per-integration task per HEALTHY active integration.

    THE RETRY-STORM GATE. `auth_state != 'ok'` is excluded here rather than left
    for `sync_integration` to refuse, because the cost being avoided is the task
    itself: a dead grant that reaches the worker has already taken a queue slot,
    a database session and an advisory lock to learn what this predicate knows
    for free. Six Google rows on one beat is nothing; six hundred across a
    tenant base, every beat, forever, is the storm.

    Excluding a row from the fan-out does NOT strand it. `auth_state` is cleared
    by the OAuth callback the moment the user re-consents, so the row rejoins
    the very next beat — no backlog to drain, no operator action, no separate
    retry schedule to get wrong.
    """
    dispatched, after, page = 0, None, 1000
    while True:
        async with session_scope() as db:
            stmt = (
                select(Integration.id)
                .where(
                    Integration.is_active.is_(True),
                    Integration.auth_state == IntegrationAuthState.OK.value,
                )
                .order_by(Integration.id)
                .limit(page)
            )
            if after is not None:
                stmt = stmt.where(Integration.id > after)
            ids = (await db.scalars(stmt)).all()
        if not ids:
            break
        for iid in ids:
            celery_app.send_task(
                "navis.orchestration.sync.sync_one_integration", args=[str(iid)]
            )
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
    """Webhook-triggered: sync the specific mailbox the push notification named.

    THE MAILBOX ADDRESS IS A SELECTOR, NOT A CREDENTIAL. The caller has already
    been authenticated as Google (signals/router.py verifies the push JWT before
    this task is ever enqueued); `email` only says WHICH connected mailbox the
    notification concerns. Organization scope is never taken from it — it is read
    from the resolved `Integration` row, and `sync_integration` re-reads that row
    inside its own transaction before building the `SignalContext`.

    Both guards below fail CLOSED, because the alternative in each case is doing
    real work against an arbitrarily-chosen tenant.
    """
    if not email:
        # `if email:` used to be optional here, so a push whose payload carried no
        # `emailAddress` dropped the filter entirely and `db.scalar` returned SOME
        # active integration — a full sync run against a tenant picked by whatever
        # order the planner happened to produce. Without the address there is no
        # mailbox to sync, so there is nothing to do.
        logger.warning("pubsub sync: no mailbox address in the notification; refusing")
        metrics.incr("signals.pubsub.unresolved", labels={"reason": "no_address"})
        return {"matched": False, "reason": "no_address"}

    async with session_scope() as db:
        matches = (
            await db.scalars(
                select(Integration).where(
                    Integration.provider == IntegrationProvider(provider),
                    Integration.is_active.is_(True),
                    Integration.external_account == email,
                )
            )
        ).all()

    if not matches:
        logger.warning("pubsub sync: no integration for %s/%s", provider, email)
        metrics.incr("signals.pubsub.unresolved", labels={"reason": "no_match"})
        return {"matched": False, "reason": "no_match"}
    if len(matches) > 1:
        # Two organizations have connected the SAME mailbox. `uq_org_provider_account`
        # permits this (it is unique per org, not globally), and a notification names
        # only the address, so nothing in the message says which tenant it is for.
        #
        # Deliberately NOT resolved by picking newest/lowest-id: that would silently
        # attribute one organization's mail to whichever row won, and the tie-break
        # is a product decision about what sharing a mailbox even means. Refusing is
        # the honest answer and the scheduled poll still syncs BOTH tenants, so no
        # mail is lost — only the push acceleration is.
        logger.error(
            "pubsub sync: %s/%s maps to %d active integrations across %d organization(s); "
            "refusing to guess which tenant this notification belongs to",
            provider, email, len(matches), len({m.organization_id for m in matches}),
        )
        metrics.incr("signals.pubsub.unresolved", labels={"reason": "ambiguous_tenant"})
        return {"matched": False, "reason": "ambiguous_tenant"}

    return await sync_integration(matches[0])


def _slack_install_narrower(envelope: dict):
    """Disambiguate a shared workspace by the INSTALL the event was delivered to.

    Slack's Events API envelope names the authorized bot alongside the team:
    `authorizations[].user_id` is the bot user id of the install Slack delivered
    to, and `api_app_id` is the app. Both are recorded at install time by
    `slack.oauth.install_fields` into `sync_state['slack']` — so when two
    organizations have genuinely distinct installs in one workspace, the event
    itself says which one it belongs to and we do not have to refuse.

    This is a NARROWER, never a chooser: it returns the subset whose persisted
    install identity matches, and `resolve_owning_integration` re-asserts
    uniqueness on whatever comes back. If the envelope carries no bot identity, or
    the identity matches several tenants, the subset stays ambiguous and the
    caller still fails closed.

    The envelope is authenticated (the HMAC was verified before this task was
    enqueued), so these fields are Slack's assertion, not a caller's. They still
    only ever SELECT among integrations we already persisted — they can never
    introduce an organization.
    """
    authorizations = envelope.get("authorizations") or []
    bot_user_ids = {
        a.get("user_id") for a in authorizations if isinstance(a, dict) and a.get("user_id")
    }
    app_id = envelope.get("api_app_id")

    def _narrow(rows):
        if not bot_user_ids and not app_id:
            return []  # nothing to narrow WITH -> leave the ambiguity intact
        matched = []
        for r in rows:
            state = (r.sync_state or {}).get("slack") or {}
            if bot_user_ids and state.get("bot_user_id") in bot_user_ids:
                matched.append(r)
            elif not bot_user_ids and app_id and state.get("app_id") == app_id:
                matched.append(r)
        return matched

    return _narrow


async def _process_slack_event(team_id: str | None, envelope: dict) -> dict:
    """Webhook-triggered: resolve the Slack install, map the event, ingest. Runs out of band
    (the webhook already ACKed). Reuses the connector ingestion path (batch+fallback+DLQ).

    TENANT RESOLUTION IS THE FIRST THING THAT HAPPENS AND IT FAILS CLOSED.

    This used to be `db.scalar(...)`, which returns the FIRST row of an unordered
    result set. `uq_org_provider_account` is unique per organization, so two
    organizations connecting the same workspace both match — and the planner, not
    the data, decided which tenant the messages were ingested as. Reproduced
    before this change: two rows, one `db.scalar`, `SignalContext.org_id` came
    back as whichever row was first, and the event was captured, mapped and
    ingested under it.

    Mirrors `_sync_account` (the Gmail push path), which already refused this way.
    """
    from navis.identity.org_service import AmbiguityReason, resolve_owning_integration
    from navis.signals.base import SignalSource
    from navis.signals.slack.events import events_from_envelope

    if not team_id:
        metrics.incr("slack.event.unresolved", labels={"reason": AmbiguityReason.NO_ACCOUNT})
        return {"matched": False, "reason": AmbiguityReason.NO_ACCOUNT}
    inner_type = (envelope.get("event") or {}).get("type") or envelope.get("type")
    async with session_scope() as db:
        integration, reason = await resolve_owning_integration(
            db,
            provider=IntegrationProvider.SLACK,
            external_account=team_id,
            narrow=_slack_install_narrower(envelope),
        )
        if reason == AmbiguityReason.AMBIGUOUS_TENANT:
            # Two organizations have connected the SAME workspace and the envelope
            # carried no install identity that separates them. Refusing is the
            # honest answer: nothing in the message says which tenant it is for,
            # and attributing it to either one writes another company's
            # conversations into their memory, state and brief — irreversibly,
            # because Memory is append-only.
            logger.error(
                "slack event: team=%s maps to multiple organizations and the envelope "
                "carries no install identity; refusing to guess which tenant this is for",
                team_id,
            )
            metrics.incr("slack.event.unresolved", labels={"reason": reason})
            return {"matched": False, "reason": reason}
        if integration is None or not integration.user_id:
            logger.warning("slack event: no active integration for team=%s", team_id)
            metrics.incr(
                "slack.event.unresolved",
                labels={"reason": reason or AmbiguityReason.NO_MATCH},
            )
            return {"matched": False, "reason": reason or AmbiguityReason.NO_MATCH}

        # M-1: workspace disconnect / token revocation. BOTH stop us hitting
        # Slack with a dead token — that failure rate is what trips Slack's 95%
        # auto-disable — but they are NOT the same event and no longer share a
        # response.
        #
        #   app_uninstalled  the app was REMOVED FROM THE WORKSPACE. That is a
        #                    person deciding to disconnect, so it moves intent:
        #                    is_active=False, exactly as before. The token is
        #                    dead too, so auth_state follows.
        #
        #   tokens_revoked   the CREDENTIAL was revoked. Nobody asked to
        #                    disconnect — an admin rotated a token, or Slack
        #                    invalidated it. Intent is untouched; only auth_state
        #                    moves.
        #
        # Collapsing the second into the first is the overload this phase
        # removes. It made an auth failure indistinguishable from a user
        # disconnect, and it hid the row from the reconnect UI that exists to fix
        # it: `integrations_for_external_account` filters on `is_active`, so a
        # deactivated row stops resolving and the workspace goes quiet with no
        # explanation. Leaving `is_active` True keeps the row visible and keeps
        # INBOUND events flowing — a webhook envelope carries its own payload and
        # needs no token to ingest — while `auth_state` stops every OUTBOUND call
        # and keeps the row out of the sync fan-out.
        if inner_type in ("app_uninstalled", "tokens_revoked"):
            integration.auth_state = IntegrationAuthState.REVOKED.value
            integration.last_failure_reason = f"slack {inner_type}"
            integration.last_attempt_at = _utcnow()
            disconnected = inner_type == "app_uninstalled"
            if disconnected:
                integration.is_active = False
            logger.warning(
                "slack %s for team=%s; auth_state=revoked is_active=%s",
                inner_type, team_id, integration.is_active,
            )
            return {"matched": True, "deactivated": disconnected, "auth_revoked": True}

        # Tenant log context from the RESOLVED integration row (T-21) — bound only
        # after the resolution above proved exactly one organization owns this
        # workspace, so an ambiguous event never stamps a tenant onto its logs.
        with org_context(integration.organization_id):
            ctx = SignalContext(integration.organization_id, integration.user_id, integration.id)
            slack_state = (integration.sync_state or {}).get("slack") or {}
            events = events_from_envelope(
                ctx, envelope,
                our_bot_id=slack_state.get("bot_id"), our_bot_user_id=slack_state.get("bot_user_id"),
            )
            if not events:
                return {"matched": True, "captured": 0}  # unmapped verb — captured nothing, by design
            outcome = await SignalSyncService(db).ingest_events(ctx, SignalSource.SLACK, events)
    return {"matched": True, "captured": outcome.ingested, "duplicates": outcome.duplicates,
            "dead_lettered": outcome.dead_lettered}


@celery_app.task(name="navis.orchestration.sync.process_slack_event")
def process_slack_event(team_id: str | None = None, envelope: dict | None = None) -> dict:
    return run_async(_process_slack_event(team_id, envelope or {}))


async def _run_retention_purge() -> dict:
    from navis.signals.governance import run_retention_purge

    async with session_scope() as db:
        return await run_retention_purge(db)


@celery_app.task(name="navis.orchestration.sync.retention_purge")
def retention_purge() -> dict:
    """Periodic content-retention enforcement (M-1) — tombstone expired content, keep skeleton."""
    return run_async(_run_retention_purge())


@celery_app.task(name="navis.orchestration.sync.sync_one_integration")
def sync_one_integration(integration_id: str) -> dict:
    return run_async(_sync_by_id(uuid.UUID(integration_id)))


@celery_app.task(name="navis.orchestration.sync.sync_all_sources")
def sync_all_sources() -> dict:
    return {"dispatched": run_async(_dispatch_all())}


@celery_app.task(name="navis.orchestration.sync.renew_watches")
def renew_watches() -> dict:
    return run_async(_renew_all_watches())


@celery_app.task(name="navis.orchestration.sync.sync_google_account")
def sync_google_account(provider: str, email: str | None = None, history_id: str | None = None) -> dict:
    return run_async(_sync_account(provider, email, history_id))
