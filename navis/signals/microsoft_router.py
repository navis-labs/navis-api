"""Microsoft 365 HTTP surface.

  GET    /integrations/microsoft/connect      start OAuth (authenticated)
  GET    /integrations/microsoft/callback     OAuth redirect target -> upserts Integration
  DELETE /integrations/microsoft/disconnect   soft-disconnect the whole suite
  POST   /integrations/microsoft/sync         manually drive one sync pass

A sibling module rather than four more handlers in `signals/router.py`, which is
already past a thousand lines and owns Google, Slack and the Pub/Sub webhook.
Nothing here imports from the Google connector packages and nothing there
imports from here — the two suites share the OAuth state store, the sync engine
and the ingestion service, and share nothing above them.

OAuth state is cryptographically random, stored server-side in Redis (10-min
TTL, single-use), and the callback resolves org/user ONLY from that store —
never from the callback query string (CSRF + authorization-code binding per the
OAuth spec). That primitive lives in `navis.signals.oauth_state` and is the
exact code the Google flow uses; see that module for why it is shared rather
than reimplemented.
"""
from __future__ import annotations

import base64
import binascii
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity.models import Integration, IntegrationProvider
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.config import settings
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.crypto import token_cipher
from navis.signals.microsoft import oauth as ms_oauth
from navis.signals.oauth_state import consume_oauth_state, create_oauth_state

logger = get_logger("signals.microsoft_router")
router = APIRouter(tags=["signals"])

# ── Scopes ─────────────────────────────────────────────────────────────────
#
# The MINIMUM the grant must carry for the integration to be worth activating
# at all. Mail is the floor: an org that grants only calendar access gets a
# product that cannot see the thing every other surface reasons about, and
# activating it would present as "connected" while doing almost nothing.
_REQUIRED_SCOPES: tuple[str, ...] = ("mail.read",)

# Which granted scope activates which per-resource Integration row. A resource
# whose scope was withheld gets NO ROW — not an inactive one, not a placeholder.
# The frontend renders "Not connected" for a missing row, which is the honest
# answer; a row would render a green pill over something nobody authorised.
#
# Mirrors `_GOOGLE_FAMILY_ACTIVATION_SCOPES` in signals/router.py, including
# the tuple-of-alternatives shape, so a future scope migration (the Google
# calendar.readonly -> calendar.events one is the precedent) has somewhere to
# put the older spelling without breaking orgs connected under it.
_MICROSOFT_ACTIVATION_SCOPES: tuple[tuple[IntegrationProvider, tuple[str, ...]], ...] = (
    (IntegrationProvider.OUTLOOK, ("mail.read",)),
    (
        IntegrationProvider.OUTLOOK_CALENDAR,
        ("calendars.readwrite", "calendars.read"),
    ),
    (IntegrationProvider.ONEDRIVE, ("files.read.all", "files.read")),
    # TEAMS intentionally absent — no connector, no scope requested. See
    # `_UNSERVED_CAPABILITIES` in organization/contract.py.
)

# Every provider the suite owns. Used by disconnect and sync so adding a
# connector later means adding it in ONE place.
_MICROSOFT_FAMILY: tuple[IntegrationProvider, ...] = (
    IntegrationProvider.OUTLOOK,
    IntegrationProvider.OUTLOOK_CALENDAR,
    IntegrationProvider.ONEDRIVE,
    IntegrationProvider.TEAMS,
)

# The connector id the frontend uses to identify this suite in redirect params.
# Not `microsoft_365`: the frontend resolves a suite from a CONNECTOR id via
# `providerOfConnector`, and `outlook` is the member that always exists when
# anything in the suite does.
_FRONTEND_PROVIDER = "outlook"


def _configured() -> bool:
    """Every value the connect flow needs before it can build a real URL.

    The redirect URI belongs in this check, not just the credentials:
    `authorization_url()` falls back to an empty string when it is unset, which
    produces precisely the half-built consent URL the 503 below exists to
    prevent — Entra answers that with an error the founder cannot act on.
    """
    return bool(
        settings.microsoft_client_id
        and settings.microsoft_client_secret
        and settings.microsoft_redirect_uri
    )


def _frontend_return_url(*, ok: bool, reason: str | None = None) -> str:
    """Where the browser lands after the OAuth code exchange finishes.

    Success -> `${frontend_url}/dashboard/account?connected=outlook`
    Failure -> `${frontend_url}/dashboard/account?error=<code>&provider=outlook`

    The `provider=` parameter on the FAILURE path is not decoration and is the
    one way this differs from the Google version. `decodeOAuthError` on the
    frontend writes the suite name into every error string, and defaults to
    "Google" when the parameter is absent — so without it a failed Outlook
    connect tells a Microsoft org to "Reconnect Google", which is not a thing
    they have. Google's own callback keeps sending nothing and keeps getting
    the default, correctly.

    Deliberately NOT shared with `signals/router.py::_frontend_return_url`:
    sharing it would mean editing a live Google redirect path to add a
    parameter only this flow needs, for no behavioural gain there.

    The destination is a server-side setting and never anything the client
    sent, so the callback cannot be talked into redirecting somewhere else.
    """
    base = settings.frontend_url.rstrip("/")
    if ok:
        return f"{base}/dashboard/account?connected={_FRONTEND_PROVIDER}"
    return (
        f"{base}/dashboard/account?error={quote(reason or 'oauth_failed')}"
        f"&provider={_FRONTEND_PROVIDER}"
    )


def _redirect_error(reason: str) -> RedirectResponse:
    return RedirectResponse(
        _frontend_return_url(ok=False, reason=reason),
        status_code=status.HTTP_303_SEE_OTHER,
    )


def _id_token_claims(id_token: str | None) -> dict[str, Any]:
    """Read the claims out of Entra's id_token.

    The token came directly from Entra's token endpoint over TLS in the
    authorization-code flow, so its contents are trusted without re-verifying
    the signature — the same reasoning (and the same implementation shape) as
    `_email_from_id_token` on the Google side.

    Note this is the CONNECTOR flow. Sign-in has a stricter job: it decides
    whether to LINK this identity to an existing Navis account, so it must also
    check that Entra asserts the email is verified. Here the org is already
    resolved from the server-side OAuth state, so the email is only a label for
    which mailbox was connected and cannot be used to reach another tenant.
    """
    if not id_token:
        return {}
    try:
        payload_b64 = id_token.split(".")[1]
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
        return claims if isinstance(claims, dict) else {}
    except (IndexError, binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _account_email(claims: dict[str, Any]) -> str | None:
    """Identify the connected mailbox from the id_token claims.

    `email` first. `preferred_username` second, but ONLY when it looks like an
    address: in a work/school tenant it is the UPN, which is usually an email
    and is sometimes not one at all, and a non-address value here would become
    the external_account half of the uniqueness constraint and quietly split
    one mailbox across two Integration rows on reconnect.
    """
    email = claims.get("email")
    if isinstance(email, str) and "@" in email:
        return email.lower()
    upn = claims.get("preferred_username")
    if isinstance(upn, str) and "@" in upn:
        return upn.lower()
    return None


# ── Connect ────────────────────────────────────────────────────────────────


@router.get(
    "/integrations/microsoft/connect",
    dependencies=[Depends(require_capability(Capability.OUTLOOK))],
)
async def microsoft_connect(
    ctx: CurrentContext = Depends(get_current_context),
) -> dict[str, str]:
    if not _configured():
        # A 503 with a specific message rather than a half-built authorization
        # URL that bounces the founder off login.microsoftonline.com with an
        # opaque Entra error they cannot act on.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Microsoft 365 is not configured on this deployment.",
        )
    state = await create_oauth_state(
        org_id=ctx.org_id,
        user_id=ctx.user.id,
        provider=IntegrationProvider.OUTLOOK.value,
    )
    return {"authorization_url": ms_oauth.authorization_url(state)}


# ── Callback ───────────────────────────────────────────────────────────────


@router.get("/integrations/microsoft/callback")
async def microsoft_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: AsyncSession = Depends(get_session),
) -> RedirectResponse:
    # 0) The user declined, or Entra refused before issuing a code. Entra sends
    # `error` with no `code`; treating that as a generic failure would tell
    # somebody who deliberately cancelled that something went wrong.
    if error or not code:
        logger.info("microsoft oauth returned without a code error=%s", error)
        # The flow is over either way, so burn the state now rather than
        # leaving a live handle in Redis for the rest of its TTL.
        if state:
            try:
                await consume_oauth_state(state)
            except HTTPException:
                pass  # already used/expired — nothing to clean up.
        # `access_denied` is the user cancelling on the consent screen. That is
        # a decision, not a fault, so send them back to a clean Connections
        # page with NO error param and let the unchanged card state speak for
        # itself. Inventing a `cancelled` code would be worse than the generic
        # one: `decodeOAuthError` matches a closed set (missing_scopes,
        # no_email_claim, exchange_failed, invalid_state, oauth_failed) and
        # renders anything else as "Navis received an unknown error code".
        if error == "access_denied":
            base = settings.frontend_url.rstrip("/")
            return RedirectResponse(
                f"{base}/dashboard/account",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return _redirect_error("oauth_failed")

    # 1) Validate state FIRST and resolve attribution ONLY from the server-side
    # store. State errors bounce the browser back with an error code — never a
    # raw 400 page, or the founder is stuck on a JSON blob with no way home.
    try:
        bound = await consume_oauth_state(state)
    except HTTPException:
        return _redirect_error("invalid_state")
    org_id, user_id = uuid.UUID(bound["org_id"]), uuid.UUID(bound["user_id"])

    # 2) Exchange the code (bound to this validated state).
    try:
        tokens = await ms_oauth.exchange_code(code)
    except Exception as exc:  # noqa: BLE001 — surface any exchange failure to the UI
        logger.warning(
            "microsoft oauth code exchange failed org=%s user=%s err=%r",
            org_id, user_id, exc,
        )
        metrics.incr("signals.oauth.exchange_failed", labels={"provider": "microsoft"})
        return _redirect_error("exchange_failed")

    # 3) Reject incomplete grants — never activate an integration missing the
    # scopes that make it useful.
    granted_raw = tokens.get("scope")
    granted = ms_oauth.normalize_scopes(granted_raw)
    missing = [s for s in _REQUIRED_SCOPES if s not in granted]
    if missing:
        metrics.incr("signals.oauth.scope_rejected", labels={"provider": "microsoft"})
        logger.warning(
            "microsoft oauth scope rejected org=%s user=%s missing=%s granted=%s",
            org_id, user_id, missing, sorted(granted),
        )
        return _redirect_error("missing_scopes")

    # 4) Identify the connected account (the dedup key) from the trusted id_token.
    claims = _id_token_claims(tokens.get("id_token"))
    email = _account_email(claims)
    if not email:
        logger.warning(
            "microsoft oauth callback missing account email org=%s user=%s", org_id, user_id
        )
        return _redirect_error("no_email_claim")

    # 5) UPSERT one row PER Microsoft-family provider whose scope was granted.
    #
    # Uniqueness enforced at the DB on (org, provider, external_account) — a
    # reconnect (same account) UPDATES each row: no duplicate integrations, no
    # duplicate sync streams, no duplicate signals. Race-safe (ON CONFLICT, not
    # read-then-insert). Sharing tokens across rows is correct because a single
    # Entra grant covers every scope in the request; each provider row just
    # makes its service independently addressable to the sync orchestrator.
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=int(tokens.get("expires_in", 3600))
    )
    enc_access = token_cipher.encrypt(tokens.get("access_token"))
    enc_refresh = token_cipher.encrypt(tokens.get("refresh_token"))

    if not tokens.get("refresh_token"):
        # `offline_access` was requested, so this should not happen — and if it
        # does, the integration works for one hour and then dies with a
        # confusing "reconsent required". Better to refuse now with the error
        # the founder can act on.
        logger.error(
            "microsoft oauth returned no refresh_token org=%s user=%s", org_id, user_id
        )
        return _redirect_error("missing_scopes")

    activated: list[str] = []
    for provider, activation_scopes in _MICROSOFT_ACTIVATION_SCOPES:
        if not any(scope in granted for scope in activation_scopes):
            continue  # user didn't grant this service — skip; no fake row.
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
                scopes=granted_raw,
                is_active=True,
                sync_state={},
            )
            .on_conflict_do_update(
                constraint="uq_org_provider_account",
                set_={
                    "user_id": user_id,
                    "access_token": enc_access,
                    # Unlike Google, Entra ALWAYS returns a refresh token here
                    # and rotates it — so this overwrites unconditionally
                    # rather than coalescing onto the stored value. Coalescing
                    # (correct for Google) would keep a token Entra has already
                    # invalidated, and the integration would die silently at
                    # the 90-day inactivity mark.
                    "refresh_token": enc_refresh,
                    "token_expires_at": expires_at,
                    "scopes": granted_raw,
                    "is_active": True,
                    "updated_at": func.now(),
                },
            )
            .returning(Integration.id)
        )
        await db.execute(stmt)
        activated.append(provider.value)
        metrics.incr(
            "signals.integration.connected", labels={"provider": provider.value}
        )

    await db.commit()
    logger.info(
        "microsoft integrations upserted org=%s user=%s account=%s providers=%s",
        org_id, user_id, email, activated,
    )
    return RedirectResponse(
        _frontend_return_url(ok=True),
        status_code=status.HTTP_303_SEE_OTHER,
    )


# ── Disconnect ─────────────────────────────────────────────────────────────


@router.delete("/integrations/microsoft/disconnect")
async def microsoft_disconnect(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Soft-disconnect every Microsoft-family integration for this org.

    Mirrors the Google and Slack disconnect contracts:
      * flips is_active=False on every Microsoft-family row so the sync
        engine's is_active filter stops touching the account without a
        redeploy;
      * clears access_token / refresh_token bytes so a later leak or dump
        doesn't expose credentials for a connection the user no longer wants;
      * clears sync_state cursors so a reconnect starts a clean window instead
        of resuming a delta link the user thought they wiped;
      * publishes INTEGRATION_DISCONNECTED per row so onboarding / capability
        gates recompute.

    The Integration row itself is NOT deleted — historical SignalEvents
    reference it by FK, and a reconnect UPSERTs into the same row to avoid
    duplicate history and dangling references.

    ── Why there is no token revocation call here ──

    Google's disconnect posts the refresh token to `/revoke`, which kills the
    whole grant at Google. Entra has no equivalent endpoint that a confidential
    client can call for a user's grant: revocation is either the user removing
    the app at myapps.microsoft.com or an admin revoking it in the portal. So
    the honest thing is to make the local disconnect complete and immediate and
    tell the user where to finish the job — which the frontend's disconnect
    dialog already does, linking to `revokeUrl` from its provider registry.
    Silently doing nothing while implying a full revoke would be worse.
    """
    rows = (
        await db.scalars(
            select(Integration).where(
                Integration.organization_id == ctx.org_id,
                Integration.provider.in_(_MICROSOFT_FAMILY),
                Integration.is_active.is_(True),
            )
        )
    ).all()

    if not rows:
        return {"disconnected": True, "count": 0, "revoked": 0}

    stmt = (
        update(Integration)
        .where(
            Integration.organization_id == ctx.org_id,
            Integration.provider.in_(_MICROSOFT_FAMILY),
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

    if deactivated:
        from navis.platform.events import DomainEvent, EventType, event_bus

        for integration_id, provider in deactivated:
            await event_bus.publish(
                DomainEvent(
                    type=EventType.INTEGRATION_DISCONNECTED,
                    org_id=ctx.org_id,
                    source="signals.microsoft_router",
                    payload={
                        "integration_id": str(integration_id),
                        "provider": provider.value,
                    },
                )
            )
        metrics.incr(
            "signals.integration.disconnected",
            labels={"provider": "microsoft"},
            value=len(deactivated),
        )
        logger.info(
            "microsoft integrations disconnected org=%s count=%d",
            ctx.org_id, len(deactivated),
        )

    return {
        "disconnected": True,
        "count": len(deactivated),
        # Always 0 — see the docstring. Reported rather than omitted so the
        # response shape matches the Google one the frontend already handles.
        "revoked": 0,
    }


# ── Manual sync ────────────────────────────────────────────────────────────


@router.post(
    "/integrations/microsoft/sync",
    dependencies=[Depends(require_capability(Capability.OUTLOOK))],
)
async def sync_microsoft_integration(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Manually drive one sync pass over the org's active Microsoft integrations.

    Normally sync runs out-of-band via Celery beat. That is not running in local
    dev unless the operator has provisioned a broker — so without this endpoint
    a freshly connected Outlook demo has no way to ingest anything and the whole
    pipeline stays cold. Same rationale, and the same orchestrator, as
    `/integrations/google/sync`.

    Safe to call repeatedly: each integration takes an advisory lock inside
    `sync_integration`, so concurrent presses are single-flighted rather than
    racing.
    """
    from navis.orchestration.sync import sync_integration

    rows = (
        await db.scalars(
            select(Integration).where(
                Integration.organization_id == ctx.org_id,
                Integration.is_active.is_(True),
                Integration.provider.in_(_MICROSOFT_FAMILY),
            )
        )
    ).all()

    if not rows:
        return {"synced": 0, "results": []}

    results = [await sync_integration(row) for row in rows]
    return {"synced": len(results), "results": results}
