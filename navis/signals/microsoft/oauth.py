"""Microsoft (Entra ID) OAuth token lifecycle — shared by all Graph connectors.

Responsibilities mirror `navis.signals.google.oauth`:
  * build the consent URL and exchange the auth code (onboarding);
  * refresh an access token from a refresh token, just-in-time, before expiry.

The TokenManager does NOT write the database — it returns refreshed tokens and a
flag so the *caller* (sync engine / orchestration) persists them on the identity
Integration. A `refresher` is injectable for tests.

── The one place this is NOT a copy of the Google module ──────────────────

Microsoft ROTATES refresh tokens. Every successful refresh returns a new
refresh_token and invalidates the one that was used; Google usually returns
none and expects you to keep the stored one. The consequence is asymmetric and
nasty: a caller that treats a missing rotated token as "unchanged" is correct
for Google and silently fatal for Microsoft, because the stored token is dead
the moment it is used once. The failure does not surface at refresh time — it
surfaces up to 90 days later when the inactivity window closes and every
integration for the org stops at once.

So `MicrosoftToken.refresh_token` is populated on EVERY refresh, not only when
it changed, and `sync_integration` persists it unconditionally. If a future
change makes that field optional again, this comment is the reason not to.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable
from urllib.parse import urlencode

from navis.platform.config import settings
from navis.platform.monitoring import get_logger

logger = get_logger("signals.microsoft.oauth")

_AUTHORITY = "https://login.microsoftonline.com"
_GRAPH_RESOURCE = "https://graph.microsoft.com/"


def _authorize_url() -> str:
    return f"{_AUTHORITY}/{settings.microsoft_tenant}/oauth2/v2.0/authorize"


def _token_url() -> str:
    return f"{_AUTHORITY}/{settings.microsoft_tenant}/oauth2/v2.0/token"


# ── Scopes ─────────────────────────────────────────────────────────────────
#
# TWO sets, and they must never merge.
#
# MICROSOFT_SIGNIN_SCOPES is what /auth/microsoft requests: enough to learn who
# somebody is, and nothing more. A founder signing in has not agreed to let us
# read their mail, and asking at the login screen is the worst possible moment
# to ask — it converts a sign-up into a security review.
#
# MICROSOFT_CONNECTOR_SCOPES is what /integrations/microsoft/connect requests,
# once the founder is inside the product and has seen what Navis does with a
# mailbox. `offline_access` is what yields a refresh token at all; without it
# the integration works for exactly one hour and then dies.
#
# Teams scopes are deliberately absent. `Chat.Read` is user-consentable, but
# `ChannelMessage.Read.All` needs admin consent AND Microsoft's Protected APIs
# approval, and requesting the first without the second produces a Teams
# connection that only ever shows DMs — which reads as broken. Teams ships when
# both are in hand.
MICROSOFT_SIGNIN_SCOPES: tuple[str, ...] = (
    "openid",
    "profile",
    "email",
    "offline_access",
    "https://graph.microsoft.com/User.Read",
)

MICROSOFT_CONNECTOR_SCOPES: tuple[str, ...] = (
    "openid",
    "profile",
    "email",
    "offline_access",
    "https://graph.microsoft.com/User.Read",
    # Mail: read to ingest, send to execute an approved reply. Requested
    # together for the same reason Google requests gmail.readonly + gmail.send
    # together — an org connected read-only gets decisions it cannot execute,
    # and re-consent costs a second admin conversation.
    "https://graph.microsoft.com/Mail.Read",
    "https://graph.microsoft.com/Mail.Send",
    # Calendars.ReadWrite subsumes read. Graph has no separate free/busy scope
    # of the kind that bit the Google v2 grant, so one scope is sufficient.
    "https://graph.microsoft.com/Calendars.ReadWrite",
    # Files.Read.All covers the user's own OneDrive plus items shared with
    # them, which is what makes shared-document activity visible at all.
    "https://graph.microsoft.com/Files.Read.All",
)

# Bumped whenever MICROSOFT_CONNECTOR_SCOPES changes. Persisted with the
# integration at consent so a later feature needing a NEW scope can detect
# "connected under an older scope set" and trigger re-consent, rather than
# assuming access it does not have.
MICROSOFT_SCOPE_VERSION = "v1"


def normalize_scopes(scopes: str | None) -> set[str]:
    """Return the granted scope set in a comparable form.

    Entra is inconsistent about how it echoes scopes back in the token
    response: sometimes fully qualified (`https://graph.microsoft.com/Mail.Read`),
    sometimes bare (`Mail.Read`), and the casing of the permission segment is
    not guaranteed to match what was requested. A naive `scope in granted.split()`
    therefore rejects grants that were actually given — and because the callback
    refuses to write a row for a scope it believes was withheld, that presents
    to the founder as "Outlook didn't grant every permission Navis needs" on a
    consent screen where they ticked everything.

    Normalising to lowercase bare names on both sides makes the comparison mean
    what it looks like it means. The OIDC scopes (openid/profile/email/
    offline_access) have no resource prefix and pass through unchanged.
    """
    if not scopes:
        return set()
    out: set[str] = set()
    for raw in scopes.split():
        value = raw.strip()
        if not value:
            continue
        if value.lower().startswith(_GRAPH_RESOURCE):
            value = value[len(_GRAPH_RESOURCE):]
        out.add(value.lower())
    return out


def scope_granted(scopes: str | None, wanted: str) -> bool:
    """True iff `wanted` (in any of Entra's spellings) is in the granted set."""
    return next(iter(normalize_scopes(wanted)), "") in normalize_scopes(scopes)


def mail_send_granted(scopes: str | None) -> bool:
    """True iff the stored grant permits sending mail.

    Mirrors `google.oauth.gmail_send_granted`, and exists for the same reason:
    an org connected under an older scope set keeps every read feature working,
    and the sender fails LOUD with a reconnect instruction BEFORE calling Graph
    rather than surfacing a bare 403 from inside a worker retry.
    """
    return scope_granted(scopes, "Mail.Send")


def calendar_write_granted(scopes: str | None) -> bool:
    """True iff the stored grant permits creating/updating calendar events."""
    return scope_granted(scopes, "Calendars.ReadWrite")


# Refresh this many seconds before the token actually expires.
_REFRESH_SKEW_SECONDS = 120


class MicrosoftReconsentRequired(PermissionError):
    """The refresh token is gone or rejected — user revoked access, admin
    revoked the app, the 90-day inactivity window closed, or the token was
    rotated out and the rotated value was never persisted. NOT retryable: the
    only recovery is a fresh OAuth consent.

    Subclasses PermissionError to match `google.oauth.ReconsentRequired`, so a
    caller with a generic `except PermissionError` handles both suites. A
    per-vendor class rather than a shared one follows the precedent Slack set
    with `SlackReconsentRequired` — the recovery instruction differs by vendor,
    and callers that surface it to a founder need to name the right product.
    """


@dataclass(frozen=True, slots=True)
class MicrosoftToken:
    access_token: str
    expires_at: datetime
    refreshed: bool  # True if this call obtained a new token (caller must persist)
    # ALWAYS populated on a refresh — see the module docstring. Never treat a
    # value here as "optional, only when changed"; Microsoft rotates every time.
    refresh_token: str | None = None


def authorization_url(
    state: str,
    *,
    redirect_uri: str | None = None,
    scopes: tuple[str, ...] = MICROSOFT_CONNECTOR_SCOPES,
    prompt: str = "consent",
) -> str:
    """Build the Entra consent URL.

    `prompt=consent` on the connector flow so a reconnect genuinely re-prompts
    and returns a fresh refresh token, rather than silently reusing a grant the
    user is trying to repair. The sign-in flow passes `prompt=select_account`
    instead — re-consenting on every login is noise, but letting somebody pick
    which account they are signing in with is the point.
    """
    params = {
        "client_id": settings.microsoft_client_id or "",
        "redirect_uri": redirect_uri or settings.microsoft_redirect_uri or "",
        "response_type": "code",
        "response_mode": "query",
        "scope": " ".join(scopes),
        "prompt": prompt,
        "state": state,
    }
    return f"{_authorize_url()}?{urlencode(params)}"


async def exchange_code(
    code: str,
    *,
    redirect_uri: str | None = None,
    scopes: tuple[str, ...] = MICROSOFT_CONNECTOR_SCOPES,
) -> dict:
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            _token_url(),
            data={
                "code": code,
                "client_id": settings.microsoft_client_id,
                "client_secret": settings.microsoft_client_secret,
                "redirect_uri": redirect_uri or settings.microsoft_redirect_uri,
                "grant_type": "authorization_code",
                "scope": " ".join(scopes),
            },
        )
        resp.raise_for_status()
        return resp.json()


async def refresh_access_token(refresh_token: str) -> dict:
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            _token_url(),
            data={
                "refresh_token": refresh_token,
                "client_id": settings.microsoft_client_id,
                "client_secret": settings.microsoft_client_secret,
                "grant_type": "refresh_token",
            },
        )
        # Entra returns 400 with an `error` body for terminal conditions.
        # `invalid_grant` covers revoked, expired and rotated-out tokens;
        # `interaction_required` means a conditional-access policy now demands
        # the user be present. Neither is retryable — both need re-consent, and
        # retrying them burns the worker's budget on a guaranteed failure.
        if resp.status_code == 400:
            try:
                err = resp.json().get("error")
            except ValueError:
                err = None
            if err in ("invalid_grant", "interaction_required"):
                raise MicrosoftReconsentRequired(
                    f"microsoft refresh rejected ({err})"
                )
        resp.raise_for_status()
        return resp.json()


# A refresher takes a refresh_token and returns Entra's token response dict.
Refresher = Callable[[str], Awaitable[dict]]


class TokenManager:
    """Resolves a valid access token for an integration, refreshing if needed.

    `integration` is duck-typed: it must expose `access_token`, `refresh_token`
    and `token_expires_at`. Injecting a fake `refresher` makes this fully
    testable without hitting Entra.

    Deliberately a separate class from the Google TokenManager despite the
    near-identical shape. They diverge on the one behaviour that matters
    (refresh-token rotation), and a shared base class with a rotation flag
    would put the two suites one wrong default apart from silent, delayed,
    org-wide token death.
    """

    def __init__(self, integration, *, refresher: Refresher | None = None) -> None:
        self.integration = integration
        self._refresher = refresher or refresh_access_token
        # Last token resolved this session — orchestration persists it if refreshed.
        self.last_token: MicrosoftToken | None = None
        # Set by invalidate() (on a 401) to force the next get_token to refresh
        # even if local expiry says the token is still valid — the server
        # rejected it anyway.
        self._force_refresh = False

    def invalidate(self) -> None:
        """Mark the current access token as rejected; the next get_token will
        refresh. Called by the transport on a 401 so a revoked or clock-skewed
        token self-heals on retry."""
        self._force_refresh = True
        self.last_token = None

    def _is_expired(self) -> bool:
        if self._force_refresh:
            return True
        exp = getattr(self.integration, "token_expires_at", None)
        if exp is None:
            return True
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= (exp - timedelta(seconds=_REFRESH_SKEW_SECONDS))

    async def get_token(self) -> MicrosoftToken:
        from navis.signals.crypto import token_cipher

        if not self._is_expired() and self.integration.access_token:
            token = MicrosoftToken(
                access_token=token_cipher.decrypt(self.integration.access_token) or "",
                expires_at=self.integration.token_expires_at,
                refreshed=False,
            )
            self.last_token = token
            return token

        refresh_token = token_cipher.decrypt(getattr(self.integration, "refresh_token", None))
        if not refresh_token:
            raise MicrosoftReconsentRequired("integration has no refresh_token")

        data = await self._refresher(refresh_token)
        access = data["access_token"]
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=int(data.get("expires_in", 3600))
        )
        # Entra rotates on every refresh. Carry whatever came back; if it did
        # not send one, carry the token we just used so the caller still writes
        # a live value rather than leaving a stale column behind.
        rotated = data.get("refresh_token") or refresh_token
        logger.info(
            "refreshed microsoft token for integration=%s rotated=%s",
            getattr(self.integration, "id", "?"),
            bool(data.get("refresh_token")),
        )
        token = MicrosoftToken(
            access_token=access,
            expires_at=expires_at,
            refreshed=True,
            refresh_token=rotated,
        )
        self._force_refresh = False  # satisfied — the refreshed token is current
        self.last_token = token
        return token

    def token_provider(self):
        """Adapt to GraphTransport's `token_provider` (returns just the string)."""

        async def _provide() -> str:
            token = await self.get_token()
            return token.access_token

        return _provide
