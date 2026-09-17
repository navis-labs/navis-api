"""Google OAuth token lifecycle (shared by all Google connectors).

Responsibilities:
  * build the consent URL and exchange the auth code (onboarding);
  * refresh an access token from a refresh token, just-in-time, before expiry.

The TokenManager does NOT write the database — it returns refreshed tokens and a
flag so the *caller* (sync engine / orchestration) persists them on the identity
Integration. This keeps signals free of identity writes while still owning the
refresh logic. A `refresher` is injectable for tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from navis.platform.config import settings
from navis.platform.monitoring import get_logger

logger = get_logger("signals.google.oauth")

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

# Workspace scopes.
#
# Read-only across Gmail + Drive. Calendar takes BOTH `calendar.readonly`
# (required by Google to query FreeBusy — the scheduling agent MUST check
# availability before proposing) AND `calendar.events` (required to write
# a new event with attendees, so Google sends the real invite email).
#
# History: v1 requested only `calendar.readonly` — write was blocked.
#          v2 dropped readonly for `calendar.events` — this broke
#          FreeBusy (403 ACCESS_TOKEN_SCOPE_INSUFFICIENT). Confirmed
#          against Google's docs: FreeBusy requires `calendar` OR
#          `calendar.readonly` and does NOT accept `calendar.events`.
#          v3 keeps both scopes side-by-side — the narrowest working
#          combination for our end-to-end scheduling flow.
# v4 adds `gmail.send`. Without it the Phase 2B proactive-reply path cannot
# work at all: every org was connected read-only, so `messages.send` 403s and a
# reviewed, approved draft dead-letters at the last step. `gmail.send` is the
# NARROWEST scope that permits sending — it grants no read access of its own and
# cannot modify or delete existing mail, unlike `gmail.modify`.
GOOGLE_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]

# Scope substrings the callback/sender treats as "calendar write allowed".
# Both `calendar.events` and the broader `calendar` grants qualify;
# `calendar.readonly` does NOT (read-only cannot insert events).
CALENDAR_WRITE_SCOPES = (
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar",
)

# Scope substrings that grant FreeBusy (availability) query access.
# `calendar.readonly` is the standard grant per Google's docs; the
# broader `calendar` scope also works. NOTE: `calendar.events` does
# NOT include FreeBusy — that's the bug the v3 grant fixes.
CALENDAR_READ_SCOPES = (
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar",
)

# Scope substrings that permit SENDING mail. `gmail.send` is the dedicated
# send-only scope; `gmail.modify` and full `mail.google.com` also cover it and
# are accepted so an org granted a broader scope elsewhere is not forced to
# reconnect. `gmail.readonly` deliberately does NOT qualify.
GMAIL_SEND_SCOPES = (
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://mail.google.com/",
)

# Bumped whenever GOOGLE_SCOPES changes. Persisted with the integration at consent so a
# later feature that needs a NEW scope can detect "connected under an older scope set" and
# trigger re-consent, instead of assuming access it doesn't actually have.
GOOGLE_SCOPE_VERSION = "v4"


def calendar_write_granted(scopes: str | None) -> bool:
    """True iff the stored scope string contains a Calendar write scope.

    Written to `Integration.scopes` at consent as a space-separated string.
    Legacy orgs granted only `calendar.readonly` return False here so the
    scheduling flow surfaces reconsent instead of silently 403ing at Google.
    """
    if not scopes:
        return False
    granted = set(scopes.split())
    return any(w in granted for w in CALENDAR_WRITE_SCOPES)


def gmail_send_granted(scopes: str | None) -> bool:
    """True iff the stored scope string permits sending mail.

    Mirrors `calendar_write_granted`, and exists for the same reason: an org
    connected under an older scope set keeps every read feature working, and the
    sender fails LOUD with a reconnect instruction BEFORE calling Google rather
    than surfacing a bare 403 from deep inside a worker retry.
    """
    if not scopes:
        return False
    granted = set(scopes.split())
    return any(s in granted for s in GMAIL_SEND_SCOPES)


def calendar_read_granted(scopes: str | None) -> bool:
    """True iff the stored scope string contains a scope that permits
    FreeBusy queries.

    The scheduling agent MUST check availability before proposing a
    slot. `calendar.events` alone does NOT grant this — that was the
    v2→v3 bug. A v3 grant carries both `calendar.readonly` (for
    FreeBusy) and `calendar.events` (for insert). Legacy v2 grants
    return False here so the agent can emit a reconsent clarification
    with a clear message instead of a bare 403.
    """
    if not scopes:
        return False
    granted = set(scopes.split())
    return any(r in granted for r in CALENDAR_READ_SCOPES)

# Refresh this many seconds before the token actually expires.
_REFRESH_SKEW_SECONDS = 120


class ReconsentRequired(PermissionError):
    """The refresh token is gone or rejected (invalid_grant) — user revoked access, changed
    password, admin revoked the app, or the token was rotated out. NOT retryable: the only
    recovery is a fresh OAuth consent. Subclasses PermissionError for backward compatibility
    with any caller already catching that; new callers catch this specifically to surface a
    re-consent prompt / emit a TOKEN_REVOKED audit event."""


@dataclass(frozen=True, slots=True)
class GoogleToken:
    access_token: str
    expires_at: datetime
    refreshed: bool  # True if this call obtained a new token (caller should persist)
    # Google MAY rotate the refresh token on a refresh response; when it does this carries
    # the NEW one so the caller persists it. None means "unchanged — keep the stored one".
    refresh_token: str | None = None


def authorization_url(state: str, redirect_uri: str | None = None) -> str:
    from urllib.parse import urlencode

    params = {
        "client_id": settings.google_client_id or "",
        "redirect_uri": redirect_uri or settings.google_gmail_redirect_uri or "",
        "response_type": "code",
        "scope": " ".join(GOOGLE_SCOPES),
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


async def exchange_code(code: str, redirect_uri: str | None = None) -> dict:
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": redirect_uri or settings.google_gmail_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def refresh_access_token(refresh_token: str) -> dict:
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "refresh_token": refresh_token,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "grant_type": "refresh_token",
            },
        )
        # invalid_grant comes back as HTTP 400 with {"error":"invalid_grant"}. It is terminal
        # (revoked/expired/rotated-out) — surface it as ReconsentRequired so the caller stops
        # retrying and prompts re-consent, instead of treating it as a generic 4xx.
        if resp.status_code == 400:
            try:
                err = resp.json().get("error")
            except ValueError:
                err = None
            if err == "invalid_grant":
                raise ReconsentRequired("google refresh rejected (invalid_grant)")
        resp.raise_for_status()
        return resp.json()


async def revoke_token(token: str) -> bool:
    """Revoke a Google OAuth token (access OR refresh) at Google's endpoint.

    Google's `/revoke` invalidates the token and, when passed a refresh_token,
    invalidates the entire consent grant — every access token derived from it
    becomes unusable server-side. This is what makes "disconnect" mean
    disconnect: without a call here, our DB says the integration is inactive
    but Google still shows us on the user's third-party apps page and could
    hand out fresh access tokens on a re-consent bounce.

    Returns True if Google acknowledged the revocation, False on soft failure
    (network / 4xx). Never raises — a failed revoke should NOT block the
    local disconnect from proceeding. The token is already logically dead in
    our DB; a Google-side dangling grant is a minor loose end the user can
    also clean up in their Google account settings.
    """
    import httpx

    if not token:
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                GOOGLE_REVOKE_URL,
                data={"token": token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        # Google returns 200 on success, 400 if token is already invalid.
        # Both mean "the grant is gone" from the user's perspective.
        return resp.status_code in (200, 400)
    except Exception:
        logger.exception("google token revoke failed (soft, disconnect continues)")
        return False


# A refresher takes a refresh_token and returns Google's token response dict.
Refresher = Callable[[str], Awaitable[dict]]


class TokenManager:
    """Resolves a valid access token for an integration, refreshing if needed.

    `integration` is duck-typed: it must expose `access_token`, `refresh_token`,
    and `token_expires_at`. Injecting a fake `refresher` makes this fully testable
    without hitting Google.
    """

    def __init__(self, integration, *, refresher: Refresher | None = None) -> None:
        self.integration = integration
        self._refresher = refresher or refresh_access_token
        # Last token resolved this session — orchestration persists it if refreshed.
        self.last_token: GoogleToken | None = None
        # Set by invalidate() (on a 401) to force the next get_token to refresh even if the
        # local expiry says the token is still valid — the server rejected it anyway.
        self._force_refresh = False

    def invalidate(self) -> None:
        """Mark the current access token as rejected; the next get_token will refresh. Called
        by the transport on a 401 so a revoked/skewed token self-heals on retry."""
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

    async def get_token(self) -> GoogleToken:
        from navis.signals.crypto import token_cipher

        if not self._is_expired() and self.integration.access_token:
            token = GoogleToken(
                access_token=token_cipher.decrypt(self.integration.access_token) or "",
                expires_at=self.integration.token_expires_at,
                refreshed=False,
            )
            self.last_token = token
            return token

        refresh_token = token_cipher.decrypt(getattr(self.integration, "refresh_token", None))
        if not refresh_token:
            raise ReconsentRequired("integration has no refresh_token")

        data = await self._refresher(refresh_token)
        access = data["access_token"]
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(data.get("expires_in", 3600)))
        # Persist a rotated refresh token if Google returned one; otherwise keep the old one.
        rotated = data.get("refresh_token") or None
        logger.info(
            "refreshed google token for integration=%s rotated=%s",
            getattr(self.integration, "id", "?"), bool(rotated),
        )
        token = GoogleToken(
            access_token=access, expires_at=expires_at, refreshed=True, refresh_token=rotated
        )
        self._force_refresh = False  # satisfied — the refreshed token is now current
        self.last_token = token
        return token

    def token_provider(self):
        """Adapt to GoogleTransport's `token_provider` (returns just the string)."""

        async def _provide() -> str:
            token = await self.get_token()
            return token.access_token

        return _provide
