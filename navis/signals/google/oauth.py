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

# Read-only Workspace scopes. Signals never write to Google.
GOOGLE_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]

# Refresh this many seconds before the token actually expires.
_REFRESH_SKEW_SECONDS = 120


@dataclass(frozen=True, slots=True)
class GoogleToken:
    access_token: str
    expires_at: datetime
    refreshed: bool  # True if this call obtained a new token (caller should persist)


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
        resp.raise_for_status()
        return resp.json()


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

    def _is_expired(self) -> bool:
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
                access_token=token_cipher.decrypt(self.integration.access_token),
                expires_at=self.integration.token_expires_at,
                refreshed=False,
            )
            self.last_token = token
            return token

        refresh_token = token_cipher.decrypt(getattr(self.integration, "refresh_token", None))
        if not refresh_token:
            raise PermissionError("integration has no refresh_token; re-consent required")

        data = await self._refresher(refresh_token)
        access = data["access_token"]
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(data.get("expires_in", 3600)))
        logger.info("refreshed google token for integration=%s", getattr(self.integration, "id", "?"))
        token = GoogleToken(access_token=access, expires_at=expires_at, refreshed=True)
        self.last_token = token
        return token

    def token_provider(self):
        """Adapt to GoogleTransport's `token_provider` (returns just the string)."""

        async def _provide() -> str:
            token = await self.get_token()
            return token.access_token

        return _provide
