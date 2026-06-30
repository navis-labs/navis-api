"""Google OAuth2 authentication service.

Handles two flows that the frontend can use:
  1. Authorization-code flow (preferred): the frontend's `initCodeClient` returns a
     `code`; this service exchanges it for tokens at Google's token endpoint, then
     extracts identity claims from the returned `id_token`.
  2. ID-token flow (legacy / fallback): the frontend passes a raw `id_token` from
     Google's `accounts.id` one-tap API; this service verifies it directly.

This module is intentionally isolated: it does NOT touch the database or produce JWTs.
It only talks to Google and returns normalized identity facts (email, sub, name).
The caller (`IdentityService.google_signin`) owns the DB work.

Security notes:
  - Code exchange is always server-side: the client_secret never leaves the backend.
  - `verify_oauth2_token` checks signature, iss, aud, and exp.
  - We reject tokens where `email_verified` is False (Google can issue these for
    newly-created or unverified accounts).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from navis.platform.config import settings
from navis.platform.monitoring import get_logger

logger = get_logger("identity.google_auth")

# Google's token endpoint — where we exchange an auth code for tokens.
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Module-level pooled httpx client. Created lazily so importing this module does
# not allocate sockets, and closed by the FastAPI lifespan on shutdown. Reused
# across requests so we don't re-do TLS to Google on every SSO login.
_HTTP_CLIENT: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
        )
    return _HTTP_CLIENT


async def aclose_http_client() -> None:
    """Drain the pooled Google OAuth client. Called from the FastAPI lifespan."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        await _HTTP_CLIENT.aclose()
        _HTTP_CLIENT = None


@dataclass(frozen=True, slots=True)
class GoogleIdentity:
    """Normalized identity claims extracted from a verified Google credential."""

    email: str
    google_sub: str          # Google's stable user identifier (never changes)
    full_name: str | None
    email_verified: bool
    hd: str | None           # Hosted domain (G Suite/Workspace), e.g. "company.com"


class GoogleAuthError(Exception):
    """Raised when Google credential verification or code exchange fails.

    Maps to HTTP 401 at the route level — the caller wraps this in HTTPException.
    """


class GoogleAuthService:
    """Verifies Google credentials and returns a normalized GoogleIdentity.

    Stateless — create one per request, no shared mutable state.
    """

    def __init__(self) -> None:
        self._client_id = settings.google_client_id
        self._client_secret = settings.google_client_secret
        # The redirect_uri MUST match what was used when initiating the OAuth flow.
        # For the popup / ux_mode="popup" code client the redirect_uri must be
        # "postmessage" (Google's special value for the popup return channel).
        self._redirect_uri = "postmessage"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def verify_code(self, code: str) -> GoogleIdentity:
        """Exchange an authorization code for tokens, then verify the id_token.

        The code flow is preferred: it grants both identity (id_token) and
        optionally Workspace scopes, and the exchange is server-side so the
        client_secret is never exposed.
        """
        if not self._client_id or not self._client_secret:
            raise GoogleAuthError("Google OAuth is not configured (missing GOOGLE_CLIENT_ID/SECRET)")

        token_data = await self._exchange_code(code)
        raw_id_token = token_data.get("id_token")
        if not raw_id_token:
            raise GoogleAuthError("Google token endpoint did not return an id_token")

        # google-auth's verify is synchronous (uses `requests` + does crypto). Off-load
        # to a worker thread so we don't block the event loop on every SSO login.
        return await asyncio.to_thread(self._verify_id_token_string, raw_id_token)

    async def verify_id_token(self, raw_id_token: str) -> GoogleIdentity:
        """Verify a raw Google id_token (e.g. from accounts.id one-tap).

        google-auth is sync (requests + crypto); we off-load via `to_thread` so the
        async route is not blocked on signature verification.
        """
        if not self._client_id:
            raise GoogleAuthError("Google OAuth is not configured (missing GOOGLE_CLIENT_ID)")
        return await asyncio.to_thread(self._verify_id_token_string, raw_id_token)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _exchange_code(self, code: str) -> dict[str, Any]:
        """POST to Google's token endpoint and return the raw JSON response."""
        payload = {
            "code": code,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "redirect_uri": self._redirect_uri,
            "grant_type": "authorization_code",
        }
        try:
            resp = await _get_http_client().post(_GOOGLE_TOKEN_URL, data=payload)
        except httpx.RequestError as exc:
            logger.warning("google token exchange network error: %s", exc)
            raise GoogleAuthError("could not reach Google's token endpoint") from exc

        if resp.status_code != 200:
            body = resp.text[:200]
            logger.warning("google token exchange failed status=%d body=%s", resp.status_code, body)
            raise GoogleAuthError(f"Google token exchange failed ({resp.status_code})")

        return resp.json()

    def _verify_id_token_string(self, raw_id_token: str) -> GoogleIdentity:
        """Verify signature, iss, aud, and exp; extract identity claims.

        Uses google-auth's `verify_oauth2_token` which fetches Google's public
        certs on the first call and caches them. This is the standard approach
        recommended by Google.
        """
        try:
            claims = google_id_token.verify_oauth2_token(
                raw_id_token,
                google_requests.Request(),
                self._client_id,
            )
        except Exception as exc:
            logger.warning("google id_token verification failed: %s", exc)
            raise GoogleAuthError(f"invalid Google credential: {exc}") from exc

        email = claims.get("email")
        sub = claims.get("sub")
        if not email or not sub:
            raise GoogleAuthError("Google id_token is missing email or sub claim")

        email_verified = bool(claims.get("email_verified", False))
        if not email_verified:
            raise GoogleAuthError(
                "Google account email is not verified — cannot use as an identity key"
            )

        return GoogleIdentity(
            email=email.strip().lower(),
            google_sub=sub,
            full_name=claims.get("name"),
            email_verified=email_verified,
            hd=claims.get("hd"),
        )
