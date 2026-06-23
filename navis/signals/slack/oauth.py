"""Slack OAuth v2 install lifecycle.

Slack bot tokens are long-lived (no expiry) unless token rotation is explicitly enabled, so
the Google refresh-on-expiry TokenManager DOES NOT apply (decision #5). This module just
exchanges the install code, learns our identity, and hands back fields the caller persists on
the existing Integration row (token sealed via token_cipher). Re-consent / rotation are future
and additive.

CR-1: we capture BOTH identifiers — `bot_user_id` (U…, from oauth.v2.access) and `bot_id`
(B…, from auth.test) — because Slack message events carry `bot_id` while the install returns
`bot_user_id`. The echo-loop guard must match on either; missing one re-feeds Navis's own
posts into Memory.
"""
from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlencode

from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("signals.slack.oauth")

SLACK_AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
SLACK_ACCESS_URL = "https://slack.com/api/oauth.v2.access"
SLACK_AUTH_TEST_URL = "https://slack.com/api/auth.test"

# Read-only, least privilege — no admin, no write/chat scopes (decision: capture only).
SLACK_SCOPES = [
    "channels:history", "groups:history", "im:history", "mpim:history",
    "channels:read", "groups:read", "users:read", "reactions:read", "files:read",
]
# An install MUST grant all of these or it cannot capture reality — reject a partial grant at
# onboarding rather than silently running blind on some channel types.
_REQUIRED_SCOPES = {
    "channels:history", "groups:history", "im:history", "mpim:history",
    "channels:read", "groups:read", "users:read", "reactions:read", "files:read",
}
SLACK_SCOPE_VERSION = "v1"


class SlackOAuthError(Exception):
    """Slack rejected the install code exchange (bad/expired code, scope, etc.)."""


def _validate_scopes(granted: str | None) -> None:
    have = set((granted or "").split(","))
    missing = _REQUIRED_SCOPES - have
    if missing:
        metrics.incr("slack.oauth.exchange.failure", labels={"error": "missing_scopes"})
        raise SlackOAuthError(f"missing required scopes: {sorted(missing)}")


def authorization_url(state: str, redirect_uri: str | None = None) -> str:
    params = {
        "client_id": settings.slack_client_id or "",
        "scope": ",".join(SLACK_SCOPES),
        "redirect_uri": redirect_uri or settings.slack_redirect_uri or "",
        "state": state,
    }
    return f"{SLACK_AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_code(code: str, redirect_uri: str | None = None) -> dict[str, Any]:
    """Exchange the install code for a bot token. Every failure mode is converted to a typed,
    actionable SlackOAuthError — a raw httpx exception or a JSON crash must never escape into
    the onboarding HTTP handler."""
    import httpx

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(SLACK_ACCESS_URL, data={
                "client_id": settings.slack_client_id,
                "client_secret": settings.slack_client_secret,
                "code": code,
                "redirect_uri": redirect_uri or settings.slack_redirect_uri,
            })
    except httpx.HTTPError as exc:  # connect/read/timeout — network failure
        metrics.incr("slack.oauth.exchange.failure", labels={"error": "network"})
        raise SlackOAuthError("oauth request failed") from exc

    if resp.status_code >= 500:
        metrics.incr("slack.oauth.exchange.failure", labels={"error": "server_error"})
        raise SlackOAuthError("oauth server error")
    try:
        body = resp.json()
    except ValueError as exc:  # HTML/Cloudflare/malformed body
        metrics.incr("slack.oauth.exchange.failure", labels={"error": "invalid_json"})
        raise SlackOAuthError("invalid oauth response") from exc
    if not body.get("ok"):
        metrics.incr("slack.oauth.exchange.failure", labels={"error": body.get("error", "unknown")})
        raise SlackOAuthError(f"oauth.v2.access failed: {body.get('error')}")

    _validate_scopes(body.get("scope"))  # reject a partial grant at onboarding
    metrics.observe_ms("slack.oauth.exchange.latency", (time.perf_counter() - started) * 1000.0)
    metrics.incr("slack.oauth.exchange.success")
    return body


async def fetch_bot_identity(bot_token: str) -> dict[str, Any]:
    """auth.test → the workspace + install identity (bot_id B…, team_id, user_id U…). bot_id is
    the identifier message events carry (echo-loop); team/user_id aid verification + support. A
    failed auth.test degrades to all-None (install still proceeds on the oauth bot_user_id) — it
    is not a hard onboarding blocker."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(SLACK_AUTH_TEST_URL, headers={"Authorization": f"Bearer {bot_token}"})
        body = resp.json()
    except (httpx.HTTPError, ValueError):
        body = {}
    if body.get("ok"):
        metrics.incr("slack.oauth.auth_test.success")
        return {"bot_id": body.get("bot_id"), "team_id": body.get("team_id"), "user_id": body.get("user_id")}
    metrics.incr("slack.oauth.auth_test.failure")
    return {"bot_id": None, "team_id": None, "user_id": None}


async def fetch_bot_id(bot_token: str) -> str | None:
    """Backward-compatible thin wrapper — existing callers want only the bot_id."""
    return (await fetch_bot_identity(bot_token)).get("bot_id")


def install_fields(access: dict[str, Any], bot_id: str | None) -> dict[str, Any]:
    """Normalize an oauth.v2.access response (+ bot_id) into the bits the caller persists on
    the Integration: team_id (external_account), the bot token to seal, granted scopes, and
    the echo-loop identity in sync_state['slack']. Fails CLOSED — never returns an install
    without a workspace identity or a token."""
    team = access.get("team") or {}
    team_id = team.get("id")
    token = access.get("access_token")
    if not team_id:
        raise SlackOAuthError("oauth response missing team.id — refusing to create an install without a workspace identity")
    if not token:
        raise SlackOAuthError("oauth response missing access_token")
    return {
        "team_id": team_id,
        "bot_token": token,
        # Rotation anchor: if Slack token rotation is later enabled it returns a refresh_token —
        # a SECRET, so it is surfaced for the caller to seal (token_cipher), NOT stored plaintext
        # in slack_state. No refresh logic today; just preserved for future migration.
        "refresh_token": access.get("refresh_token"),
        "scopes": access.get("scope"),
        "slack_state": {
            "bot_id": bot_id,                          # B… (from auth.test)
            "bot_user_id": access.get("bot_user_id"),  # U… (from oauth)
            "app_id": access.get("app_id"),
            "scope_version": SLACK_SCOPE_VERSION,
            # Non-secret rotation metadata (safe in JSONB) — present only if rotation is on.
            "token_type": access.get("token_type"),
            "token_expires_in": access.get("expires_in"),
        },
    }
