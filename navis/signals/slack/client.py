"""Slack Web API client for onboarding backfill (conversations + users).

Read-only, bot-token. Reuses the existing retry framework — Slack rate limits hard (Tier 3
≈ 50/min on conversations.history) and signals a 429 with Retry-After, plus an in-body
`ok:false, error:"ratelimited"`. Both are surfaced as a RETRYABLE SlackAPIError so backfill
backs off instead of hammering Slack into the 95% auto-disable. The bot token is decrypted
by the caller (token_cipher) and injected per request — no shared client across workspaces.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.reliability.retry import RetryPolicy, async_retrying

logger = get_logger("signals.slack.client")

_BASE = "https://slack.com/api"
# A malformed/non-JSON body is treated as a transient gateway glitch (Cloudflare/proxy) and
# retried — bounded by the policy's max_attempts.
_RETRYABLE_ERRORS = {"ratelimited", "internal_error", "service_unavailable", "invalid_json"}
# Terminal auth failures — the token is dead; retrying cannot help, only re-consent can.
_AUTH_ERRORS = {"invalid_auth", "token_revoked", "account_inactive", "not_authed", "token_expired"}


class SlackAPIError(Exception):
    def __init__(self, error: str, *, status: int = 200, retry_after: float | None = None) -> None:
        super().__init__(f"slack api error: {error}")
        self.error = error
        self.status = status
        self.retry_after = retry_after  # honored by the retry framework's Retry-After-aware wait

    @property
    def retryable(self) -> bool:
        return self.status == 429 or self.status >= 500 or self.error in _RETRYABLE_ERRORS


class SlackReconsentRequired(SlackAPIError):
    """Terminal auth failure (invalid_auth / token_revoked / account_inactive / …) — the
    integration needs re-consent. NEVER retried. Mirrors Google's ReconsentRequired so the
    caller can deactivate the install rather than loop on a dead token."""

    @property
    def retryable(self) -> bool:
        return False


def _parse_retry_after(value: str | None) -> float | None:
    # Slack sends Retry-After as integer delta-seconds; guard non-numeric defensively so a bad
    # header can't crash the request path (the wait itself is applied by the retry framework).
    try:
        return float(value) if value else None
    except (TypeError, ValueError):
        return None


class SlackClient:
    def __init__(
        self, token: str, *, policy: RetryPolicy | None = None, client: httpx.AsyncClient | None = None
    ) -> None:
        self._token = token
        self._policy = policy
        # Conservative pool for backfill (one workspace at a time, bounded concurrency) —
        # mirrors GoogleTransport's Limits pattern; not a per-workspace pool.
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        def is_retryable(exc: BaseException) -> bool:
            return isinstance(exc, SlackAPIError) and exc.retryable

        async for attempt in async_retrying(self._policy, retry_on=is_retryable):
            with attempt:
                return await self._do(method, params)
        raise AssertionError("unreachable")

    async def _do(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        resp = await self._client.post(
            f"{_BASE}/{method}",
            data={k: v for k, v in params.items() if v is not None},
            headers={"Authorization": f"Bearer {self._token}"},
        )
        metrics.observe_ms("slack.api.latency", (time.perf_counter() - started) * 1000.0, labels={"method": method})
        metrics.incr("slack.api.request", labels={"method": method})
        if resp.status_code == 429:
            metrics.incr("slack.api.rate_limited", labels={"method": method})
            raise SlackAPIError("ratelimited", status=429, retry_after=_parse_retry_after(resp.headers.get("Retry-After")))
        if resp.status_code >= 500:
            metrics.incr("slack.api.failure", labels={"method": method, "error": "server_error"})
            raise SlackAPIError("server_error", status=resp.status_code)
        try:
            body = resp.json()
        except ValueError:  # HTML error page / Cloudflare / malformed JSON — not a crash
            metrics.incr("slack.api.failure", labels={"method": method, "error": "invalid_json"})
            raise SlackAPIError("invalid_json", status=resp.status_code)
        if not body.get("ok"):
            err = body.get("error", "unknown")
            if err in _AUTH_ERRORS:
                metrics.incr("slack.api.auth_failure", labels={"method": method, "error": err})
                raise SlackReconsentRequired(err, status=resp.status_code)
            metrics.incr("slack.api.failure", labels={"method": method, "error": err})
            raise SlackAPIError(err)
        metrics.incr("slack.api.success", labels={"method": method})
        return body

    async def conversations_list(self, *, cursor: str | None = None, limit: int = 200) -> dict[str, Any]:
        # Only types the bot can read; excludes archived by default at the caller.
        return await self._call("conversations.list", {
            "cursor": cursor, "limit": limit,
            "types": "public_channel,private_channel,mpim,im",
        })

    async def conversations_history(
        self, channel: str, *, cursor: str | None = None, limit: int = 200, oldest: str | None = None
    ) -> dict[str, Any]:
        return await self._call("conversations.history", {
            "channel": channel, "cursor": cursor, "limit": limit, "oldest": oldest,
        })

    async def users_list(self, *, cursor: str | None = None, limit: int = 200) -> dict[str, Any]:
        return await self._call("users.list", {"cursor": cursor, "limit": limit})
