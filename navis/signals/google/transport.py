"""GoogleTransport — one resilient HTTP path for all Google API calls.

  * fetches a fresh bearer token via an injected async `token_provider`
    (so 401s self-heal by refreshing and retrying once);
  * retries 429/5xx with exponential backoff + jitter, honoring Retry-After;
  * surfaces a typed GoogleAPIError carrying status + retryability.

A single pooled httpx client is reused (connection keep-alive). For tests,
connectors take a fake client instead of this transport — the transport itself
is exercised separately.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable

import httpx

from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.reliability.retry import RetryPolicy, async_retrying

logger = get_logger("signals.google.transport")

TokenProvider = Callable[[], Awaitable[str]]
# Called on a 401 to invalidate the cached token so the immediate retry forces a refresh.
AuthFailureHook = Callable[[], None]

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GoogleAPIError(Exception):
    def __init__(self, status: int, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(f"google api {status}: {message}")
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status in _RETRYABLE_STATUS

    @property
    def is_auth_error(self) -> bool:
        return self.status == 401

    @property
    def is_not_found(self) -> bool:
        return self.status == 404

    @property
    def is_gone(self) -> bool:
        return self.status == 410


class GoogleTransport:
    def __init__(
        self,
        token_provider: TokenProvider,
        *,
        on_auth_failure: AuthFailureHook | None = None,
        policy: RetryPolicy | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token_provider = token_provider
        self._on_auth_failure = on_auth_failure
        self._policy = policy
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
    ) -> dict[str, Any]:
        def is_retryable(exc: BaseException) -> bool:
            return isinstance(exc, GoogleAPIError) and exc.retryable

        async for attempt in async_retrying(self._policy, retry_on=is_retryable):
            with attempt:
                return await self._do_request(method, url, params=params, json=json)
        raise AssertionError("unreachable")  # async_retrying reraises on exhaustion

    async def _do_request(
        self, method: str, url: str, *, params: dict | None, json: dict | None,
        _auth_retried: bool = False,
    ) -> dict[str, Any]:
        token = await self._token_provider()
        started = time.perf_counter()
        resp = await self._client.request(
            method,
            url,
            params=params,
            json=json,
            headers={"Authorization": f"Bearer {token}"},
        )
        metrics.observe_ms("google.api.latency", (time.perf_counter() - started) * 1000.0)
        metrics.incr("google.api.request")
        # 401 self-heal: the token was rejected (revoked, or rejected despite a not-yet-expired
        # local expiry). Force a refresh and retry ONCE. Without this the comment promising
        # self-heal was a lie — 401 is not in the retryable set, so it fell straight through.
        if resp.status_code == 401 and self._on_auth_failure is not None and not _auth_retried:
            metrics.incr("google.api.auth_retry")
            logger.warning("google 401 for %s; forcing token refresh + single retry", url)
            self._on_auth_failure()
            return await self._do_request(method, url, params=params, json=json, _auth_retried=True)
        if resp.status_code >= 400:
            metrics.incr("google.api.failure", labels={"status": str(resp.status_code)})
            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
            raise GoogleAPIError(resp.status_code, resp.text[:500], retry_after=retry_after)
        metrics.incr("google.api.success")
        if not resp.content:
            return {}
        return resp.json()


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header. RFC 7231 allows BOTH a delta-seconds integer and an
    HTTP-date; Google returns the date form occasionally, which the old code silently
    dropped (so the server's explicit backoff was ignored)."""
    if not value:
        return None
    value = value.strip()
    try:
        return float(value)  # delta-seconds form
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)  # HTTP-date (RFC 7231) form
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
