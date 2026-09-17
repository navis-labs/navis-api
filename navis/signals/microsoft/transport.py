"""GraphTransport — one resilient HTTP path for all Microsoft Graph calls.

Mirrors `navis.signals.google.transport.GoogleTransport`:
  * fetches a fresh bearer token via an injected async `token_provider`
    (so 401s self-heal by refreshing and retrying once);
  * retries 429/5xx with exponential backoff + jitter, honoring Retry-After;
  * surfaces a typed GraphAPIError carrying status + retryability.

A single pooled httpx client is reused (connection keep-alive). For tests,
connectors take a fake client instead of this transport — the transport itself
is exercised separately.

Graph-specific notes that shaped this file:

  * Graph throttles far more aggressively than Google's APIs and almost always
    sends a `Retry-After`. Honoring it is not an optimisation — ignoring it is
    how a backfill gets an app-wide throttle applied to the whole tenant.

  * Delta endpoints answer 410 Gone when a delta token has aged out. That is a
    normal, expected lifecycle event, not an error, so `is_gone` is surfaced as
    its own property for connectors to turn into `full_resync_required` rather
    than dead-lettering a whole sync pass.

  * A Graph error body is `{"error": {"code": ..., "message": ...}}`. The code
    string is worth keeping: `MailboxNotEnabledForRESTAPI` is the difference
    between "we are broken" and "this account has no Exchange mailbox", and
    only one of those is worth waking somebody up for.
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

logger = get_logger("signals.microsoft.transport")

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

TokenProvider = Callable[[], Awaitable[str]]
# Called on a 401 to invalidate the cached token so the immediate retry forces a refresh.
AuthFailureHook = Callable[[], None]

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GraphAPIError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        retry_after: float | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(f"graph api {status}: {message}")
        self.status = status
        self.retry_after = retry_after
        # Graph's machine-readable error code, when the body carried one.
        self.code = code

    @property
    def retryable(self) -> bool:
        return self.status in _RETRYABLE_STATUS

    @property
    def is_auth_error(self) -> bool:
        return self.status == 401

    @property
    def is_forbidden(self) -> bool:
        """403 — authenticated but not permitted. On Graph this is usually a
        missing scope (the grant predates the permission) rather than anything
        about the resource, which is why senders check the stored scope string
        BEFORE calling rather than interpreting this after."""
        return self.status == 403

    @property
    def is_not_found(self) -> bool:
        return self.status == 404

    @property
    def is_conflict(self) -> bool:
        return self.status == 409

    @property
    def is_gone(self) -> bool:
        """410 — a delta token has expired. Connectors translate this into
        `full_resync_required` rather than a failure: the data is fine, only
        our cursor into it is stale."""
        return self.status == 410


class GraphTransport:
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
        """Issue one Graph call.

        `url` may be a path relative to GRAPH_BASE ("/me/messages") or an
        absolute URL. Absolute is not a convenience: every Graph delta and
        paging response hands back a fully-qualified `@odata.nextLink` /
        `@odata.deltaLink` that already carries its own opaque token in the
        query string, and those MUST be followed verbatim. Reassembling one
        from parts drops the token and silently restarts the enumeration.
        """
        target = url if url.startswith("http") else f"{GRAPH_BASE}{url}"

        def is_retryable(exc: BaseException) -> bool:
            return isinstance(exc, GraphAPIError) and exc.retryable

        async for attempt in async_retrying(self._policy, retry_on=is_retryable):
            with attempt:
                return await self._do_request(target, method, params=params, json=json)
        raise AssertionError("unreachable")  # async_retrying reraises on exhaustion

    async def _do_request(
        self,
        url: str,
        method: str,
        *,
        params: dict | None,
        json: dict | None,
        _auth_retried: bool = False,
    ) -> dict[str, Any]:
        token = await self._token_provider()
        started = time.perf_counter()
        resp = await self._client.request(
            method,
            url,
            params=params,
            json=json,
            headers={
                "Authorization": f"Bearer {token}",
                # Ask Graph for immutable ids. Without this an item's id can
                # change when it is moved between folders, and the id is half
                # of our dedupe key — a moved message would re-ingest as a new
                # fact and the same email would appear twice in a founder's
                # signals. This header is the supported way to prevent that.
                "Prefer": 'IdType="ImmutableId"',
            },
        )
        metrics.observe_ms("graph.api.latency", (time.perf_counter() - started) * 1000.0)
        metrics.incr("graph.api.request")
        # 401 self-heal: the token was rejected (revoked, or rejected despite a
        # not-yet-expired local expiry). Force a refresh and retry ONCE. 401 is
        # deliberately not in the retryable set — retrying it without first
        # invalidating the cached token would just replay the same dead bearer.
        if resp.status_code == 401 and self._on_auth_failure is not None and not _auth_retried:
            metrics.incr("graph.api.auth_retry")
            logger.warning("graph 401 for %s; forcing token refresh + single retry", url)
            self._on_auth_failure()
            return await self._do_request(
                url, method, params=params, json=json, _auth_retried=True
            )
        if resp.status_code >= 400:
            metrics.incr("graph.api.failure", labels={"status": str(resp.status_code)})
            raise GraphAPIError(
                resp.status_code,
                resp.text[:500],
                retry_after=_parse_retry_after(resp.headers.get("Retry-After")),
                code=_error_code(resp),
            )
        metrics.incr("graph.api.success")
        if not resp.content:
            return {}
        return resp.json()


def _error_code(resp: httpx.Response) -> str | None:
    """Pull Graph's machine-readable error code out of the body, if present."""
    try:
        body = resp.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        code = error.get("code")
        return code if isinstance(code, str) else None
    return None


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header.

    RFC 7231 allows BOTH a delta-seconds integer and an HTTP-date. Graph sends
    the integer form in practice, but a parser that only handles one form
    silently drops the server's explicit backoff instruction when it meets the
    other — and dropping Graph's backoff instruction is how a tenant-wide
    throttle happens.

    (`navis.signals.google.transport` carries a private copy of this identical
    parser. Left alone on purpose: collapsing the two means editing a live
    Google code path to no functional benefit. A third connector should hoist
    this into `signals.reliability` and migrate both at once.)
    """
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
