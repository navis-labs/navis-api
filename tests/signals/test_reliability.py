"""Reliability: retries, Retry-After, transport 429, token refresh, poison isolation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalEventType,
    SignalSource,
    SyncResult,
    WorkspaceConnector,
)
from navis.signals.google.oauth import TokenManager
from navis.signals.google.transport import GoogleAPIError, GoogleTransport
from navis.signals.reliability.retry import RetryPolicy, async_retrying
from navis.signals.sync import SignalSyncService
from tests.signals.fakes import FakeDLQ, FakeIngestion, fake_integration, make_ctx

FAST = RetryPolicy(max_attempts=4, initial_backoff=0.0, max_backoff=0.0, jitter=0.0)


# ---- RetryPolicy ----

async def test_async_retrying_recovers_after_transient_failures():
    calls = {"n": 0}

    def is_retryable(exc: BaseException) -> bool:
        return isinstance(exc, GoogleAPIError) and exc.retryable

    async for attempt in async_retrying(FAST, retry_on=is_retryable):
        with attempt:
            calls["n"] += 1
            if calls["n"] < 3:
                raise GoogleAPIError(503, "transient")
            result = "ok"
    assert result == "ok"
    assert calls["n"] == 3


async def test_async_retrying_gives_up_on_non_retryable():
    def is_retryable(exc: BaseException) -> bool:
        return isinstance(exc, GoogleAPIError) and exc.retryable

    with pytest.raises(GoogleAPIError):
        async for attempt in async_retrying(FAST, retry_on=is_retryable):
            with attempt:
                raise GoogleAPIError(403, "forbidden")  # not retryable


# ---- Transport ----

async def test_transport_retries_429_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
        return httpx.Response(200, json={"ok": True})

    async def token() -> str:
        return "tok"

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = GoogleTransport(token, policy=FAST, client=client)
    result = await transport.request("GET", "https://example.test/x")
    await transport.aclose()

    assert result == {"ok": True}
    assert calls["n"] == 2


async def test_transport_self_heals_on_401():
    # 401 -> force a token refresh (on_auth_failure) and retry ONCE with the fresh token.
    calls = {"n": 0, "invalidated": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, text="unauthorized")
        return httpx.Response(200, json={"ok": True})

    tokens = iter(["stale", "fresh"])

    async def token() -> str:
        return next(tokens)

    def on_auth_failure() -> None:
        calls["invalidated"] += 1

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = GoogleTransport(token, on_auth_failure=on_auth_failure, policy=FAST, client=client)
    result = await transport.request("GET", "https://example.test/x")
    await transport.aclose()

    assert result == {"ok": True}
    assert calls["n"] == 2 and calls["invalidated"] == 1  # refreshed + retried exactly once


async def test_transport_401_without_hook_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="no")

    async def token() -> str:
        return "t"

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = GoogleTransport(token, policy=FAST, client=client)
    with pytest.raises(GoogleAPIError) as e:
        await transport.request("GET", "https://example.test/x")
    await transport.aclose()
    assert e.value.status == 401


def test_parse_retry_after_handles_seconds_and_http_date():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    from navis.signals.google.transport import _parse_retry_after

    assert _parse_retry_after("30") == 30.0
    assert _parse_retry_after(None) is None
    future = datetime.now(timezone.utc) + timedelta(seconds=120)
    secs = _parse_retry_after(format_datetime(future))  # RFC 7231 HTTP-date form
    assert secs is not None and 100 <= secs <= 121


# ---- Token refresh ----

async def test_token_refreshes_when_expired():
    integ = fake_integration(token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1))

    async def refresher(_rt: str) -> dict:
        return {"access_token": "fresh", "expires_in": 3600}

    tm = TokenManager(integ, refresher=refresher)
    token = await tm.get_token()
    assert token.access_token == "fresh"
    assert token.refreshed is True


async def test_token_reused_when_valid():
    integ = fake_integration(
        access_token="still-good",
        token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    async def refresher(_rt: str) -> dict:  # pragma: no cover - must not be called
        raise AssertionError("should not refresh")

    tm = TokenManager(integ, refresher=refresher)
    token = await tm.get_token()
    assert token.access_token == "still-good"
    assert token.refreshed is False


# ---- Poison isolation via the sync engine ----

class _StubConnector(WorkspaceConnector):
    source = SignalSource.GMAIL

    def __init__(self, events):
        self._events = events

    async def full_sync(self, ctx, client):
        return SyncResult(events=self._events, cursor={"history_id": "1"})

    async def incremental_sync(self, ctx, client, cursor):
        return SyncResult(events=self._events, cursor={"history_id": "1"})


def _ev(ctx, ext):
    return SignalEvent.create(
        ctx,
        source=SignalSource.GMAIL,
        event_type=SignalEventType.EMAIL_RECEIVED,
        external_id=ext,
        external_version="1",
    )


async def test_poison_item_is_dead_lettered_others_flow():
    ctx = make_ctx()
    events = [_ev(ctx, "good1"), _ev(ctx, "poison"), _ev(ctx, "good2")]
    ingestion = FakeIngestion(fail_on={"poison"})
    dlq = FakeDLQ()
    engine = SignalSyncService(ingestion=ingestion, dlq=dlq)

    outcome = await engine.sync(ctx, _StubConnector(events), client=None, cursor=None)

    assert outcome.ingested == 2  # good1, good2
    assert outcome.dead_lettered == 1  # poison isolated
    assert len(dlq.records) == 1
    assert {e.external_id for e in ingestion.ingested} == {"good1", "good2"}


async def test_idempotent_resync_yields_duplicates_not_dupes():
    ctx = make_ctx()
    events = [_ev(ctx, "a"), _ev(ctx, "b")]
    ingestion = FakeIngestion()
    engine = SignalSyncService(ingestion=ingestion)

    first = await engine.sync(ctx, _StubConnector(events), client=None, cursor=None)
    second = await engine.sync(ctx, _StubConnector(events), client=None, cursor={"history_id": "1"})

    assert first.ingested == 2
    assert second.ingested == 0  # replaying the same facts ingests nothing new
    assert second.duplicates == 2


async def test_batch_path_ingests_whole_chunk_in_one_call():
    # No poison -> the fast batch path handles the whole chunk (one ingest_many call).
    ctx = make_ctx()
    events = [_ev(ctx, f"e{i}") for i in range(5)]
    ingestion = FakeIngestion()
    calls = {"many": 0, "one": 0}
    orig_many, orig_one = ingestion.ingest_many, ingestion.ingest

    async def many(evs):
        calls["many"] += 1
        return await orig_many(evs)

    async def one(ev):
        calls["one"] += 1
        return await orig_one(ev)

    ingestion.ingest_many, ingestion.ingest = many, one  # type: ignore[method-assign]
    outcome = await SignalSyncService(ingestion=ingestion).sync(
        ctx, _StubConnector(events), client=None, cursor=None
    )
    assert outcome.ingested == 5
    assert calls["many"] == 1 and calls["one"] == 0  # single batch call, no per-event path


async def test_batch_failure_falls_back_per_event_and_isolates_poison():
    # A poison row aborts the batch -> fall back to per-event; good events still flow,
    # poison is dead-lettered. (FakeIngestion.ingest_many raises if any external_id poisons.)
    ctx = make_ctx()
    events = [_ev(ctx, "good1"), _ev(ctx, "poison"), _ev(ctx, "good2")]
    ingestion = FakeIngestion(fail_on={"poison"})
    dlq = FakeDLQ()
    outcome = await SignalSyncService(ingestion=ingestion, dlq=dlq).sync(
        ctx, _StubConnector(events), client=None, cursor=None
    )
    assert outcome.ingested == 2
    assert outcome.dead_lettered == 1
    assert {e.external_id for e in ingestion.ingested} == {"good1", "good2"}
