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
