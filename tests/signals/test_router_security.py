"""signals/router.py security hardening: OAuth state, scope validation, account email,
Pub/Sub dedup, and the production seed guard."""
from __future__ import annotations

import base64
import json
import uuid

import pytest
from fastapi import HTTPException

from navis.identity.models import IntegrationProvider
from navis.signals import router as R


class FakeRedis:
    """In-memory stand-in supporting the exact ops the router uses (set/getdel/set-nx)."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def getdel(self, key):
        return self.store.pop(key, None)


# ---- P1: OAuth state (random, server-side, single-use, attribution from store only) ----

async def test_oauth_state_is_random_and_resolves_attribution(monkeypatch):
    monkeypatch.setattr(R, "redis_client", FakeRedis())
    org, user = uuid.uuid4(), uuid.uuid4()
    state = await R._create_oauth_state(org_id=org, user_id=user, provider="gmail")
    assert state and ":" not in state and len(state) >= 32   # opaque, not org:user
    bound = await R._consume_oauth_state(state)
    assert bound["org_id"] == str(org) and bound["user_id"] == str(user)
    assert bound["provider"] == "gmail" and bound["created_at"]


async def test_oauth_state_is_single_use(monkeypatch):
    monkeypatch.setattr(R, "redis_client", FakeRedis())
    state = await R._create_oauth_state(org_id=uuid.uuid4(), user_id=uuid.uuid4(), provider="gmail")
    await R._consume_oauth_state(state)                       # first use ok
    with pytest.raises(HTTPException) as exc:                 # replay rejected
        await R._consume_oauth_state(state)
    assert exc.value.status_code == 400


async def test_oauth_state_invalid_and_missing_rejected(monkeypatch):
    monkeypatch.setattr(R, "redis_client", FakeRedis())
    for bad in (None, "", "never-issued-state"):
        with pytest.raises(HTTPException) as exc:
            await R._consume_oauth_state(bad)
        assert exc.value.status_code == 400


# ---- P2: scope validation ----

def test_scope_validation_rejects_incomplete_grant():
    gmail_ro = "https://www.googleapis.com/auth/gmail.readonly"
    assert R._missing_scopes(IntegrationProvider.GMAIL, "openid email") == [gmail_ro]
    assert R._missing_scopes(IntegrationProvider.GMAIL, f"openid {gmail_ro}") == []
    assert R._missing_scopes(IntegrationProvider.GMAIL, None) == [gmail_ro]


# ---- P1: account identity from the trusted id_token (dedup key) ----

def test_email_from_id_token():
    claims = base64.urlsafe_b64encode(json.dumps({"email": "u@x.com"}).encode()).decode().rstrip("=")
    assert R._email_from_id_token(f"header.{claims}.sig") == "u@x.com"
    assert R._email_from_id_token(None) is None
    assert R._email_from_id_token("garbage") is None          # malformed -> None (no crash)


# ---- P2: Pub/Sub redelivery dedup ----

async def test_pubsub_dedup_blocks_redelivery(monkeypatch):
    monkeypatch.setattr(R, "redis_client", FakeRedis())
    assert await R._is_duplicate_pubsub("u@x.com", "100") is False   # first delivery
    assert await R._is_duplicate_pubsub("u@x.com", "100") is True    # redelivery deduped
    assert await R._is_duplicate_pubsub("u@x.com", "101") is False   # different historyId
    assert await R._is_duplicate_pubsub(None, None) is False         # no key -> process


# ---- P2: seed endpoint disabled in production ----

async def test_seed_disabled_in_production(monkeypatch):
    from navis.platform.config import settings

    monkeypatch.setattr(settings, "environment", "production")
    body = R.SeedSignal(external_id="x", user_id=uuid.uuid4(), integration_id=uuid.uuid4())
    with pytest.raises(HTTPException) as exc:
        await R.seed_signal(body, ctx=None, db=None)  # type: ignore[arg-type]
    assert exc.value.status_code == 404


async def test_seed_allowed_outside_production(monkeypatch):
    from navis.platform.config import settings

    monkeypatch.setattr(settings, "environment", "development")
    # Should pass the guard and reach ingestion (which we stub to avoid a real DB).
    captured = {}

    class _FakeIngestion:
        def __init__(self, db):
            pass

        async def ingest(self, event):
            captured["event"] = event
            return True

    # Seed now resolves integration_id via a DB lookup — return a fake row so the
    # caller-supplied path is validated without hitting Postgres.
    supplied_integration_id = uuid.uuid4()

    class _FakeIntegration:
        id = supplied_integration_id

    class _FakeDb:
        async def scalar(self, *args, **kwargs):
            return _FakeIntegration()

        async def rollback(self):  # never fired on the happy path, but ingest may branch
            pass

    monkeypatch.setattr(R, "SignalIngestionService", _FakeIngestion)
    # Silence the background bus publish — the test asserts the sync return only.
    monkeypatch.setattr(R, "logger", R.logger)
    body = R.SeedSignal(external_id="x", integration_id=supplied_integration_id)

    class _User:
        id = uuid.uuid4()

    class _Ctx:
        org_id = uuid.uuid4()
        user = _User()

    out = await R.seed_signal(body, ctx=_Ctx(), db=_FakeDb())  # type: ignore[arg-type]
    assert out["ingested"] is True and captured["event"].external_id == "x"
    assert out["integration_id"] == str(supplied_integration_id)
