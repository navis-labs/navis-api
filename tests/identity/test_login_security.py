"""IDENTITY-002 (login audit) + IDENTITY-003 (brute-force throttle).

Throttle logic is exercised against an in-memory FakeRedis; the login route is driven
directly with the FakeSession + a fake Request (no live Postgres/Redis needed).
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import HTTPException
from redis.exceptions import RedisError

from navis.identity import router as R
from navis.identity.enterprise_models import AuditLog
from navis.identity.models import Membership, Role, User
from navis.identity.schemas import LoginRequest
from navis.identity.throttle import LoginThrottle
from navis.platform.security import hash_password
from tests.identity.fakes import ORG_A, FakeSession


# ---- doubles ----

class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, int] = {}

    async def get(self, k):
        v = self.store.get(k)
        return str(v) if v is not None else None

    async def incr(self, k):
        self.store[k] = int(self.store.get(k, 0)) + 1
        return self.store[k]

    async def expire(self, k, seconds):
        return True

    async def delete(self, *keys):
        n = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                n += 1
        return n


class FailingRedis:
    async def get(self, *a, **k):
        raise RedisError("down")

    async def incr(self, *a, **k):
        raise RedisError("down")

    async def expire(self, *a, **k):
        raise RedisError("down")

    async def delete(self, *a, **k):
        raise RedisError("down")


class _FakeRequest:
    def __init__(self, ip: str, xff: str | None = None) -> None:
        self.headers = {"x-forwarded-for": xff} if xff else {}
        self.client = type("C", (), {"host": ip})()


# ---- IDENTITY-003: throttle ----

async def test_locks_after_max_failures():
    t = LoginThrottle(FakeRedis(), max_attempts=3, lockout_seconds=900)
    assert not await t.is_locked("a@b.com", "1.1.1.1")
    for _ in range(3):
        await t.record_failure("a@b.com", "1.1.1.1")
    assert await t.is_locked("a@b.com", "1.1.1.1")


async def test_reset_unlocks():
    r = FakeRedis()
    t = LoginThrottle(r, max_attempts=2)
    await t.record_failure("a@b.com", "1.1.1.1")
    await t.record_failure("a@b.com", "1.1.1.1")
    assert await t.is_locked("a@b.com", "1.1.1.1")
    await t.reset("a@b.com", "1.1.1.1")
    assert not await t.is_locked("a@b.com", "1.1.1.1")


async def test_ip_counter_locks_across_emails():
    # Credential stuffing: one IP spraying many accounts trips the IP counter.
    r = FakeRedis()
    t = LoginThrottle(r, max_attempts=2)
    await t.record_failure("a@b.com", "9.9.9.9")
    await t.record_failure("c@d.com", "9.9.9.9")
    assert await t.is_locked("never-tried@x.com", "9.9.9.9")


async def test_lockout_expiry_releases():
    r = FakeRedis()
    t = LoginThrottle(r, max_attempts=2)
    await t.record_failure("a@b.com", "1.1.1.1")
    await t.record_failure("a@b.com", "1.1.1.1")
    assert await t.is_locked("a@b.com", "1.1.1.1")
    r.store.clear()  # simulate the TTL window elapsing
    assert not await t.is_locked("a@b.com", "1.1.1.1")


async def test_redis_outage_fails_open():
    t = LoginThrottle(FailingRedis(), max_attempts=2)
    assert await t.is_locked("a@b.com", "1.1.1.1") is False  # never lock on outage
    await t.record_failure("a@b.com", "1.1.1.1")             # must not raise
    await t.reset("a@b.com", "1.1.1.1")                      # must not raise


async def test_concurrent_failures_count_atomically():
    r = FakeRedis()
    t = LoginThrottle(r, max_attempts=5)
    await asyncio.gather(*[t.record_failure("a@b.com", "1.1.1.1") for _ in range(5)])
    assert r.store["login:fail:email:a@b.com"] == 5
    assert await t.is_locked("a@b.com", "1.1.1.1")


# ---- IDENTITY-002 + 003: login route ----

def _seed_user(db, email="user@x.com", password="secret", active=True):
    uid = uuid.uuid4()
    db.add(User(id=uid, email=email, hashed_password=hash_password(password), is_active=active))
    db.add(Membership(id=uuid.uuid4(), user_id=uid, organization_id=ORG_A, role=Role.FOUNDER))
    return uid


async def test_login_success_audits_once_and_resets(monkeypatch):
    monkeypatch.setattr(R, "redis_client", FakeRedis())
    db = FakeSession()
    uid = _seed_user(db)
    pair = await R.login(LoginRequest(email="user@x.com", password="secret"), _FakeRequest("1.2.3.4"), db)
    assert pair.access_token and pair.refresh_token
    logins = [a for a in db.store.get(AuditLog, []) if a.action == "login"]
    assert len(logins) == 1                       # exactly once
    assert logins[0].actor_id == uid
    assert logins[0].organization_id == ORG_A
    assert logins[0].audit_metadata.get("method") == "password"


async def test_login_locked_is_generic_and_skips_auth(monkeypatch):
    redis = FakeRedis()
    redis.store["login:fail:email:user@x.com"] = 99   # pre-locked
    monkeypatch.setattr(R, "redis_client", redis)
    db = FakeSession()  # no user seeded — must reject before touching authenticate
    with pytest.raises(HTTPException) as ei:
        await R.login(LoginRequest(email="user@x.com", password="whatever"), _FakeRequest("1.2.3.4"), db)
    assert ei.value.status_code == 401
    assert ei.value.detail == "invalid email or password"   # no "locked" leak
    assert not db.store.get(AuditLog, [])                    # no audit on a blocked attempt


async def test_login_failure_records_strike_no_audit(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(R, "redis_client", redis)
    db = FakeSession()
    _seed_user(db)
    with pytest.raises(HTTPException) as ei:
        await R.login(LoginRequest(email="user@x.com", password="WRONG"), _FakeRequest("5.5.5.5"), db)
    assert ei.value.status_code == 401
    assert ei.value.detail == "invalid email or password"
    assert redis.store.get("login:fail:email:user@x.com") == 1      # strike recorded
    assert not [a for a in db.store.get(AuditLog, []) if a.action == "login"]   # no LOGIN audit
