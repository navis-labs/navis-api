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


# ── T-23: per-organization rate limiting on expensive routes ────────────────
#
# There was no HTTP rate limiting anywhere except the login throttle. `POST /chat`
# embeds a query, runs a full-scan vector search over the org's corpus, folds
# state and calls an LLM, and was reachable at unlimited rate — one organization
# could exhaust the DB pool, the Redis pool, the LLM quota and the API event loop
# for every other tenant.


class _CountingRedis:
    """Minimal INCR/EXPIRE stand-in that records the keys it was given."""

    def __init__(self, *, fail: bool = False) -> None:
        self.counts: dict[str, int] = {}
        self.expires: dict[str, int] = {}
        self.fail = fail

    async def incr(self, key):
        if self.fail:
            from redis.exceptions import RedisError

            raise RedisError("redis down")
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key, seconds):
        self.expires[key] = seconds
        return True


async def test_a_tenant_is_throttled_at_its_own_limit():
    import uuid as _uuid

    from navis.identity.throttle import OrgRateLimiter

    r = _CountingRedis()
    limiter = OrgRateLimiter(r, window_seconds=60)
    org = _uuid.uuid4()

    for _ in range(3):
        allowed, retry = await limiter.check("chat", org, 3)
        assert allowed is True and retry == 0

    allowed, retry = await limiter.check("chat", org, 3)
    assert allowed is False
    assert 0 < retry <= 60, "Retry-After must name a real backoff inside the window"


async def test_one_tenants_burst_does_not_throttle_another():
    """THE ISOLATION PROPERTY. The execution rate limiter is keyed on the provider
    alone and is deployment-wide; this one is keyed per organization, so a noisy
    tenant spends only its own budget."""
    import uuid as _uuid

    from navis.identity.throttle import OrgRateLimiter

    limiter = OrgRateLimiter(_CountingRedis(), window_seconds=60)
    noisy, quiet = _uuid.uuid4(), _uuid.uuid4()

    for _ in range(10):
        await limiter.check("chat", noisy, 3)
    assert (await limiter.check("chat", noisy, 3))[0] is False

    allowed, retry = await limiter.check("chat", quiet, 3)
    assert allowed is True and retry == 0, "one tenant's burst starved another"


async def test_buckets_are_independent():
    """Exhausting chat must not lock a tenant out of its brief pipeline."""
    import uuid as _uuid

    from navis.identity.throttle import OrgRateLimiter

    limiter = OrgRateLimiter(_CountingRedis(), window_seconds=60)
    org = _uuid.uuid4()

    for _ in range(5):
        await limiter.check("chat", org, 2)
    assert (await limiter.check("chat", org, 2))[0] is False
    assert (await limiter.check("pipeline", org, 2))[0] is True


async def test_every_key_is_bounded_by_a_ttl():
    """No unbounded Redis keys: the namespace is (buckets x active orgs) for one
    window and expires on its own."""
    import uuid as _uuid

    from navis.identity.throttle import OrgRateLimiter

    r = _CountingRedis()
    limiter = OrgRateLimiter(r, window_seconds=60)
    org = _uuid.uuid4()
    await limiter.check("chat", org, 5)

    assert len(r.counts) == 1
    key = next(iter(r.counts))
    assert key.startswith("ratelimit:chat:")
    assert str(org) in key, "the key must be scoped to the authenticated organization"
    assert r.expires[key] == 60, "key created without a TTL"


async def test_the_limiter_fails_open_when_redis_is_down():
    """AVAILABILITY control, not an authorization one: the request is already
    authenticated and authorized by the time it reaches here, so a cache outage
    must not become a full outage. Authorization controls fail CLOSED; this is
    deliberately the other way, matching LoginThrottle and execution.ratelimit."""
    import uuid as _uuid

    from navis.identity.throttle import OrgRateLimiter

    limiter = OrgRateLimiter(_CountingRedis(fail=True), window_seconds=60)
    allowed, retry = await limiter.check("chat", _uuid.uuid4(), 1)
    assert allowed is True and retry == 0


async def test_limits_are_configurable_and_can_be_disabled(monkeypatch):
    import uuid as _uuid

    from navis.identity.throttle import OrgRateLimiter
    from navis.platform.config import settings

    limiter = OrgRateLimiter(_CountingRedis(), window_seconds=60)
    org = _uuid.uuid4()

    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    for _ in range(50):
        assert (await limiter.check("chat", org, 1))[0] is True

    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    assert (await limiter.check("chat", org, 0))[0] is True, "limit<=0 means unlimited"


async def test_the_dependency_rejects_with_429_and_retry_after(monkeypatch):
    import uuid as _uuid

    import fastapi

    from navis.identity import throttle as T
    from navis.identity.authz import Principal, PrincipalKind

    r = _CountingRedis()
    monkeypatch.setattr(T, "_limiter", lambda: T.OrgRateLimiter(r, window_seconds=60))
    monkeypatch.setattr(T.settings, "rate_limit_chat_per_window", 1)

    dep = T.rate_limit("chat", "rate_limit_chat_per_window")
    inner = dep.__wrapped__ if hasattr(dep, "__wrapped__") else dep
    p = Principal(
        kind=PrincipalKind.USER, org_id=_uuid.uuid4(),
        subject_id=_uuid.uuid4(), permissions=frozenset(),
    )

    assert await inner(principal=p) is p           # first call allowed
    with pytest.raises(fastapi.HTTPException) as exc:
        await inner(principal=p)                    # second exceeds limit=1
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"].isdigit()


def test_the_expensive_routes_declare_a_rate_limit():
    """Structural: the day a new expensive AI/retrieval route is added, this is
    what fails until someone decides its bucket."""
    import inspect

    import navis.api.brief_router as BR
    import navis.api.chat_router as CR
    import navis.api.context_router as XR
    import navis.intelligence.router as IR

    for mod in (CR, BR, XR, IR):
        assert "rate_limit(" in inspect.getsource(mod), (
            f"{mod.__name__} exposes expensive work with no per-tenant limit"
        )
