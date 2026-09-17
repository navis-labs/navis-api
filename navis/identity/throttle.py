"""Redis-backed brute-force / credential-stuffing protection for /auth/login.

Two independent counters per attempt — by email and by client IP — are incremented
atomically in Redis (INCR + sliding EXPIRE), so the throttle is multi-instance safe and
horizontally scalable with no shared in-process state. Once either counter reaches
`login_max_attempts` the identifier is locked until `login_lockout_seconds` of inactivity.

Fail-OPEN by design: any Redis error allows the login to proceed (availability over
enforcement) and emits a security metric. A Redis outage must never lock every user out
or break authentication. The email counter still defends a single account when an
attacker rotates IPs; the IP counter defends against one host spraying many accounts.

The service NEVER reveals lock state to the caller — callers map every failure (bad
password, unknown email, disabled account, locked) to one generic 401 (no enumeration).
"""
from __future__ import annotations

import time

from fastapi import Depends, HTTPException, status
from redis.asyncio import Redis
from redis.exceptions import RedisError

# Safe at module scope: `authz` imports models/permissions/tokens/platform and
# never imports this module, so there is no cycle to defer around.
from navis.identity.authz import Principal, get_principal
from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("identity.throttle")


def _mask(email: str) -> str:
    """Mask an email for security logs (avoid persisting full PII in log streams)."""
    local, _, domain = email.partition("@")
    head = local[:1] if local else ""
    return f"{head}***@{domain}" if domain else "***"


class LoginThrottle:
    def __init__(self, redis: Redis, *, max_attempts: int | None = None, lockout_seconds: int | None = None) -> None:
        self.redis = redis
        self.max_attempts = max_attempts if max_attempts is not None else settings.login_max_attempts
        self.lockout_seconds = lockout_seconds if lockout_seconds is not None else settings.login_lockout_seconds

    @staticmethod
    def _email_key(email: str) -> str:
        return f"login:fail:email:{email.strip().lower()}"

    @staticmethod
    def _ip_key(ip: str) -> str:
        return f"login:fail:ip:{ip}"

    @classmethod
    def _keys(cls, email: str, ip: str) -> tuple[str, str]:
        return (cls._email_key(email), cls._ip_key(ip))

    async def is_locked(self, email: str, ip: str) -> bool:
        """True if either the email or IP counter has reached the threshold.

        Fail-open: on any Redis error, return False (allow) and emit a security metric."""
        try:
            ek, ik = self._keys(email, ip)
            ev, iv = await self.redis.get(ek), await self.redis.get(ik)
            email_hits = int(ev) if ev is not None else 0
            ip_hits = int(iv) if iv is not None else 0
            locked = email_hits >= self.max_attempts or ip_hits >= self.max_attempts
            if locked:
                metrics.incr("login.locked")
                logger.warning("login.locked email=%s ip=%s", _mask(email), ip)
            return locked
        except RedisError:
            metrics.incr("login.throttle.redis_error", labels={"op": "check"})
            logger.warning("login.throttle redis unavailable on check — failing open")
            return False

    async def record_failure(self, email: str, ip: str) -> None:
        """Increment both counters atomically and (re)set the sliding lock window."""
        try:
            for key in self._keys(email, ip):
                await self.redis.incr(key)
                await self.redis.expire(key, self.lockout_seconds)
            metrics.incr("login.failed")
            logger.info("login.failed email=%s ip=%s", _mask(email), ip)
        except RedisError:
            metrics.incr("login.throttle.redis_error", labels={"op": "record"})
            logger.warning("login.throttle redis unavailable on record — failing open")

    async def reset(self, email: str, ip: str) -> None:
        """Clear both counters after a successful login. Emits login.unlocked if the
        identifier had accumulated failures (i.e. a real unlock occurred)."""
        try:
            cleared = await self.redis.delete(*self._keys(email, ip))
            if cleared:
                metrics.incr("login.unlocked")
                logger.info("login.unlocked email=%s ip=%s", _mask(email), ip)
        except RedisError:
            metrics.incr("login.throttle.redis_error", labels={"op": "reset"})

    # ---- IP-only API (used by SSO endpoints where email is not known until AFTER
    # cryptographic verification, so only the IP counter can gate the work). The key
    # namespace is shared with the email/ip throttle above — an IP spraying both
    # /auth/login and /auth/google reaches the threshold faster, not slower.

    async def is_ip_locked(self, ip: str) -> bool:
        try:
            v = await self.redis.get(self._ip_key(ip))
            hits = int(v) if v is not None else 0
            locked = hits >= self.max_attempts
            if locked:
                metrics.incr("login.locked", labels={"by": "ip"})
                logger.warning("login.locked ip=%s (ip-only)", ip)
            return locked
        except RedisError:
            metrics.incr("login.throttle.redis_error", labels={"op": "ip_check"})
            logger.warning("login.throttle redis unavailable on ip_check — failing open")
            return False

    async def record_ip_failure(self, ip: str) -> None:
        try:
            key = self._ip_key(ip)
            await self.redis.incr(key)
            await self.redis.expire(key, self.lockout_seconds)
            metrics.incr("login.failed", labels={"by": "ip"})
            logger.info("login.failed ip=%s (ip-only)", ip)
        except RedisError:
            metrics.incr("login.throttle.redis_error", labels={"op": "ip_record"})
            logger.warning("login.throttle redis unavailable on ip_record — failing open")


# ── Per-ORGANIZATION rate limiting for expensive routes (T-23) ──────────────
#
# WHAT WAS MISSING. There was no HTTP rate limiting anywhere except the login
# throttle above — no middleware, no dependency, no per-tenant quota. `POST /chat`
# embeds a query, runs a full-scan vector search over the org's whole corpus,
# folds state and calls an LLM, and was reachable at unlimited rate by any
# authenticated principal. One organization could exhaust the database pool, the
# Redis pool, the LLM provider quota and the API process's event loop for every
# other tenant.
#
# KEYED ON THE AUTHENTICATED ORGANIZATION, never on anything the caller sends.
# `Principal.org_id` is resolved by `get_principal` from a `Membership` row, so
# there is no request field that can move a caller into another tenant's bucket
# (to spend it) or out of their own (to escape it).
#
# WHY PER-ORG AND NOT PER-USER. The property being defended is tenant isolation:
# one organization must not be able to starve another. A per-user key would let an
# org with many seats consume proportionally more shared capacity, which is the
# thing this exists to prevent. Within one tenant, how members share their own
# budget is that tenant's problem.
#
# FAILS OPEN, DELIBERATELY. This is an AVAILABILITY control, not an authorization
# one — the request is already authenticated and authorized by the time it gets
# here. If Redis is unreachable the safe direction is to serve the request and
# emit a metric, exactly as `LoginThrottle` and `execution.ratelimit` already do.
# The opposite choice would turn a cache outage into a full outage. Authorization
# controls in this codebase fail CLOSED; this one is not an authorization control.
#
# KEY SHAPE AND BOUNDEDNESS. `ratelimit:{bucket}:{org}:{window}` where `window` is
# the floor of epoch seconds to the window size. Every key gets an EXPIRE on
# creation, so the namespace is bounded by (buckets x active orgs) for one window
# and disappears on its own — no unbounded growth, no sweeper needed.

_RATE_LIMIT_PREFIX = "ratelimit:"


class OrgRateLimiter:
    """Fixed-window request counter, one bucket per (route class, organization)."""

    def __init__(self, redis: Redis, *, window_seconds: int | None = None) -> None:
        self.redis = redis
        self.window_seconds = (
            window_seconds
            if window_seconds is not None
            else settings.rate_limit_window_seconds
        )

    def _key(self, bucket: str, org_id, now: float) -> str:
        window = int(now) // self.window_seconds
        return f"{_RATE_LIMIT_PREFIX}{bucket}:{org_id}:{window}"

    async def check(self, bucket: str, org_id, limit: int) -> tuple[bool, int]:
        """`(allowed, retry_after_seconds)`. Counts the request that asks.

        INCR-then-EXPIRE rather than a Lua script: the window is short and a lost
        EXPIRE (process death between the two calls) costs one stranded key that
        the next window's key replaces, which is not worth a scripting dependency.
        """
        if not settings.rate_limit_enabled or limit <= 0:
            return True, 0
        now = time.time()
        key = self._key(bucket, org_id, now)
        try:
            hits = await self.redis.incr(key)
            if hits == 1:
                await self.redis.expire(key, self.window_seconds)
        except RedisError:
            metrics.incr("ratelimit.redis_error", labels={"bucket": bucket})
            logger.warning("rate limiter redis unavailable — failing open bucket=%s", bucket)
            return True, 0
        if hits > limit:
            # Bounded label: `bucket` is a fixed string per route class, never the
            # organization id — a per-org label would scale metric cardinality with
            # the customer count.
            metrics.incr("ratelimit.throttled", labels={"bucket": bucket})
            logger.warning(
                "rate limit exceeded bucket=%s org=%s hits=%d limit=%d",
                bucket, org_id, hits, limit,
            )
            retry_after = self.window_seconds - int(now) % self.window_seconds
            return False, max(retry_after, 1)
        return True, 0


def rate_limit(bucket: str, limit_setting: str):
    """FastAPI dependency: throttle this route per authenticated organization.

    `limit_setting` names the Settings field holding this bucket's limit, read at
    REQUEST time rather than captured at import — so a deployment can change a
    limit without a code change, and tests can move it without re-importing the
    router.

    Returns 429 with a `Retry-After` header naming the seconds until the window
    rolls, which is what lets a well-behaved client back off instead of spinning.
    """

    async def _dep(principal: Principal = Depends(get_principal)) -> Principal:
        limit = int(getattr(settings, limit_setting))
        allowed, retry_after = await _limiter().check(bucket, principal.org_id, limit)
        if not allowed:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"rate limit exceeded for this organization; retry in {retry_after}s",
                headers={"Retry-After": str(retry_after)},
            )
        return principal

    return _dep


def _limiter() -> OrgRateLimiter:
    from navis.platform.cache import redis_client

    return OrgRateLimiter(redis_client)
