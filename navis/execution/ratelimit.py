"""Per-provider rate limiting — stop provider storming + retry amplification.

Distributed fixed-window counter in Redis (shared across all workers, no global
lock). Fail-open: if Redis is unreachable we allow (availability over throttling),
and log. An in-memory limiter (injectable clock) is used for tests.
"""
from __future__ import annotations

import time
from typing import Protocol

from navis.execution.config import PROVIDER_RATE_LIMITS
from navis.platform.monitoring import get_logger

logger = get_logger("execution.ratelimit")


class RateLimiter(Protocol):
    async def allow(self, key: str) -> bool: ...


class RedisRateLimiter:
    def __init__(self, limits: dict[str, int], redis) -> None:
        self._limits = limits
        self._redis = redis

    async def allow(self, key: str) -> bool:
        limit = self._limits.get(key)
        if limit is None:
            return True
        window = int(time.time())
        rk = f"exec:rl:{key}:{window}"
        try:
            # SET NX EX before INCR: the TTL is attached BEFORE the key can hold a count,
            # so a crash between the two calls can never strand a TTL-less key. The old
            # incr-then-expire order leaked a PERMANENT key whenever a worker died in
            # between (window keys are never read again — pure accumulation).
            await self._redis.set(rk, 0, nx=True, ex=2)
            n = await self._redis.incr(rk)
            if n == 1:
                # INCR created the key itself (SET-vs-expiry race under clock skew) —
                # attach the TTL it would otherwise lack. Redundant on the normal path
                # (refreshes a 2s TTL on a throwaway per-second key — harmless).
                await self._redis.expire(rk, 2)
            return n <= limit
        except Exception:  # pragma: no cover - fail open on redis outage
            # WARNING, not debug: throttling silently OFF is an operational event —
            # provider-storm protection is disabled until Redis returns.
            logger.warning("rate limiter degraded (redis unavailable); allowing %s", key)
            return True


class InMemoryRateLimiter:
    """For tests / single-process dev. `clock` returns the current epoch second.
    Keeps only the CURRENT window per key (one entry per key, never grows) — the old
    (key, second)-keyed dict accumulated one entry per elapsed second forever."""

    def __init__(self, limits: dict[str, int], *, clock=lambda: int(time.time())) -> None:
        self._limits = limits
        self._clock = clock
        self._buckets: dict[str, tuple[int, int]] = {}  # key -> (window, count)

    async def allow(self, key: str) -> bool:
        limit = self._limits.get(key)
        if limit is None:
            return True
        now = self._clock()
        window, count = self._buckets.get(key, (now, 0))
        if window != now:
            window, count = now, 0
        count += 1
        self._buckets[key] = (window, count)
        return count <= limit


def default_rate_limiter() -> RateLimiter:
    from navis.platform.cache import redis_client

    return RedisRateLimiter(PROVIDER_RATE_LIMITS, redis_client)
