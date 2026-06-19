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
            n = await self._redis.incr(rk)
            if n == 1:
                await self._redis.expire(rk, 2)
            return n <= limit
        except Exception:  # pragma: no cover - fail open on redis outage
            logger.debug("rate limiter degraded (redis unavailable); allowing %s", key)
            return True


class InMemoryRateLimiter:
    """For tests / single-process dev. `clock` returns the current epoch second."""

    def __init__(self, limits: dict[str, int], *, clock=lambda: int(time.time())) -> None:
        self._limits = limits
        self._clock = clock
        self._counts: dict[tuple[str, int], int] = {}

    async def allow(self, key: str) -> bool:
        limit = self._limits.get(key)
        if limit is None:
            return True
        bucket = (key, self._clock())
        self._counts[bucket] = self._counts.get(bucket, 0) + 1
        return self._counts[bucket] <= limit


def default_rate_limiter() -> RateLimiter:
    from navis.platform.cache import redis_client

    return RedisRateLimiter(PROVIDER_RATE_LIMITS, redis_client)
