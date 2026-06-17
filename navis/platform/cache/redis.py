"""Async Redis client + small typed cache helpers.

Production posture:
  * bounded connection pool with socket timeouts so a slow/unreachable Redis
    fails fast instead of hanging request workers;
  * retry-on-timeout for transient blips;
  * periodic health checks on pooled connections;
  * cache helpers degrade gracefully — a Redis outage never breaks a request,
    it just turns into a cache miss.
"""
from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import RedisError

from navis.platform.config import settings
from navis.platform.monitoring import get_logger

logger = get_logger("cache")

redis_client: aioredis.Redis = aioredis.from_url(
    settings.redis_url,
    encoding="utf-8",
    decode_responses=True,
    max_connections=settings.redis_max_connections,
    socket_timeout=settings.redis_socket_timeout_seconds,
    socket_connect_timeout=settings.redis_socket_timeout_seconds,
    socket_keepalive=True,
    health_check_interval=30,
    retry=Retry(ExponentialBackoff(cap=1.0, base=0.05), retries=2),
    retry_on_timeout=True,
)


async def get_redis() -> aioredis.Redis:
    return redis_client


async def ping() -> bool:
    """Readiness check. Never raises."""
    try:
        return bool(await redis_client.ping())
    except RedisError:
        return False


async def aclose() -> None:
    """Close the pool on shutdown."""
    try:
        await redis_client.aclose()
    except RedisError:  # pragma: no cover
        pass


# ---- Typed cache helpers (graceful-degrade) ----

async def cache_get_json(key: str) -> Any | None:
    try:
        raw = await redis_client.get(key)
    except RedisError:
        logger.debug("cache get failed for %s (redis unavailable)", key)
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def cache_set_json(key: str, value: Any, ttl_seconds: int | None = None) -> bool:
    try:
        await redis_client.set(key, json.dumps(value, default=str), ex=ttl_seconds)
        return True
    except RedisError:
        logger.debug("cache set failed for %s (redis unavailable)", key)
        return False


async def cache_delete(key: str) -> None:
    try:
        await redis_client.delete(key)
    except RedisError:
        pass
