"""Liveness + readiness checks for orchestrators (k8s probes, load balancers).

Liveness = "the process is up". Readiness = "dependencies are reachable", so the
LB only routes traffic once Postgres and Redis answer.
"""
from __future__ import annotations

from navis.platform.cache import redis as cache_redis
from navis.platform.config import settings
from navis.platform.database import ping as db_ping


def liveness() -> dict:
    return {"status": "alive", "service": settings.app_name, "version": settings.app_version}


async def readiness() -> dict:
    db_ok = await db_ping()
    redis_ok = await cache_redis.ping()
    ready = db_ok and redis_ok
    return {
        "status": "ready" if ready else "degraded",
        "checks": {"database": db_ok, "redis": redis_ok},
    }
