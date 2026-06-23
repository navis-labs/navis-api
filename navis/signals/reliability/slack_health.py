"""Slack webhook health — pre-empt Slack's 95% auto-disable.

Slack disables an app's event subscription if >95% of deliveries fail (non-2xx / timeout) in
a 60-min window. That is about our ENDPOINT's HTTP outcomes, NOT our async DLQ (we 200-ACK
before processing). So we count webhook outcomes per workspace per hour in Redis and alert at
>=80% — early enough to fix a signature/handler regression before Slack cuts us off.

Fail-open everywhere: a Redis hiccup must never break the webhook or the monitor.
"""
from __future__ import annotations

from typing import Any, Awaitable, cast

from navis.platform.cache import redis_client
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("signals.slack.health")

_PREFIX = "slack:health:"
_TTL = 2 * 60 * 60          # keep ~2h of buckets
_ALERT_RATIO = 0.80         # alert well before Slack's 0.95 auto-disable
_MIN_DELIVERIES = 50        # ignore tiny samples (Slack also exempts <1000/h apps)


def _bucket(team_id: str, hour: int) -> str:
    return f"{_PREFIX}{team_id}:{hour}"


async def record_webhook_outcome(team_id: str | None, *, ok: bool, hour: int) -> None:
    """Called from the webhook (hour = epoch//3600). One INCR per request — fast + fail-open."""
    if not team_id:
        return
    try:
        # cast: redis.asyncio's hincrby/hgetall aren't typed as awaitable in the stubs though
        # they are at runtime (a known redis-py typing gap), so name the awaitable explicitly.
        key = _bucket(team_id, hour)
        await cast("Awaitable[int]", redis_client.hincrby(key, "total", 1))
        if not ok:
            await cast("Awaitable[int]", redis_client.hincrby(key, "fail", 1))
        await redis_client.expire(key, _TTL)
    except Exception:  # pragma: no cover - health accounting must never break ingestion
        metrics.incr("slack.health.record_error")


async def workspace_failure_ratio(team_id: str, hour: int) -> tuple[float, int]:
    """(fail_ratio, total) for a workspace+hour. (0.0, 0) when unknown/empty."""
    try:
        data = await cast("Awaitable[dict[Any, Any]]", redis_client.hgetall(_bucket(team_id, hour)))
    except Exception:  # pragma: no cover
        return 0.0, 0
    total = int(data.get(b"total") or data.get("total") or 0)
    fail = int(data.get(b"fail") or data.get("fail") or 0)
    return (fail / total if total else 0.0), total


def is_unhealthy(fail_ratio: float, total: int) -> bool:
    """A workspace is at-risk of Slack auto-disable: enough volume AND a high failure rate."""
    return total >= _MIN_DELIVERIES and fail_ratio >= _ALERT_RATIO
