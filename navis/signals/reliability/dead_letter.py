"""DeadLetterQueue — poison-item isolation.

Source of truth is **PostgreSQL** (`signal_dead_letters`, via DeadLetterRepository).
Redis is a best-effort *secondary* alert stream carrying ids + error only — never
the payload — so a Redis outage can never lose a dead letter and no PII lands in
Redis. Replay reads from PostgreSQL.

A `repository` (PG) and/or `sink` (tests) are injectable; with neither, send
degrades to the Redis-only alert (legacy fallback) and logs loudly.
"""
from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable, cast

from navis.platform.monitoring import get_logger

logger = get_logger("signals.dlq")

Sink = Callable[[dict[str, Any]], Awaitable[None]]


class DeadLetterQueue:
    STREAM = "navis:signals:dlq"

    def __init__(self, *, repository: Any = None, sink: Sink | None = None, mirror: bool = True) -> None:
        self._repository = repository  # DeadLetterRepository (PostgreSQL, primary)
        self._sink = sink  # test hook
        self._mirror = mirror  # Redis secondary alert

    async def send(
        self,
        *,
        stage: str,
        error: BaseException | str,
        context: dict[str, Any],
        item: Any = None,
    ) -> None:
        error_str = repr(error) if isinstance(error, BaseException) else str(error)
        org_id = _as_uuid(context.get("org_id"))
        integration_id = _as_uuid(context.get("integration_id"))
        source = context.get("source")
        event_id = _as_uuid(item.get("event_id")) if isinstance(item, dict) else None
        # user_id rides along (from the failed event) so the DLQ's PII copy is GDPR-targetable.
        user_id = _as_uuid(item.get("user_id")) if isinstance(item, dict) else None
        dedupe = _dedupe_from(item)

        logger.error("DLQ stage=%s source=%s org=%s err=%s", stage, source, org_id, error_str)

        # --- PRIMARY: PostgreSQL (the source of truth) ---
        persisted = False
        if self._repository is not None:
            await self._repository.record(
                stage=stage,
                error=error_str,
                org_id=org_id,
                integration_id=integration_id,
                user_id=user_id,
                source=source,
                event_id=event_id,
                dedupe_key=dedupe,
                payload=item if isinstance(item, dict) else {"context": context},
            )
            persisted = True

        if self._sink is not None:  # tests
            await self._sink(
                {"stage": stage, "error": error_str, "context": context, "item": item}
            )
            persisted = True

        if not persisted:
            logger.critical("DLQ has no PostgreSQL repository; poison item only alerted to Redis")

        # --- SECONDARY: Redis alert (ids + error only, NO payload) ---
        if self._mirror:
            await self._alert(
                {
                    "stage": stage,
                    "source": source or "",
                    "org_id": str(org_id) if org_id else "",
                    "event_id": str(event_id) if event_id else "",
                    "dedupe_key": dedupe or "",
                    "error": error_str[:500],
                }
            )

    async def _alert(self, fields: dict[str, str]) -> None:
        try:
            from navis.platform.cache import redis_client

            # cast: redis-py types `fields` as an INVARIANT Dict[FieldT, EncodableT], so a
            # plain dict[str, str] (whose values are perfectly encodable) won't structurally
            # match. The cast documents that and silences the stub-only friction.
            await redis_client.xadd(
                self.STREAM, cast("dict[Any, Any]", fields), maxlen=100000, approximate=True
            )
        except Exception:  # pragma: no cover - redis optional; alert is best-effort
            logger.debug("DLQ redis alert skipped (redis unavailable)")


def _as_uuid(value: Any) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str) and value:
        try:
            return uuid.UUID(value)
        except ValueError:
            return None
    return None


def _dedupe_from(item: Any) -> str | None:
    # Format MUST match SignalEvent.make_dedupe_key (org, integration_id, source, external_id,
    # external_version) so a dead-lettered item's key lines up with the live signal's.
    if not isinstance(item, dict):
        return None
    org, integ, src, ext, ver = (
        item.get("org_id"),
        item.get("integration_id"),
        item.get("source"),
        item.get("external_id"),
        item.get("external_version"),
    )
    if org and integ and src and ext is not None and ver is not None:
        return f"{org}:{integ}:{src}:{ext}:{ver}"
    return None


# Legacy module singleton (Redis-only alert) for callers without a DB session.
dead_letter_queue = DeadLetterQueue()
