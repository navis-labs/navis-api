"""In-process async event bus with a decorator-based subscription API.

Handlers run in-process and concurrently. Production posture:
  * each handler is bounded by a timeout so one slow handler can't pin the publish;
  * a handler that raises or times out is NOT silently swallowed — it is logged
    with full context AND pushed to a Redis dead-letter stream for replay/alerting;
  * lightweight counters are exposed for monitoring.

The publish path also mirrors every event to a Redis stream so out-of-process
consumers can later subscribe to the same vocabulary without code changes.

Usage:
    from navis.platform.events import subscribe, EventType, DomainEvent

    @subscribe(EventType.COMMITMENT_DETECTED)
    async def on_commitment(event: DomainEvent) -> None:
        ...
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from typing import Awaitable, Callable

from navis.platform.config import settings
from navis.platform.events.events import DomainEvent, EventType
from navis.platform.monitoring import get_logger

logger = get_logger("events")

Handler = Callable[[DomainEvent], Awaitable[None]]

EVENT_STREAM = "navis:events"
DEAD_LETTER_STREAM = "navis:events:dead"


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._mirror_to_redis: bool = True
        self._timeout: float = settings.event_handler_timeout_seconds
        # Observability counters (published, handled, failures).
        self.metrics: Counter = Counter()

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)
        logger.debug("subscribed %s -> %s", handler.__name__, event_type.value)

    async def publish(self, event: DomainEvent) -> None:
        """Fan out to all handlers concurrently. One failure never blocks others."""
        self.metrics["published"] += 1
        logger.info("event %s org=%s id=%s", event.type.value, event.org_id, event.id)

        if self._mirror_to_redis:
            await self._mirror(EVENT_STREAM, {"data": json.dumps(event.to_dict())})

        handlers = list(self._handlers.get(event.type, []))
        if not handlers:
            return

        results = await asyncio.gather(
            *(self._run_handler(h, event) for h in handlers),
            return_exceptions=True,
        )
        for handler, result in zip(handlers, results):
            if isinstance(result, BaseException):
                await self._dead_letter(handler, event, result)
            else:
                self.metrics["handled"] += 1

    async def _run_handler(self, handler: Handler, event: DomainEvent) -> None:
        """Run one handler under a timeout so a hung handler can't stall publish."""
        await asyncio.wait_for(handler(event), timeout=self._timeout)

    async def _dead_letter(
        self, handler: Handler, event: DomainEvent, error: BaseException
    ) -> None:
        self.metrics["failures"] += 1
        kind = "timeout" if isinstance(error, asyncio.TimeoutError) else "error"
        logger.error(
            "handler %s %s on %s (event_id=%s org=%s): %r",
            handler.__name__,
            kind,
            event.type.value,
            event.id,
            event.org_id,
            error,
            exc_info=not isinstance(error, asyncio.TimeoutError),
        )
        await self._mirror(
            DEAD_LETTER_STREAM,
            {
                "event": json.dumps(event.to_dict()),
                "handler": handler.__name__,
                "error_kind": kind,
                "error": repr(error),
            },
        )

    async def _mirror(self, stream: str, fields: dict) -> None:
        if not self._mirror_to_redis:
            return
        try:
            from navis.platform.cache import redis_client

            await redis_client.xadd(stream, fields, maxlen=10000, approximate=True)
        except Exception:  # pragma: no cover - redis optional; mirror is best-effort
            logger.debug("redis mirror to %s skipped (redis unavailable)", stream)


event_bus = EventBus()


def subscribe(event_type: EventType) -> Callable[[Handler], Handler]:
    """Decorator to register a module-level handler with the global bus."""

    def decorator(handler: Handler) -> Handler:
        event_bus.subscribe(event_type, handler)
        return handler

    return decorator
