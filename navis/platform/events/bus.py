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
import time
from collections import Counter, defaultdict
from typing import Awaitable, Callable

from navis.platform.config import settings
from navis.platform.events.events import DomainEvent, EventType
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics as _metrics

logger = get_logger("events")

Handler = Callable[[DomainEvent], Awaitable[None]]
# An external transport (durable outbox / Kafka / NATS) can be registered here to
# fan events OUT of this process. Abstraction point only — not wired today, because
# in-process delivery is correct and sufficient at current scale.
ExternalPublisher = Callable[[DomainEvent], Awaitable[None]]

EVENT_STREAM = "navis:events"
DEAD_LETTER_STREAM = "navis:events:dead"


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._mirror_to_redis: bool = True
        self._timeout: float = settings.event_handler_timeout_seconds
        # Observability counters (published, handled, failures).
        self.metrics: Counter = Counter()
        self._external: ExternalPublisher | None = None

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)
        logger.debug("subscribed %s -> %s", handler.__name__, event_type.value)

    def set_external_publisher(self, publisher: ExternalPublisher | None) -> None:
        """Register a durable/distributed transport for cross-process fan-out.

        DELIVERY CONTRACT: handlers registered via @subscribe run IN THE PUBLISHING
        PROCESS. That is correct and sufficient today (every process imports the
        registry, so all handlers exist everywhere). To deliver across processes
        durably, register an external publisher here — the seam exists so adopting
        Kafka/NATS/an outbox-relay is a config change, not a rewrite.
        """
        self._external = publisher

    async def publish(self, event: DomainEvent) -> None:
        """Fan out to all in-process handlers concurrently. One failure never
        blocks others. Also mirrors to Redis and any external transport."""
        start = time.perf_counter()
        self.metrics["published"] += 1
        _metrics.incr("events.published", labels={"type": event.type.value})
        logger.info(
            "event %s org=%s id=%s corr=%s", event.type.value, event.org_id, event.id, event.correlation_id
        )

        if self._mirror_to_redis:
            await self._mirror(EVENT_STREAM, {"data": json.dumps(event.to_dict())})
        if self._external is not None:
            try:
                await self._external(event)
            except Exception:  # external transport is best-effort; never block publish
                logger.exception("external event publisher failed for %s", event.type.value)

        handlers = list(self._handlers.get(event.type, []))
        if not handlers:
            _metrics.observe_ms("events.publish_latency", (time.perf_counter() - start) * 1000)
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
                _metrics.incr("events.handled", labels={"type": event.type.value})
        _metrics.observe_ms("events.publish_latency", (time.perf_counter() - start) * 1000)

    async def _run_handler(self, handler: Handler, event: DomainEvent) -> None:
        """Run one handler under a timeout so a hung handler can't stall publish."""
        await asyncio.wait_for(handler(event), timeout=self._timeout)

    async def _dead_letter(
        self, handler: Handler, event: DomainEvent, error: BaseException
    ) -> None:
        self.metrics["failures"] += 1
        kind = "timeout" if isinstance(error, asyncio.TimeoutError) else "error"
        _metrics.incr("events.failures", labels={"type": event.type.value, "kind": kind})
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
        """Best-effort mirror to the EVENTS Redis (its own failure domain).

        NOTE: this stream is a bounded mirror for out-of-process consumers + recent
        audit — NOT the replay source of truth. Authoritative replay comes from the
        append-only domain tables in Postgres; losing the tail of this stream loses
        no truth.
        """
        if not self._mirror_to_redis:
            return
        try:
            from navis.platform.cache import events_redis_client

            await events_redis_client.xadd(
                stream, fields, maxlen=settings.event_stream_maxlen, approximate=True
            )
        except Exception:  # pragma: no cover - redis optional; mirror is best-effort
            logger.debug("redis mirror to %s skipped (redis unavailable)", stream)


event_bus = EventBus()


def subscribe(event_type: EventType) -> Callable[[Handler], Handler]:
    """Decorator to register a module-level handler with the global bus."""

    def decorator(handler: Handler) -> Handler:
        event_bus.subscribe(event_type, handler)
        return handler

    return decorator
