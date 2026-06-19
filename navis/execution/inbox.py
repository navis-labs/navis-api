"""Idempotent consumer (inbox pattern).

The outbox relay delivers events at-least-once. A consumer (e.g. Observation V2)
wraps its handler in `process_once`: the first delivery of an (consumer, event_id)
runs the handler; duplicates are no-ops. This is the standard pairing with the
transactional outbox — at-least-once delivery + idempotent consumers, no
exactly-once fantasy.
"""
from __future__ import annotations

import uuid
from typing import Awaitable, Callable


async def process_once(
    repo, consumer: str, event_id: uuid.UUID, handler: Callable[[], Awaitable[None]]
) -> bool:
    """Run `handler` iff this is the first time `consumer` sees `event_id`.

    The inbox row is claimed in the SAME transaction the handler runs in, so a
    crash before commit re-delivers and re-runs (still idempotent downstream)."""
    first_time = await repo.inbox_claim(consumer, event_id)
    if not first_time:
        return False
    await handler()
    return True
