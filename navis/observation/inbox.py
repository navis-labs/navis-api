"""Idempotent consumer (inbox pattern) for Observation.

Execution delivers events at-least-once. A consumer wraps its handler in
`process_once`: the first delivery of an (consumer, event_id) runs the handler;
duplicates are no-ops. Paired with the deterministic outcome_id, duplicate
deliveries create ONE outcome, never two.
"""
from __future__ import annotations

from typing import Awaitable, Callable


async def process_once(
    repo, consumer: str, event_id: str, handler: Callable[[], Awaitable[None]]
) -> bool:
    first_time = await repo.inbox_claim(consumer, event_id)
    if not first_time:
        return False
    await handler()
    return True
