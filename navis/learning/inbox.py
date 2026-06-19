"""Idempotent consumer for Learning (inbox pattern). Same as Execution/Observation."""
from __future__ import annotations

from typing import Awaitable, Callable


async def process_once(
    repo, consumer: str, event_id: str, handler: Callable[[], Awaitable[None]]
) -> bool:
    if not await repo.inbox_claim(consumer, event_id):
        return False
    await handler()
    return True
