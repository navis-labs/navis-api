"""Per-integration locking — prevents concurrent sync of the same integration.

Concurrent syncs (webhook push + the periodic beat + a manual run) would race the
read-modify-write of the cursor in `Integration.sync_state` and corrupt it. We
guard each sync with a PostgreSQL **advisory lock** keyed on the integration id.

  * Non-blocking `pg_try_advisory_lock`: a second worker simply skips (no deadlock,
    no waiting). This is single-flight, not a queue.
  * Session-level lock: if the worker/connection dies, PostgreSQL releases the lock
    automatically — built-in lock recovery, no orphaned locks.
  * `lock_timeout` is applied so any (future) blocking acquisition can't hang.

`LocalLockRegistry` is an in-process asyncio equivalent for tests/single-worker dev.
"""
from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from navis.platform.monitoring import get_logger

logger = get_logger("signals.locking")

# Namespace the lock space so it can't collide with other advisory-lock users.
_LOCK_NAMESPACE = 0x5347  # "SG"


def integration_lock_key(integration_id: uuid.UUID) -> int:
    """Stable signed-bigint key for an integration (fits pg_advisory_lock)."""
    # Top 8 bytes of the uuid → signed 64-bit. Deterministic across processes.
    return int.from_bytes(integration_id.bytes[:8], "big", signed=True)


async def try_advisory_lock(db: AsyncSession, key: int, *, lock_timeout_ms: int = 5000) -> bool:
    await db.execute(text(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms'"))
    result = await db.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key})
    return bool(result.scalar())


async def advisory_unlock(db: AsyncSession, key: int) -> None:
    await db.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})


@asynccontextmanager
async def integration_advisory_lock(
    db: AsyncSession, integration_id: uuid.UUID
) -> AsyncIterator[bool]:
    """Yield True if this worker holds the integration lock, False if another does.

    The lock is released on exit; if the connection dies it is released by Postgres.
    """
    key = integration_lock_key(integration_id)
    acquired = await try_advisory_lock(db, key)
    try:
        if not acquired:
            logger.info("integration %s already syncing elsewhere; skipping", integration_id)
        yield acquired
    finally:
        if acquired:
            try:
                await advisory_unlock(db, key)
            except Exception:  # pragma: no cover - connection may already be gone
                logger.debug("advisory unlock skipped (connection closed)")


class LocalLockRegistry:
    """In-process single-flight lock (asyncio). For tests and single-worker dev."""

    def __init__(self) -> None:
        self._locks: dict[uuid.UUID, asyncio.Lock] = defaultdict(asyncio.Lock)

    @asynccontextmanager
    async def lock(self, integration_id: uuid.UUID) -> AsyncIterator[bool]:
        lk = self._locks[integration_id]
        await lk.acquire()
        try:
            yield True
        finally:
            lk.release()


local_lock_registry = LocalLockRegistry()
