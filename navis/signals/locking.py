"""Per-integration locking — prevents concurrent sync of the same integration.

Concurrent syncs (webhook push + the periodic beat + a manual run) would race the
read-modify-write of the cursor in `Integration.sync_state` and corrupt it. We guard each
sync with a PostgreSQL **advisory lock** keyed on the integration id.

  * Non-blocking `pg_try_advisory_xact_lock`: a second worker simply skips (no deadlock,
    no waiting). Single-flight, not a queue.
  * TRANSACTION-scoped: the lock auto-releases when the enclosing transaction commits or
    rolls back. This is the correct primitive under a transaction pooler (PgBouncer) — a
    *session*-level lock could leak onto a reused backend connection if an explicit unlock
    were missed (SIGNAL-LOCK-001). It also releases EXACTLY at commit, so a second worker
    can never read the cursor in the window between "unlock" and "cursor committed".
  * Namespaced + full-uuid key (SIGNAL-LOCK-003/004): the lock space is personalized to
    signals (no collision with other domains' advisory locks) and derived from the whole
    uuid (no 64-bit truncation).

Retry ownership (SIGNAL-LOCK-002): a skipped sync is NOT lost. The lock *holder* performs
the sync, and the periodic `sync-signals` beat re-runs every integration; because
incremental sync reads from the persisted cursor it is self-healing — whichever trigger
runs next captures everything since the cursor. `signals.sync.lock_skipped` is metered so a
permanently-skipped integration (a stuck holder) is observable.

`LocalLockRegistry` is an in-process asyncio equivalent for tests/single-worker dev.
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("signals.locking")

# blake2b personalization (<=16 bytes) — namespaces the signals advisory-lock space so it
# can't collide with advisory locks taken by other domains (memory/state/execution/...).
_LOCK_NAMESPACE = b"navis.signals"


def integration_lock_key(integration_id: uuid.UUID) -> int:
    """Stable signed-bigint advisory-lock key for an integration.

    blake2b over the FULL uuid (not a 64-bit truncation — SIGNAL-LOCK-004), personalized by
    the signals namespace (SIGNAL-LOCK-003), folded to a signed 64-bit int for
    pg_advisory_*lock. Deterministic across processes; ~2^-64 collision probability."""
    digest = hashlib.blake2b(integration_id.bytes, digest_size=8, person=_LOCK_NAMESPACE).digest()
    return int.from_bytes(digest, "big", signed=True)


async def try_advisory_xact_lock(db: AsyncSession, key: int) -> bool:
    """Acquire a TRANSACTION-scoped advisory lock (non-blocking). True if acquired.

    Auto-released when the enclosing transaction ends — no manual unlock, so nothing can
    leak onto a pooled backend connection, and it's released atomically with the commit."""
    result = await db.execute(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": key})
    return bool(result.scalar())


@asynccontextmanager
async def integration_advisory_lock(
    db: AsyncSession, integration_id: uuid.UUID
) -> AsyncIterator[bool]:
    """Yield True if this worker holds the integration lock, False if another does.

    The lock is bound to ``db``'s current transaction and released automatically when that
    transaction commits/rolls back — the caller MUST do its cursor write inside the same
    transaction (it does: see orchestration.sync.sync_integration)."""
    key = integration_lock_key(integration_id)
    acquired = await try_advisory_xact_lock(db, key)
    if not acquired:
        metrics.incr("signals.sync.lock_skipped")
        logger.info(
            "integration %s already syncing elsewhere; skipping (the holder syncs it; the "
            "beat + cursor-incremental sync self-heal — no signal loss)",
            integration_id,
        )
    yield acquired
    # No explicit unlock: a transaction-scoped advisory lock releases when `db`'s
    # transaction ends (commit/rollback), which is PgBouncer-safe and gap-free.


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
