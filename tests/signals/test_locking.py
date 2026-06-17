"""P0-3: per-integration locking prevents concurrent-sync cursor races."""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from navis.signals.locking import (
    LocalLockRegistry,
    integration_advisory_lock,
    integration_lock_key,
    try_advisory_lock,
)


def test_lock_key_is_deterministic_and_fits_bigint():
    iid = uuid.uuid4()
    k1 = integration_lock_key(iid)
    k2 = integration_lock_key(iid)
    assert k1 == k2  # stable across calls/processes
    assert -(2**63) <= k1 < 2**63  # valid signed bigint for pg_advisory_lock
    assert integration_lock_key(uuid.uuid4()) != k1  # different integrations differ


class _FakeResult:
    def __init__(self, v: Any) -> None:
        self._v = v

    def scalar(self) -> Any:
        return self._v


class _FakeSession:
    """Minimal AsyncSession stand-in for advisory-lock SQL."""

    def __init__(self, lock_result: bool) -> None:
        self.lock_result = lock_result
        self.unlocked = False

    async def execute(self, stmt, params=None):
        text = str(stmt)
        if "pg_try_advisory_lock" in text:
            return _FakeResult(self.lock_result)
        if "pg_advisory_unlock" in text:
            self.unlocked = True
        return _FakeResult(None)


async def test_try_advisory_lock_returns_acquisition_result():
    assert await try_advisory_lock(_FakeSession(True), 123) is True
    assert await try_advisory_lock(_FakeSession(False), 123) is False


async def test_advisory_lock_context_unlocks_when_acquired():
    db = _FakeSession(True)
    async with integration_advisory_lock(db, uuid.uuid4()) as acquired:
        assert acquired is True
    assert db.unlocked is True  # lock released on exit


async def test_advisory_lock_context_skips_when_held_elsewhere():
    db = _FakeSession(False)
    async with integration_advisory_lock(db, uuid.uuid4()) as acquired:
        assert acquired is False
    assert db.unlocked is False  # never acquired → nothing to release


async def test_local_lock_serializes_read_modify_write():
    """The cursor race: two concurrent syncs read-modify-write shared state.
    Under the lock there is no lost update."""
    reg = LocalLockRegistry()
    iid = uuid.uuid4()
    shared = {"cursor": 0}
    order: list[int] = []

    async def worker():
        async with reg.lock(iid):
            current = shared["cursor"]
            await asyncio.sleep(0)  # force interleave if unlocked
            shared["cursor"] = current + 1
            order.append(shared["cursor"])

    await asyncio.gather(worker(), worker())

    assert shared["cursor"] == 2  # no lost update
    assert sorted(order) == [1, 2]
