"""Shared setup for the Postgres integration modules.

WHY THIS FILE EXISTS: A SHARED REDIS POOL vs PER-MODULE EVENT LOOPS.

Every module here runs on its OWN event loop (`pytest.mark.asyncio(loop_scope=
"module")`) — that is what lets a module build its schema once and hold one
engine for its whole run. `navis.platform.cache.redis_client`, by contrast, is a
process-wide singleton created at import, and `redis.asyncio` binds each pooled
connection to the loop that opened it.

So the first module to touch Redis — through the per-organization rate limiter on
any authenticated route — leaves connections bound to ITS loop. When that module
finishes and its loop closes, the next module's first rate-limited request picks
one of those connections out of the pool and fails with

    RuntimeError: Event loop is closed

The failure lands on whichever module happens to run SECOND, which is why it
looks like a different test breaking depending on the collection order, and why
each module passes in isolation. It is a property of the harness, not of the
application: production has one loop for the process's lifetime.

The fix is to hand every module a cold pool. `aclose()` disconnects the existing
connections; the client re-opens lazily on next use, on the loop that is then
current. Module-scoped and autouse, so no test has to remember.

Deliberately NOT `monkeypatch`ing the limiter off. The limiter is part of what
these modules exercise — `test_developer_agents_postgres` asserts that it fires
and that one tenant's burst cannot spend another's budget — so it has to run for
real, against the same Redis production would use.
"""
from __future__ import annotations

import pytest_asyncio


@pytest_asyncio.fixture(scope="module", autouse=True, loop_scope="module")
async def _redis_pool_on_this_loop():
    """Disconnect the shared Redis pool at module start AND at module end.

    Both ends, on purpose. The start clears connections a previous module left
    bound to a now-closed loop; the end releases this module's own before its
    loop closes, so the next module does not inherit them either. Either half
    alone would work in one direction and leave the other order broken.

    Errors are swallowed: if Redis is unreachable the limiter already fails open
    (it is an availability control, not an authorization one), and a fixture that
    could not tidy a pool must not turn that into a collection error.
    """
    from navis.platform.cache import redis_client

    try:
        await redis_client.aclose()
    except Exception:
        pass
    yield
    try:
        await redis_client.aclose()
    except Exception:
        pass
