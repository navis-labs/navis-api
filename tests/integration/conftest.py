"""Shared setup for the Postgres integration modules.

TWO pools, two problems, two fixtures. Both are needed: they are unrelated bugs
that happen to live in the same file.

── 1. THE SQLALCHEMY / ASYNCPG ENGINE POOL ──────────────────────────────

pytest.ini sets `asyncio_default_fixture_loop_scope = function`, so most tests
run on a NEW event loop. `navis.platform.database.session.engine` is the
opposite: a module-level singleton built once at import, with a pool that
survives the whole session. asyncpg binds each connection to the loop it was
opened on, so a connection pooled by one module and handed out in the next
fails with:

    InterfaceError: cannot perform operation: another operation is in progress

This is the same one-event-loop-per-process constraint the Procfile already
records for the Celery worker ("run_async holds one event loop per process and
asyncpg pools bind to it") — it just also applies to the test session.

It surfaced as test_outbox_staged_in_txn_then_relay_publishes. Its relay pass
publishes to three event-bus subscribers, which the bus runs concurrently via
asyncio.gather; each opens its own session_scope() on the global engine. All
three failed with the error above, so no row was marked published and the relay
returned 0. The test passed alone and passed as a whole module — it only failed
once earlier integration modules had already filled the pool on their own event
loops, which is why it went unseen until CI started running these tests at all.

Deliberately SYNCHRONOUS and MODULE-scoped:

  * synchronous, because four modules here declare module-scoped async fixtures
    (loop_scope="module"); an async autouse fixture at function scope is
    rejected outright by pytest-asyncio when it meets them.
  * module-scoped, because the pollution is between modules — clearing at the
    start of each module is the granularity that matters, and it keeps the
    number of pool teardowns to one per module.

`dispose(close=False)` rather than `dispose()`: the pooled connections belong to
an event loop that has already closed, so they cannot be awaited shut. This is
SQLAlchemy's documented option for exactly that situation — drop the references
and let the connections be collected, instead of trying to close them from a
context that no longer owns them.

── 2. THE SHARED REDIS POOL ─────────────────────────────────────────────

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

import pytest
import pytest_asyncio


@pytest.fixture(scope="module", autouse=True)
def _isolate_global_engine_pool_per_event_loop():
    from navis.platform.database.session import engine

    engine.sync_engine.dispose(close=False)
    yield
    engine.sync_engine.dispose(close=False)


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
