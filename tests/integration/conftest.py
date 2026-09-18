"""Keep the global engine's connection pool from outliving a test's event loop.

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
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module", autouse=True)
def _isolate_global_engine_pool_per_event_loop():
    from navis.platform.database.session import engine

    engine.sync_engine.dispose(close=False)
    yield
    engine.sync_engine.dispose(close=False)
