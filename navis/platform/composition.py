"""Composition-root guard — guarantees event subscribers exist in THIS process.

The event bus delivers in-process: a handler registered with `@subscribe` runs
inside whichever process called `publish()`. That contract is fine, but it makes
delivery depend on an import side-effect (`navis.registry`) having happened in
the *publishing* process. When it hasn't, `publish()` finds zero handlers and
returns successfully — the event is silently and permanently dropped.

That is not hypothetical. The Celery worker loaded the registry from a
`worker_process_init` hook, which Celery only fires for the prefork and solo
pools — NOT for `--pool=threads` (or gevent/eventlet), where tasks execute in
the main worker process. Under those pools the outbox relay ran, published into
an empty bus, and marked every row published.

`ensure_subscribers()` removes that dependency on how the process happened to be
started: any code path that is about to publish calls it first. The import is
idempotent (module cache), so the cost after the first call is a dict lookup.

Failure is deliberately LOUD. If the registry cannot be imported the caller must
not proceed to publish — silently continuing is exactly the data-loss mode this
module exists to prevent.
"""
from __future__ import annotations

from navis.platform.monitoring import get_logger

logger = get_logger("platform.composition")

_loaded = False


def ensure_subscribers() -> None:
    """Import the composition root so `@subscribe` handlers attach to the bus.

    Idempotent and safe to call on every task invocation. Raises if the registry
    import fails — a process that cannot wire its handlers must fail loudly
    rather than publish into an empty bus.
    """
    global _loaded
    if _loaded:
        return
    # Imported lazily: navis.registry pulls in every domain module (including the
    # ones that register Celery tasks), so a module-level import here would create
    # a cycle for anything the registry itself imports.
    import navis.registry  # noqa: F401  (subscribe + task registration side-effects)

    _loaded = True
    logger.info("composition root loaded — event subscribers registered in this process")


def reset_for_tests() -> None:
    """Clear the memo so a test can assert the guard re-runs. Test-only."""
    global _loaded
    _loaded = False
