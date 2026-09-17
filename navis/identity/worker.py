"""Identity background workers: session + token reaper tasks.

Both tables grow forever if nothing cleans them up:
  * `sessions`        — one row per refresh-token issuance; expired rows are dead weight.
  * `revoked_tokens`  — jti denylist; entries after token expiry can never be presented again.

These tasks are registered in the Celery beat schedule (platform/queue/celery_app.py) and
run hourly. Each task is idempotent and safe to run concurrently (bulk DELETE WHERE).
"""
from __future__ import annotations

import asyncio

from navis.platform.database.session import session_scope
from navis.platform.monitoring import get_logger
from navis.platform.queue.celery_app import celery_app

logger = get_logger("identity.worker")


def _run(coro):
    """Run a coroutine from a synchronous Celery task context."""
    return asyncio.get_event_loop().run_until_complete(coro)


@celery_app.task(name="navis.identity.worker.purge_expired_sessions", bind=True, max_retries=3)
def purge_expired_sessions(self) -> dict:
    """Delete sessions past their `expires_at`. The refresh token can't be used after
    expiry anyway — these rows serve no purpose and bloat the sessions table.

    Uses a bulk DELETE on the `ix_sessions_expires_at` index (O(expired rows), not O(table)).
    """
    async def _run_async() -> int:
        from navis.identity.service import IdentityService

        async with session_scope() as db:
            count = await IdentityService(db).purge_expired_sessions()
            return count

    try:
        deleted = _run(_run_async())
        logger.info("session reaper: deleted %d expired sessions", deleted)
        return {"deleted_sessions": deleted}
    except Exception as exc:
        logger.exception("session reaper failed: %r", exc)
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(name="navis.identity.worker.purge_revoked_tokens", bind=True, max_retries=3)
def purge_revoked_tokens(self) -> dict:
    """Delete revoked-token denylist entries whose JWT `exp` has already passed.

    Once a token's `exp` has passed it can never be presented again — keeping its jti in
    the denylist wastes storage and makes `is_jti_revoked()` lookups slower over time.
    Uses a bulk DELETE on `ix_rt_expires` (O(expired rows)).
    """
    async def _run_async() -> int:
        from navis.identity.tokens import TokenRevocationService

        async with session_scope() as db:
            count = await TokenRevocationService(db).purge_expired()
            return count

    try:
        deleted = _run(_run_async())
        logger.info("token reaper: deleted %d expired revoked-token entries", deleted)
        return {"deleted_revoked_tokens": deleted}
    except Exception as exc:
        logger.exception("token reaper failed: %r", exc)
        raise self.retry(exc=exc, countdown=60)
