"""Scheduled identity hygiene: reap expired sessions and expired revoked-token rows.

Both reapers already exist (IdentityService.purge_expired_sessions /
TokenRevocationService.purge_expired) but had no production trigger. This module wraps
them as Celery tasks (registered via navis.registry) and they are scheduled in the
platform beat_schedule.

Idempotent by construction: each is a single bulk `DELETE ... WHERE expired` served by an
index (ix_sessions_expires_at / ix_rt_expires). A re-run or an overlapping run simply
deletes the now-empty set — no double-effect, no row-by-row loop. Tasks follow the
execution/worker.py pattern: lazy imports + asyncio.run + a per-run session_scope so a
sync Celery worker drives the async DB work in its own transaction.
"""
from __future__ import annotations

from navis.platform.monitoring import get_logger

logger = get_logger("identity.maintenance")


async def _purge_sessions() -> int:
    from navis.identity.service import IdentityService
    from navis.platform.database import session_scope

    async with session_scope() as db:
        deleted = await IdentityService(db).purge_expired_sessions()
    logger.info("purged %d expired session(s)", deleted)
    return deleted


async def _purge_tokens() -> int:
    from navis.identity.tokens import TokenRevocationService
    from navis.platform.database import session_scope

    async with session_scope() as db:
        deleted = await TokenRevocationService(db).purge_expired()
    logger.info("purged %d expired revoked-token row(s)", deleted)
    return deleted


def _register_tasks():
    from navis.platform.queue import celery_app, run_async

    @celery_app.task(name="navis.identity.maintenance.purge_expired_sessions")
    def purge_expired_sessions() -> int:
        return run_async(_purge_sessions())

    @celery_app.task(name="navis.identity.maintenance.purge_expired_tokens")
    def purge_expired_tokens() -> int:
        return run_async(_purge_tokens())

    return purge_expired_sessions, purge_expired_tokens


purge_expired_sessions, purge_expired_tokens = _register_tasks()
