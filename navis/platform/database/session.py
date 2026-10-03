"""Async SQLAlchemy engine + session management.

Production notes:
  * Neon's pooled endpoint (`-pooler`) runs PgBouncer in transaction mode WITH
    protocol-level prepared-statement tracking (PgBouncer >= 1.21,
    `max_prepared_statements` > 0), so the asyncpg and SQLAlchemy statement
    caches stay on there — see `settings.db_pooler_statement_cache` for the
    measurement and for the one kind of pooler where they must be turned off.
  * `pool_recycle` is set below Neon's idle-connection timeout so we never hand a
    dead connection to a request.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from navis.platform.config import settings
from navis.platform.config.settings import Settings


def build_connect_args(s: Settings) -> dict:
    """The asyncpg `connect_args` for the runtime engine. Pure, so it is testable."""
    args: dict = {}
    # asyncpg needs TLS via connect_args (not the libpq URL query string).
    if s.db_requires_ssl:
        args["ssl"] = True
    if s.db_is_pooled and not s.db_pooler_statement_cache:
        # A pooler that does NOT track prepared statements: cached statement
        # names would collide across the backends a transaction may land on.
        args["statement_cache_size"] = 0
        args["prepared_statement_cache_size"] = 0
    # Hard cap on how long any single statement may run, enforced by asyncpg. A
    # runaway query is cancelled instead of pinning a pooled connection forever —
    # bounding the blast radius of a bad query under load. Tune via
    # DB_STATEMENT_TIMEOUT_SECONDS.
    if s.db_statement_timeout_seconds and s.db_statement_timeout_seconds > 0:
        args["command_timeout"] = s.db_statement_timeout_seconds
    return args


_connect_args: dict = build_connect_args(settings)

engine = create_async_engine(
    settings.async_database_url,
    echo=settings.debug and settings.log_level.upper() == "DEBUG",
    # See settings.db_pool_pre_ping: this is one extra round trip per checkout,
    # which `pool_recycle` below already makes redundant against Neon's idle drop.
    pool_pre_ping=settings.db_pool_pre_ping,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_recycle=settings.db_pool_recycle_seconds,
    pool_timeout=settings.db_pool_timeout_seconds,
    connect_args=_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields a session, commits on success, rolls back on error."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Programmatic session (workers, orchestrators) outside the request cycle."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def ping() -> bool:
    """Readiness check: can we round-trip a query? Never raises."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def dispose_engine() -> None:
    """Close the pool on shutdown so connections drain cleanly."""
    await engine.dispose()
