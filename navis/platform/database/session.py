"""Async SQLAlchemy engine + session management.

Production notes:
  * Neon's pooled endpoint (`-pooler`) runs PgBouncer in transaction mode, which
    is incompatible with asyncpg's prepared-statement cache. We disable both the
    asyncpg statement cache and SQLAlchemy's prepared-statement cache for pooled
    URLs, otherwise queries fail intermittently with
    `prepared statement "__asyncpg_..." does not exist` under concurrency.
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

# asyncpg needs TLS via connect_args (not the libpq URL query string).
_connect_args: dict = {}
if settings.db_requires_ssl:
    _connect_args["ssl"] = True
if settings.db_is_pooled:
    # PgBouncer transaction pooling: no server-side prepared statements.
    _connect_args["statement_cache_size"] = 0
    _connect_args["prepared_statement_cache_size"] = 0
# Hard cap on how long any single statement may run, enforced by asyncpg. A runaway
# query is cancelled instead of pinning a pooled connection forever — bounding the
# blast radius of a bad query under load. Tune via DB_STATEMENT_TIMEOUT_SECONDS.
if settings.db_statement_timeout_seconds and settings.db_statement_timeout_seconds > 0:
    _connect_args["command_timeout"] = settings.db_statement_timeout_seconds

engine = create_async_engine(
    settings.async_database_url,
    echo=settings.debug and settings.log_level.upper() == "DEBUG",
    pool_pre_ping=True,
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
