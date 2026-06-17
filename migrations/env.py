"""Alembic environment — async, settings-driven.

Imports navis.registry so every model (including the signals contract) is on
Base.metadata for autogenerate. The DB URL comes from settings, not the ini.
"""
from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy import pool

import navis.registry  # noqa: F401  (registers all ORM models on the metadata)
from navis.platform.config import settings
from navis.platform.database import Base

config = context.config
target_metadata = Base.metadata

_connect_args: dict = {}
if settings.db_requires_ssl:
    _connect_args["ssl"] = True
if settings.db_is_pooled:
    _connect_args["statement_cache_size"] = 0
    _connect_args["prepared_statement_cache_size"] = 0


def run_migrations_offline() -> None:
    context.configure(
        url=settings.async_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = async_engine_from_config(
        {"sqlalchemy.url": settings.async_database_url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=_connect_args,
    )
    async with engine.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
