"""Alembic environment — async, settings-driven.

Imports navis.registry so every model (including the signals contract) is on
Base.metadata for autogenerate. The DB URL comes from settings, not the ini.

── MIGRATIONS DO NOT USE THE RUNTIME CONNECTION ────────────────────────
Alembic connects through `DIRECT_DATABASE_URL` when it is set, and falls back
to `DATABASE_URL` when it is not. Both name the SAME database; the difference
is which endpoint reaches it.

Runtime traffic is thousands of short transactions and belongs behind Neon's
pooler. A migration is one long-lived session issuing DDL, and PgBouncer's
transaction pooling is the wrong shape for that: session state and advisory
locks are not guaranteed to stay on one backend, so a lock taken in one
statement and released in another — or `CREATE INDEX CONCURRENTLY`, which
cannot run in a transaction at all — can land somewhere unexpected.

The fallback is deliberate backward compatibility: an environment that defines
only `DATABASE_URL` migrates exactly as it did before this existed. It is
ANNOUNCED rather than silent, because a run that quietly used the pooled
endpoint looks identical to one that used the direct endpoint right up until a
lock misbehaves.

Nothing here changes the application's engine. `navis/platform/database/
session.py` still builds from `settings.async_database_url` — the runtime URL —
and no second runtime engine is created.
"""
from __future__ import annotations

import asyncio
import sys

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy import pool

import navis.registry  # noqa: F401  (registers all ORM models on the metadata)
from navis.platform.config import settings
from navis.platform.config.settings import describe_database_url
from navis.platform.database import Base

config = context.config
target_metadata = Base.metadata


def _announce_migration_target() -> None:
    """Say which endpoint this run will migrate, without printing a credential.

    Written to STDERR so it cannot corrupt `alembic upgrade --sql`, whose
    stdout is SQL a operator may pipe straight into psql.

    Only ever `host/database` — `describe_database_url` drops the password, the
    username and the query string, and a Neon URL carries its password inline.
    """
    target = describe_database_url(settings.migration_database_url)
    if settings.migration_uses_direct_url:
        print(f"[alembic] migrating via DIRECT_DATABASE_URL -> {target}", file=sys.stderr)
    else:
        print(
            f"[alembic] DIRECT_DATABASE_URL is not set; falling back to DATABASE_URL "
            f"-> {target}",
            file=sys.stderr,
        )
        if settings.migration_db_is_pooled:
            print(
                "[alembic] WARNING: this is a POOLED endpoint. DDL is running through "
                "a transaction pooler, where session state and advisory locks are not "
                "guaranteed to stay on one backend. Set DIRECT_DATABASE_URL to the "
                "same database without '-pooler'.",
                file=sys.stderr,
            )


# Connect args follow the MIGRATION url, not the runtime one: the two endpoints
# can differ in both TLS and pooling, and using the runtime answer here would
# disable the statement cache for a direct connection that supports it.
_connect_args: dict = {}
if settings.migration_db_requires_ssl:
    _connect_args["ssl"] = True
if settings.migration_db_is_pooled:
    _connect_args["statement_cache_size"] = 0
    _connect_args["prepared_statement_cache_size"] = 0


def run_migrations_offline() -> None:
    context.configure(
        url=settings.async_migration_database_url,
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
        {"sqlalchemy.url": settings.async_migration_database_url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=_connect_args,
    )
    async with engine.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await engine.dispose()


_announce_migration_target()

if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
