"""The runtime engine's asyncpg `connect_args`, decided per endpoint.

The prepared-statement caches are the single largest latency lever on a remote
database: with them off, every ORM statement costs three round trips instead of
one. They stay ON behind Neon's pooler, whose PgBouncer tracks protocol-level
named statements, and can be switched off for a pooler that does not
(`DB_POOLER_STATEMENT_CACHE=false`). See `settings.db_pooler_statement_cache`.

Absence of a key means "the library default" (asyncpg `statement_cache_size=100`,
SQLAlchemy `prepared_statement_cache_size=100`), so these tests assert on the
keys, not on numbers this module does not choose.

`Settings(...)` is constructed directly, never the module singleton, so nothing
here depends on the machine's own .env.
"""
from __future__ import annotations

import pytest

from navis.platform.config.settings import Settings
from navis.platform.database.session import build_connect_args

POOLED = "postgresql://u:p@ep-x-pooler.c-4.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
DIRECT = "postgresql://u:p@ep-x.c-4.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
LOCAL = "postgresql://localhost/navis"
CACHE_KEYS = ("statement_cache_size", "prepared_statement_cache_size")


def _args(**overrides) -> dict:
    return build_connect_args(Settings(_env_file=None, **overrides))


def test_pooled_endpoint_keeps_the_statement_caches_by_default():
    args = _args(database_url=POOLED)

    for key in CACHE_KEYS:
        assert key not in args


def test_pooled_endpoint_disables_both_caches_when_the_pooler_cannot_track_them():
    """The kill switch turns off BOTH layers: SQLAlchemy's adapter cache is the
    one the ORM path uses, asyncpg's is the one its own fetch path uses."""
    args = _args(database_url=POOLED, db_pooler_statement_cache=False)

    assert args["statement_cache_size"] == 0
    assert args["prepared_statement_cache_size"] == 0


@pytest.mark.parametrize("url", [DIRECT, LOCAL])
@pytest.mark.parametrize("flag", [True, False])
def test_a_direct_connection_always_uses_the_library_defaults(url, flag):
    """The switch is about poolers. A direct backend holds its own statements."""
    args = _args(database_url=url, db_pooler_statement_cache=flag)

    for key in CACHE_KEYS:
        assert key not in args


@pytest.mark.parametrize("flag", [True, False])
def test_tls_and_the_statement_timeout_do_not_depend_on_the_cache_switch(flag):
    args = _args(database_url=POOLED, db_pooler_statement_cache=flag,
                 db_statement_timeout_seconds=12.5)

    assert args["ssl"] is True
    assert args["command_timeout"] == 12.5


def test_the_switch_reads_from_the_environment(monkeypatch):
    monkeypatch.setenv("DB_POOLER_STATEMENT_CACHE", "false")

    s = Settings(_env_file=None, database_url=POOLED)

    assert s.db_pooler_statement_cache is False
    assert build_connect_args(s)["statement_cache_size"] == 0
