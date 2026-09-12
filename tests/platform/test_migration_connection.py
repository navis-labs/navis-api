"""PHASE 6.1A — migrations and the application do not share a connection.

Alembic connects through `DIRECT_DATABASE_URL` when it is set and falls back to
`DATABASE_URL` when it is not. Both name the SAME database; what differs is the
endpoint — Neon's pooled `-pooler` host for runtime traffic, the direct host for
DDL, because PgBouncer's transaction pooling does not guarantee that session
state or advisory locks stay on one backend.

THE INVARIANT THESE TESTS DEFEND, stated plainly: adding a migration-only URL
must not change what the application connects to. Every case below therefore
asserts the runtime URL as well as the migration URL — a change that made
Alembic correct while quietly re-pointing the app would pass a
migration-only test and take down production.

`Settings(...)` is constructed directly, never the module singleton, so nothing
here depends on the machine's own .env.
"""
from __future__ import annotations

import pytest

from navis.platform.config.settings import Settings, describe_database_url

POOLED = "postgresql://u:p@ep-x-pooler.c-4.us-east-2.aws.neon.tech/neondb?sslmode=require"
DIRECT = "postgresql://u:p@ep-x.c-4.us-east-2.aws.neon.tech/neondb?sslmode=require"


def _settings(**overrides) -> Settings:
    fields = {"database_url": POOLED, **overrides}
    return Settings(_env_file=None, **fields)


# ── A. DIRECT_DATABASE_URL set ──────────────────────────────────────────

def test_alembic_prefers_the_direct_url_when_it_is_set():
    s = _settings(direct_database_url=DIRECT)

    assert s.migration_uses_direct_url is True
    assert s.migration_database_url == DIRECT
    assert "-pooler" not in s.migration_database_url
    assert s.migration_db_is_pooled is False


def test_the_direct_url_does_not_change_what_the_application_connects_to():
    """The whole point. Migrations move; the runtime engine does not."""
    s = _settings(direct_database_url=DIRECT)

    assert s.database_url == POOLED
    assert "-pooler" in s.async_database_url
    # The app's engine is built from this, and it must still be the pooled one.
    assert s.async_database_url == "postgresql+asyncpg://u:p@ep-x-pooler.c-4.us-east-2.aws.neon.tech/neondb"


# ── B. DIRECT_DATABASE_URL absent ───────────────────────────────────────

def test_falls_back_to_database_url_when_direct_is_absent():
    s = _settings()

    assert s.migration_uses_direct_url is False
    assert s.migration_database_url == POOLED


def test_an_empty_direct_url_is_treated_as_absent_not_as_a_url():
    """`DIRECT_DATABASE_URL=` in a .env file is unset, not a URL of length zero."""
    for blank in ("", "   "):
        s = _settings(direct_database_url=blank)
        assert s.migration_uses_direct_url is False
        assert s.migration_database_url == POOLED


def test_the_fallback_is_detectable_as_pooled():
    """So `env.py` can warn that DDL is about to run through a pooler."""
    s = _settings()
    assert s.migration_db_is_pooled is True


# ── C. No database configured at all ────────────────────────────────────

def test_no_configured_database_fails_with_a_message_naming_both_variables():
    s = _settings(database_url="")

    with pytest.raises(ValueError) as excinfo:
        _ = s.migration_database_url

    message = str(excinfo.value)
    assert "DIRECT_DATABASE_URL" in message
    assert "DATABASE_URL" in message


# ── D. Malformed URLs ───────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["not-a-url", "postgresql://", "/just/a/path"])
def test_a_malformed_migration_url_fails_clearly(bad: str):
    s = _settings(direct_database_url=bad)

    with pytest.raises(ValueError) as excinfo:
        _ = s.migration_database_url

    assert "DIRECT_DATABASE_URL" in str(excinfo.value)


def test_a_malformed_fallback_url_names_the_variable_that_is_wrong():
    """The error must name DATABASE_URL, not the variable that was never set."""
    s = _settings(database_url="not-a-url")

    with pytest.raises(ValueError) as excinfo:
        _ = s.migration_database_url

    assert "DATABASE_URL" in str(excinfo.value)
    assert "DIRECT_DATABASE_URL" not in str(excinfo.value)


def test_a_malformed_url_error_does_not_leak_the_credential():
    s = _settings(direct_database_url="postgresql://user:hunter2@")

    with pytest.raises(ValueError) as excinfo:
        _ = s.migration_database_url

    assert "hunter2" not in str(excinfo.value)


# ── E. Runtime is untouched in every configuration ──────────────────────

@pytest.mark.parametrize("direct", [None, "", DIRECT])
def test_runtime_url_is_always_database_url_whatever_direct_is_set_to(direct):
    s = _settings(direct_database_url=direct)

    assert s.database_url == POOLED
    assert s.async_database_url.endswith("/neondb")
    assert "-pooler" in s.async_database_url
    # Runtime pooling/TLS decisions read the RUNTIME url, never the migration one.
    assert s.db_is_pooled is True
    assert s.db_requires_ssl is True


def test_migration_tls_and_pooling_are_decided_from_the_migration_url():
    """Not from the runtime URL — the two endpoints legitimately differ.

    Using the runtime answer would disable the statement cache on a direct
    connection that supports it, which is the optimisation the direct endpoint
    exists to allow.
    """
    s = _settings(direct_database_url=DIRECT)

    assert s.migration_db_is_pooled is False   # direct
    assert s.db_is_pooled is True              # runtime, still pooled
    assert s.migration_db_requires_ssl is True # neon.tech


def test_async_migration_url_is_normalised_for_asyncpg():
    s = _settings(direct_database_url=DIRECT)

    url = s.async_migration_database_url
    assert url.startswith("postgresql+asyncpg://")
    assert "?" not in url  # libpq query params stripped; asyncpg rejects them


# ── The redactor everything logs through ────────────────────────────────

def test_describe_database_url_drops_every_secret():
    described = describe_database_url(POOLED)

    assert "p@" not in described
    assert "u:" not in described
    assert "sslmode" not in described
    assert described == "ep-x-pooler.c-4.us-east-2.aws.neon.tech/neondb"


def test_describe_database_url_is_total():
    """It runs on error paths, so it must never raise."""
    for value in ("", "not-a-url", "://", "postgresql://host"):
        assert isinstance(describe_database_url(value), str)
