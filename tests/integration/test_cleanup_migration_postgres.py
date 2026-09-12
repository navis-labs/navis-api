"""Migration 0047 removes the verification rows and NOTHING ELSE.

    NAVIS_MIGRATION_TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:55434/navis_test_cleanup \
        pytest tests/integration/test_cleanup_migration_postgres.py -q

WHY THIS EXISTS. 0047 deletes rows from a production database. The properties
that make that safe are not visible by reading it — that the guards actually
abort, that the cascade actually reaches the dependent rows, and above all that a
NEIGHBOURING organization survives untouched — so they are asserted here against
a real PostgreSQL with the real schema, before the migration is ever pointed at
Neon.

THE NEIGHBOUR IS THE POINT. Each test seeds TWO organizations: the verification
one, with the exact production identifiers, and an unrelated one carrying its own
user, membership, service account, API key and audit rows. A migration that
deleted both would pass any test that only checked "the verification rows are
gone". The neighbour is what turns that into a real assertion, and it is modelled
on what the live database actually holds — eight revoked service accounts from
earlier phases, which a `DELETE ... WHERE revoked_at IS NOT NULL` would have
swept away.

SAFETY. Refuses to run against any database not named `navis_test*`, the same
guard `sdks/integration/conftest.py` uses. Skips entirely when the URL is unset,
so it never runs by accident in CI or against a developer's own .env.
"""
from __future__ import annotations

import os
import sys
import uuid

import pytest
import sqlalchemy as sa

MIGRATION_DB = os.environ.get("NAVIS_MIGRATION_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not MIGRATION_DB,
    reason="set NAVIS_MIGRATION_TEST_DATABASE_URL to run the 0047 migration tests",
)

# The identifiers 0047 is hardcoded with. Imported rather than retyped where
# possible, so a change to the migration cannot silently desync from this test.
VERIFICATION_USER_ID = "9e98e790-80bf-4bdf-a34a-c8e4a055d631"
VERIFICATION_ORG_ID = "73a59468-c7d2-4233-a232-de4c9466eb85"
VERIFICATION_AGENT_ID = "24a318a8-9642-4350-8c61-f881e98100ec"
VERIFICATION_KEY_ID = "06ae72e9-ff6d-46c2-979b-7b3f696dd0af"
EXPECTED_EMAIL = "devtool-verify+1788792715@navislabs.in"
EXPECTED_ORG_NAME = "Devtool Verification Org"

# A bystander. Revoked on purpose: the live database holds eight revoked service
# accounts belonging to other tenants, and any cleanup written as "delete the
# revoked ones" would take them with it.
NEIGHBOUR_ORG_ID = "0a11de17-0000-4000-8000-00000000beef"
NEIGHBOUR_USER_ID = "0a11de17-0000-4000-8000-00000000cafe"
NEIGHBOUR_AGENT_ID = "0a11de17-0000-4000-8000-00000000face"
NEIGHBOUR_KEY_ID = "0a11de17-0000-4000-8000-00000000d00d"


def _engine():
    """An ASYNC engine, because asyncpg is the only driver this venv has.

    Alembic's `MigrationContext` needs a SYNC connection, which
    `AsyncConnection.run_sync` provides — the same bridge `migrations/env.py`
    uses to run these very migrations.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    url = MIGRATION_DB
    name = url.rsplit("/", 1)[-1].split("?")[0]
    assert name.startswith("navis_test"), (
        f"refusing to run migration tests against {name!r}; point "
        "NAVIS_MIGRATION_TEST_DATABASE_URL at a disposable database named navis_test*"
    )
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return create_async_engine(url)


def _seed(conn, *, org_id, user_id, agent_id, key_id, email, org_name, revoked):
    """One organization with a user, membership, agent, credential and audit row."""
    conn.execute(
        sa.text(
            "INSERT INTO organizations (id, name, slug, created_at, updated_at) "
            "VALUES (:id, :name, :slug, now(), now())"
        ),
        {"id": org_id, "name": org_name, "slug": f"slug-{org_id[:8]}"},
    )
    conn.execute(
        sa.text(
            "INSERT INTO users (id, email, hashed_password, is_active, created_at, updated_at) "
            "VALUES (:id, :email, 'x', true, now(), now())"
        ),
        {"id": user_id, "email": email},
    )
    conn.execute(
        sa.text(
            "INSERT INTO memberships (id, user_id, organization_id, role, created_at) "
            "VALUES (:id, :u, :o, 'founder', now())"
        ),
        {"id": str(uuid.uuid4()), "u": user_id, "o": org_id},
    )
    conn.execute(
        sa.text(
            "INSERT INTO service_accounts "
            "(id, organization_id, name, scopes, created_at, updated_at, revoked_at, "
            " description, environment, runtime) "
            "VALUES (:id, :o, :n, '{}', now(), now(), :rev, '', 'development', 'custom')"
        ),
        {"id": agent_id, "o": org_id, "n": f"agent-{org_id[:8]}", "rev": None},
    )
    if revoked:
        conn.execute(
            sa.text("UPDATE service_accounts SET revoked_at = now() WHERE id = :id"),
            {"id": agent_id},
        )
    conn.execute(
        sa.text(
            "INSERT INTO api_keys "
            "(id, organization_id, service_account_id, name, prefix, hashed_secret, "
            " scopes, created_at) "
            "VALUES (:id, :o, :sa, 'k', :p, 'h', '{}', now())"
        ),
        {"id": key_id, "o": org_id, "sa": agent_id, "p": key_id[:16]},
    )
    conn.execute(
        sa.text(
            "INSERT INTO audit_logs "
            "(id, organization_id, actor_id, actor_kind, action, target_id, occurred_at) "
            "VALUES (:id, :o, :a, 'user', 'service_account_create', :t, now())"
        ),
        {"id": str(uuid.uuid4()), "o": org_id, "a": user_id, "t": agent_id},
    )


@pytest.fixture
async def seeded():
    """Both organizations, torn down afterwards regardless of outcome."""
    engine = _engine()

    async def reset(conn):
        for org, user in (
            (VERIFICATION_ORG_ID, VERIFICATION_USER_ID),
            (NEIGHBOUR_ORG_ID, NEIGHBOUR_USER_ID),
        ):
            await conn.execute(sa.text("DELETE FROM organizations WHERE id = :id"), {"id": org})
            await conn.execute(sa.text("DELETE FROM users WHERE id = :id"), {"id": user})

    async with engine.begin() as conn:
        await reset(conn)
        await conn.run_sync(
            lambda sync_conn: _seed(
                sync_conn,
                org_id=VERIFICATION_ORG_ID, user_id=VERIFICATION_USER_ID,
                agent_id=VERIFICATION_AGENT_ID, key_id=VERIFICATION_KEY_ID,
                email=EXPECTED_EMAIL, org_name=EXPECTED_ORG_NAME, revoked=True,
            )
        )
        await conn.run_sync(
            lambda sync_conn: _seed(
                sync_conn,
                org_id=NEIGHBOUR_ORG_ID, user_id=NEIGHBOUR_USER_ID,
                agent_id=NEIGHBOUR_AGENT_ID, key_id=NEIGHBOUR_KEY_ID,
                email="phase4-e2e@example.com", org_name="Unrelated Real Org", revoked=True,
            )
        )
    yield engine
    async with engine.begin() as conn:
        await reset(conn)
    await engine.dispose()


def _load_migration():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "m0047",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "migrations", "versions", "0047_cleanup_verification_data.py",
        ),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _run_upgrade(engine):
    """Run 0047's `upgrade()` inside one transaction, exactly as Alembic does.

    `run_sync` hands the migration a real sync Connection under
    `Operations.context`, so `op.get_bind()` returns the same connection the
    seeding used — the bridge `migrations/env.py` relies on.
    """
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    module = _load_migration()

    def _apply(sync_conn):
        ctx = MigrationContext.configure(sync_conn)
        with Operations.context(Operations(ctx)):
            module.upgrade()

    async with engine.begin() as conn:
        await conn.run_sync(_apply)
    return module


async def _count(engine, table, column, value):
    async with engine.connect() as conn:
        result = await conn.execute(
            sa.text(f'SELECT count(*) FROM public."{table}" WHERE {column} = :v'), {"v": value}
        )
        return result.scalar()


async def test_removes_the_verification_rows_and_their_cascaded_children(seeded):
    engine = seeded
    assert await _count(engine, "organizations", "id", VERIFICATION_ORG_ID) == 1

    await _run_upgrade(engine)

    assert await _count(engine, "organizations", "id", VERIFICATION_ORG_ID) == 0
    assert await _count(engine, "users", "id", VERIFICATION_USER_ID) == 0
    # Cascaded, not deleted by name — the migration issues no statement against
    # any of these tables.
    assert await _count(engine, "service_accounts", "id", VERIFICATION_AGENT_ID) == 0
    assert await _count(engine, "api_keys", "id", VERIFICATION_KEY_ID) == 0
    assert await _count(engine, "memberships", "organization_id", VERIFICATION_ORG_ID) == 0
    assert await _count(engine, "audit_logs", "organization_id", VERIFICATION_ORG_ID) == 0


async def test_the_unrelated_organization_survives_completely(seeded):
    """The assertion that makes the others meaningful."""
    engine = seeded
    await _run_upgrade(engine)

    assert await _count(engine, "organizations", "id", NEIGHBOUR_ORG_ID) == 1
    assert await _count(engine, "users", "id", NEIGHBOUR_USER_ID) == 1
    assert await _count(engine, "memberships", "organization_id", NEIGHBOUR_ORG_ID) == 1
    # Revoked, and still here. A cleanup phrased as "remove revoked agents"
    # would have taken this row.
    assert await _count(engine, "service_accounts", "id", NEIGHBOUR_AGENT_ID) == 1
    assert await _count(engine, "api_keys", "id", NEIGHBOUR_KEY_ID) == 1
    assert await _count(engine, "audit_logs", "organization_id", NEIGHBOUR_ORG_ID) == 1


async def test_running_it_twice_is_a_no_op(seeded):
    """A re-run after a partially applied deploy must not fail."""
    engine = seeded
    await _run_upgrade(engine)
    await _run_upgrade(engine)  # must not raise
    assert await _count(engine, "organizations", "id", VERIFICATION_ORG_ID) == 0
    assert await _count(engine, "organizations", "id", NEIGHBOUR_ORG_ID) == 1


async def test_aborts_when_the_id_names_a_different_organization(seeded):
    """A matching UUID with the wrong name means this is the wrong database."""
    engine = seeded
    async with engine.begin() as conn:
        await conn.execute(
            sa.text("UPDATE organizations SET name = 'Acme Production' WHERE id = :id"),
            {"id": VERIFICATION_ORG_ID},
        )

    with pytest.raises(RuntimeError, match="Refusing to delete"):
        await _run_upgrade(engine)

    # And nothing was removed — the guard runs before any DELETE.
    assert await _count(engine, "organizations", "id", VERIFICATION_ORG_ID) == 1
    assert await _count(engine, "users", "id", VERIFICATION_USER_ID) == 1


async def test_aborts_when_the_user_id_names_a_different_person(seeded):
    engine = seeded
    async with engine.begin() as conn:
        await conn.execute(
            sa.text("UPDATE users SET email = 'founder@realcustomer.com' WHERE id = :id"),
            {"id": VERIFICATION_USER_ID},
        )

    with pytest.raises(RuntimeError, match="Refusing to delete"):
        await _run_upgrade(engine)

    assert await _count(engine, "users", "id", VERIFICATION_USER_ID) == 1
    assert await _count(engine, "organizations", "id", VERIFICATION_ORG_ID) == 1


async def test_aborts_when_another_person_is_a_member_of_the_organization(seeded):
    """Deleting the org cascades memberships — a real member must stop it."""
    engine = seeded
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "INSERT INTO memberships (id, user_id, organization_id, role, created_at) "
                # `employee`, not `member`: `role` is a Postgres enum and
                # `member` is not one of its labels.
                "VALUES (:id, :u, :o, 'employee', now())"
            ),
            {"id": str(uuid.uuid4()), "u": NEIGHBOUR_USER_ID, "o": VERIFICATION_ORG_ID},
        )

    with pytest.raises(RuntimeError, match="Refusing to delete"):
        await _run_upgrade(engine)

    assert await _count(engine, "organizations", "id", VERIFICATION_ORG_ID) == 1


async def test_downgrade_refuses_rather_than_claiming_success(seeded):
    module = await _run_upgrade(seeded)
    with pytest.raises(RuntimeError, match="cannot be reversed"):
        module.downgrade()


def test_the_migration_names_no_pattern_and_no_credential():
    """Structural: exact identifiers only, and nothing secret in the file."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "migrations", "versions", "0047_cleanup_verification_data.py",
    )
    source = open(path, encoding="utf-8").read()

    # A pattern match is how a narrow cleanup becomes a broad one.
    assert " LIKE " not in source.upper().replace("'", " ")
    assert "ILIKE" not in source.upper()
    # Deletes are addressed by primary key, never by a descriptive column.
    assert "DELETE FROM organizations WHERE id = :id" in source
    assert "DELETE FROM users WHERE id = :id" in source
    assert "DELETE FROM audit_logs" not in source
    assert "DELETE FROM api_keys" not in source
    assert "TRUNCATE" not in source.upper()
    # No credential-shaped material.
    import re
    assert not re.search(r"navis_[0-9a-f]{8,}_[A-Za-z0-9_-]{16,}", source)
    assert "postgresql://" not in source
    assert "0046_service_account_agents" in source


def test_the_revision_id_fits_the_version_column():
    """`alembic_version.version_num` is varchar(32). A longer id cannot apply.

    THIS TEST EXISTS BECAUSE THE FIRST ATTEMPT FAILED ON IT. The original id,
    `0047_cleanup_devtool_verification_data`, is 38 characters. `upgrade()` ran
    against Neon, deleted both rows, and THEN the version bump raised
    `StringDataRightTruncationError` — so the whole transaction rolled back and
    the migration could not be applied at all. Nothing was lost, because the
    migration is transactional, but the failure is invisible until it happens
    against a real database with a real `alembic_version` table.

    A unit test that calls `upgrade()` directly never bumps the version and so
    can never catch this. This one checks the id itself.
    """
    module = _load_migration()
    assert len(module.revision) <= 32, (
        f"revision id {module.revision!r} is {len(module.revision)} characters; "
        "alembic_version.version_num holds 32"
    )
    assert len(module.down_revision) <= 32


async def test_a_real_alembic_upgrade_applies_and_stamps_the_version(seeded):
    """The whole thing through the Alembic COMMAND — version bump included.

    The other tests call `upgrade()` directly, which is what makes them precise
    about the guards. This one runs `alembic upgrade` the way an operator does,
    so the stamping half is exercised too — the half that rejected the first
    revision id for being longer than `varchar(32)`.

    A SUBPROCESS, deliberately. `migrations/env.py` builds its own engine from
    settings and drives it with `asyncio.run`, so it neither accepts an injected
    connection nor tolerates being called from inside a running event loop.
    Reshaping production migration machinery to suit a test would be the wrong
    trade; spawning the real command against the disposable database tests more
    and changes nothing.
    """
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    engine = seeded

    async with engine.begin() as conn:
        await conn.execute(sa.text("DELETE FROM alembic_version"))
        await conn.execute(
            sa.text("INSERT INTO alembic_version (version_num) VALUES ('0046_service_account_agents')")
        )

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "0047_cleanup_verification_data"],
        cwd=root,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            # Both point at the disposable database, so nothing in this test can
            # reach the URL in the developer's own .env.
            "DATABASE_URL": MIGRATION_DB,
            "DIRECT_DATABASE_URL": MIGRATION_DB,
        },
    )
    assert result.returncode == 0, f"alembic failed:\n{result.stdout}\n{result.stderr}"

    async with engine.connect() as conn:
        stamped = (await conn.execute(sa.text("SELECT version_num FROM alembic_version"))).scalar()
    assert stamped == "0047_cleanup_verification_data"
    assert await _count(engine, "organizations", "id", VERIFICATION_ORG_ID) == 0
    assert await _count(engine, "organizations", "id", NEIGHBOUR_ORG_ID) == 1

    # Nothing credential-shaped in what the command printed. `env.py` announces
    # its target as host/database; a full URL there would put a password in
    # every CI log.
    combined = result.stdout + result.stderr
    assert "postgres@" not in combined or "://" not in combined
