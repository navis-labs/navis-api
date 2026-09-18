"""PostgreSQL integration tests for the Organization domain.

Runs ONLY when `NAVIS_TEST_DATABASE_URL` points at a disposable async Postgres. Proves
what unit fakes cannot: ORG-001 enum CHECK constraints reject invalid values, and ORG-002
bootstrap get-or-create is race-free under concurrency.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://... pytest tests/integration -q
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL to run Postgres integration tests",
)


@pytest.fixture()
async def engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401  register all models (incl. CHECK constraints)
    from navis.platform.database import Base

    eng = create_async_engine(os.environ["NAVIS_TEST_DATABASE_URL"])
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


async def _make_org(engine) -> uuid.UUID:
    from sqlalchemy.ext.asyncio import AsyncSession

    from navis.identity.models import Organization

    async with AsyncSession(engine, expire_on_commit=False) as s:
        org = Organization(name="Acme", slug=f"acme-{uuid.uuid4().hex[:6]}")
        s.add(org)
        await s.commit()
        return org.id


# ---- ORG-001: enum CHECK constraints ----

@pytest.mark.parametrize(
    "table,column,bad",
    [
        ("organization_lifecycles", "status", "not_a_status"),
        ("organization_lifecycles", "plan", "super_enterprise_plus"),
        ("organization_onboarding", "stage", "bogus_stage"),
    ],
)
async def test_invalid_enum_values_rejected(engine, table, column, bad):
    from sqlalchemy.ext.asyncio import AsyncSession

    org_id = await _make_org(engine)
    base_cols = "(organization_id, status, plan)" if table == "organization_lifecycles" else \
                "(organization_id, stage, signals_count, memory_count, stage_timestamps)"
    if table == "organization_lifecycles":
        vals = {"status": "trial", "plan": "free"}
        vals[column] = bad
        sql = f"INSERT INTO {table} {base_cols} VALUES (:o, :status, :plan)"
        params = {"o": str(org_id), **vals}
    else:
        sql = (f"INSERT INTO {table} {base_cols} "
               f"VALUES (:o, :stage, 0, 0, '{{}}'::jsonb)")
        params = {"o": str(org_id), "stage": bad}
    async with AsyncSession(engine, expire_on_commit=False) as s:
        with pytest.raises((IntegrityError, DBAPIError)):
            await s.execute(text(sql), params)
            await s.flush()


async def test_valid_enum_values_accepted(engine):
    from sqlalchemy.ext.asyncio import AsyncSession

    org_id = await _make_org(engine)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        await s.execute(
            text("INSERT INTO organization_lifecycles (organization_id, status, plan) "
                 "VALUES (:o, 'active', 'enterprise')"),
            {"o": str(org_id)},
        )
        await s.commit()
        got = (await s.execute(
            text("SELECT status, plan FROM organization_lifecycles WHERE organization_id=:o"),
            {"o": str(org_id)})).first()
    assert got == ("active", "enterprise")


# ---- ORG-002: concurrent bootstrap safety ----

@pytest.mark.parametrize("concurrency", [2, 5])
async def test_concurrent_ensure_is_race_free(engine, concurrency):
    from sqlalchemy.ext.asyncio import AsyncSession

    from navis.organization.repository import OrganizationRepository

    org_id = await _make_org(engine)

    async def ensure_both():
        async with AsyncSession(engine, expire_on_commit=False) as s:
            repo = OrganizationRepository(s)
            await repo.ensure_lifecycle(org_id)
            await repo.ensure_onboarding(org_id)
            await s.commit()

    # No IntegrityError despite N concurrent first-touch inserts.
    await asyncio.gather(*[ensure_both() for _ in range(concurrency)])

    async with AsyncSession(engine, expire_on_commit=False) as s:
        nl = (await s.execute(text("SELECT count(*) FROM organization_lifecycles WHERE organization_id=:o"), {"o": str(org_id)})).scalar()
        no = (await s.execute(text("SELECT count(*) FROM organization_onboarding WHERE organization_id=:o"), {"o": str(org_id)})).scalar()
        state = (await s.execute(text("SELECT status, plan, stage FROM organization_lifecycles l JOIN organization_onboarding o USING (organization_id) WHERE organization_id=:o"), {"o": str(org_id)})).first()
    assert nl == 1 and no == 1                 # exactly one row each — no duplicates
    assert state == ("trial", "free", "created")  # same final default state
