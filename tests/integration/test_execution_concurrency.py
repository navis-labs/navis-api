"""P0-2: real-PostgreSQL concurrency — N workers cannot execute one request twice.

Runs only with NAVIS_TEST_DATABASE_URL (real Postgres; FOR UPDATE SKIP LOCKED and
ON CONFLICT cannot be exercised against fakes). Proves: (a) concurrent idempotent
requests collapse to one row; (b) concurrent execute passes execute an APPROVED
request exactly once.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL to run execution concurrency tests",
)


@pytest.fixture()
async def engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401
    from navis.platform.database import Base

    eng = create_async_engine(os.environ["NAVIS_TEST_DATABASE_URL"])
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


async def _seed_org_user(session):
    from navis.identity.models import Organization, User

    org = Organization(name="Acme", slug=f"acme-{uuid.uuid4().hex[:6]}")
    user = User(email=f"u-{uuid.uuid4().hex[:6]}@acme.com")
    session.add_all([org, user])
    await session.flush()
    return org.id, user.id


async def test_concurrent_requests_collapse_to_one_row(engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from navis.execution.service import ExecutionService

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        org_id, _ = await _seed_org_user(s)
        await s.commit()

    decision = uuid.uuid4()
    payload = {"to": "a@x.com", "subject": "Hi"}

    async def one():
        async with Session() as s:
            req, created = await ExecutionService(s).request(
                org_id, decision, "send_email", payload,
                reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
            )
            await s.commit()
            return req.id, created

    results = await asyncio.gather(*[one() for _ in range(25)], return_exceptions=True)
    ids = {r[0] for r in results if isinstance(r, tuple)}
    created_count = sum(1 for r in results if isinstance(r, tuple) and r[1])
    assert len(ids) == 1            # one deterministic execution_id
    assert created_count == 1       # exactly one creator; rest idempotent no-ops


async def test_concurrent_workers_execute_once(engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from navis.execution.repository import ExecutionRepository
    from navis.execution.service import ExecutionService
    from navis.execution.status import ExecutionStatus
    from navis.execution.worker import execute_pass

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        org_id, user_id = await _seed_org_user(s)
        svc = ExecutionService(s)
        req, _ = await svc.request(org_id, uuid.uuid4(), "create_task", {"title": "x"},
                                   reasoning_signal_ids=[], reasoning_version=1, state_watermark="w")
        await svc.approve(req.id, user_id)
        await s.commit()

    async def worker():
        async with Session() as s:
            out = await execute_pass(ExecutionRepository(s))
            await s.commit()
            return out["executed"]

    totals = await asyncio.gather(*[worker() for _ in range(10)], return_exceptions=True)
    executed = sum(t for t in totals if isinstance(t, int))
    assert executed == 1  # exactly one worker executed it (SKIP LOCKED + status guard)

    async with Session() as s:
        row = await ExecutionRepository(s).get(req.id)
        assert row.status == ExecutionStatus.EXECUTED.value
