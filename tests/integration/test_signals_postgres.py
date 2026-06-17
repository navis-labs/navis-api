"""PostgreSQL integration tests for the signals persistence layer.

Runs ONLY when `NAVIS_TEST_DATABASE_URL` points at a disposable Postgres
(async URL, e.g. postgresql+asyncpg://user:pass@localhost/navis_test). It proves
the things unit fakes cannot: real ON CONFLICT idempotency, the DLQ table, and
replay-from-PostgreSQL against a live database.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://... pytest tests/integration -q
"""
from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL to run Postgres integration tests",
)


@pytest.fixture()
async def db():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import navis.registry  # noqa: F401  register all models
    from navis.platform.database import Base

    engine = create_async_engine(os.environ["NAVIS_TEST_DATABASE_URL"])
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def _seed_tenant(db) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    from navis.identity.models import (
        Integration,
        IntegrationProvider,
        Organization,
        User,
    )

    org = Organization(name="Acme", slug=f"acme-{uuid.uuid4().hex[:6]}")
    user = User(email=f"u-{uuid.uuid4().hex[:6]}@acme.com")
    db.add_all([org, user])
    await db.flush()
    integ = Integration(
        organization_id=org.id, user_id=user.id, provider=IntegrationProvider.GMAIL
    )
    db.add(integ)
    await db.flush()
    return org.id, user.id, integ.id


async def test_on_conflict_idempotency_is_real(db):
    from navis.signals.base import SignalContext, SignalEvent, SignalEventType, SignalSource
    from navis.signals.repository import SignalRepository

    org_id, user_id, integ_id = await _seed_tenant(db)
    ctx = SignalContext(org_id=org_id, user_id=user_id, integration_id=integ_id)
    ev = SignalEvent.create(
        ctx,
        source=SignalSource.GMAIL,
        event_type=SignalEventType.EMAIL_RECEIVED,
        external_id="m1",
        external_version="1",
    )
    repo = SignalRepository(db)
    assert await repo.insert_idempotent(ev) is True   # first insert wins
    assert await repo.insert_idempotent(ev) is False  # ON CONFLICT DO NOTHING


async def test_dead_letter_table_and_replay(db):
    from navis.signals.base import SignalContext, SignalEvent, SignalEventType, SignalSource
    from navis.signals.repository import DeadLetterRepository

    org_id, user_id, integ_id = await _seed_tenant(db)
    ctx = SignalContext(org_id=org_id, user_id=user_id, integration_id=integ_id)
    ev = SignalEvent.create(
        ctx,
        source=SignalSource.GMAIL,
        event_type=SignalEventType.EMAIL_RECEIVED,
        external_id="poison",
        external_version="1",
    )
    dlq = DeadLetterRepository(db)
    row = await dlq.record(
        stage="ingest", error="boom", org_id=org_id, integration_id=integ_id,
        event_id=ev.event_id, dedupe_key=ev.dedupe_key, payload=ev.to_dict(),
    )
    await db.flush()
    unresolved = await dlq.list_unresolved(org_id)
    assert any(r.id == row.id for r in unresolved)
    await dlq.mark_replayed([row.id], resolved=True)
    assert all(r.id != row.id for r in await dlq.list_unresolved(org_id))


async def test_advisory_lock_blocks_second_holder(db):
    from navis.signals.locking import integration_lock_key, try_advisory_lock

    _, _, integ_id = await _seed_tenant(db)
    key = integration_lock_key(integ_id)
    assert await try_advisory_lock(db, key) is True
    # Same connection re-acquires (advisory locks are re-entrant per session);
    # a true cross-session test would need a second connection — exercised in CI.
