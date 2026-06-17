"""PostgreSQL integration test for Memory V2: real append-only + replay determinism.

Runs only with NAVIS_TEST_DATABASE_URL. Proves the drop-and-replay guarantee
against a live database: replaying the same signals yields byte-identical
memory_ids and zero new rows on the second pass.
"""
from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL to run Memory Postgres integration tests",
)


@pytest.fixture()
async def db():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import navis.registry  # noqa: F401
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


async def _seed_signal(db):
    from navis.identity.models import Organization, User
    from navis.signals.models import SignalEventRecord

    org = Organization(name="Acme", slug=f"acme-{uuid.uuid4().hex[:6]}")
    user = User(email=f"u-{uuid.uuid4().hex[:6]}@acme.com")
    db.add_all([org, user])
    await db.flush()
    sig = SignalEventRecord(
        id=uuid.uuid4(),
        organization_id=org.id,
        user_id=user.id,
        integration_id=None,
        source="gmail",
        event_type="email.received",
        external_id="m1",
        external_version="1",
        occurred_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        payload={"from": "a@acme.com", "to": "b@vc.com", "thread_id": "t1",
                 "body_text": "I'll send the deck by Friday."},
    )
    db.add(sig)
    await db.flush()
    return org.id, sig.id


async def test_replay_is_deterministic_and_idempotent(db):
    from navis.memory.replay import MemoryReplayService

    org_id, _sig_id = await _seed_signal(db)
    svc = MemoryReplayService(db)

    first = await svc.replay_org(org_id)
    assert first["new"] > 0  # facts written

    second = await svc.replay_org(org_id)
    assert second["facts"] == first["facts"]
    assert second["new"] == 0  # replay produced identical ids → nothing new
