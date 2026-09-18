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


# ── Disclosure is org-scoped IN THE QUERY (P0-3E) ───────────────────────────


async def test_disclosure_lookup_returns_no_rows_for_another_tenants_memory_id(db):
    """CROSS-ORGANIZATION MEMORY -> 0 ROWS, proved against real PostgreSQL.

    `disclosure_by_memory_id` is what tells the Developer boundary what a
    retrieved fact may say. `tests/orchestration/test_memory_minimization.py`
    proves the SQL carries the `organization_id` predicate and that the caller
    withholds when the lookup comes back empty; this proves the predicate
    actually excludes the other tenant's row when a database executes it —
    which no compiled-SQL assertion can.

    Both orgs get a REAL memory record with a REAL thread body in its payload,
    so a regression that dropped the filter would return Org B's snippet here
    and this would fail on the returned row rather than on a count.
    """
    from datetime import datetime, timezone

    from navis.memory.lineage import disclosure_by_memory_id
    from navis.memory.models import MemoryRecord

    org_a, sig_a = await _seed_signal(db)
    org_b, sig_b = await _seed_signal(db)

    now = datetime.now(timezone.utc)
    ids = {}
    for org_id, sig_id, snippet in (
        (org_a, sig_a, "Org A: the bridge closes Friday."),
        (org_b, sig_b, "Org B: we are being acquired, do not forward."),
    ):
        memory_id = uuid.uuid4()
        ids[org_id] = memory_id
        db.add(
            MemoryRecord(
                memory_id=memory_id,
                organization_id=org_id,
                entity_type="thread",
                entity_id=uuid.uuid4(),
                source_signal_event_id=sig_id,
                source_integration_id=None,
                extractor_name="email_thread",
                extractor_version="1",
                schema_version=1,
                event_timestamp=now,
                dedup_key=f"thread:{memory_id}",
                payload={"subject": "Bridge", "participants": ["a@acme.com"],
                         "snippet": snippet},
                confidence=1.0,
            )
        )
    await db.flush()

    # Org A asks about BOTH ids — the shape a regression would take: a caller
    # holding one credential and an id from somewhere else.
    got = await disclosure_by_memory_id(db, org_a, list(ids.values()))

    assert set(got) == {ids[org_a]}, "the lookup crossed a tenant boundary"
    assert ids[org_b] not in got
    assert "acquired" not in str(got), "another tenant's snippet reached the caller"
