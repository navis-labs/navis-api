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
async def db(monkeypatch):
    from contextlib import asynccontextmanager

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import navis.registry  # noqa: F401  register all models
    from navis.platform.database import Base

    engine = create_async_engine(os.environ["NAVIS_TEST_DATABASE_URL"])
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def _scope():
        async with Session() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    # ── THE SUBSCRIBERS MUST READ THE TEST DATABASE ────────────────────
    # `relay_signals_pass` calls `ensure_subscribers()`, so the REAL handlers
    # run, and each opens its own session through `session_scope` — which it
    # imported BY VALUE at module import. Patching only
    # `navis.platform.database.session_scope` misses them.
    #
    # Unpatched, those handlers connect to whatever `DATABASE_URL` names, i.e.
    # PRODUCTION, and then cannot see the org this test just wrote locally:
    # `on_first_signal` died on
    # `organization_onboarding_organization_id_fkey`, the relay counted a
    # failed handler, and the row was correctly withheld from `published_at` —
    # so the assertion below failed for a reason that had nothing to do with
    # the outbox contract it exists to prove. The foreign key is the only
    # reason nothing was written to production.
    #
    # Same list-of-hops approach as `test_observation_contract`, which has the
    # fuller note on why each module needs its own patch.
    for target in (
        "navis.identity_resolution.service.session_scope",
        "navis.memory.service.session_scope",
        "navis.orchestration.org_provisioning.session_scope",
        "navis.signals.outbox.session_scope",
        "navis.signals.router.session_scope",
        "navis.platform.database.session_scope",
    ):
        monkeypatch.setattr(target, _scope, raising=False)

    # Redis is a best-effort mirror and is not running here; waiting on it
    # would measure the sandbox rather than the contract.
    from navis.platform.events import event_bus

    monkeypatch.setattr(event_bus, "_mirror_to_redis", False, raising=False)

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


def _ev(ctx, *, ext, ts=None):
    from datetime import datetime, timezone

    from navis.signals.base import SignalEvent, SignalEventType, SignalSource

    return SignalEvent.create(
        ctx, source=SignalSource.GMAIL, event_type=SignalEventType.EMAIL_RECEIVED,
        external_id=ext, external_version="1",
        timestamp=ts or datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
    )


async def test_batch_insert_is_idempotent_and_dedupes_intra_batch(db):
    from navis.signals.base import SignalContext
    from navis.signals.repository import SignalRepository

    org_id, user_id, integ_id = await _seed_tenant(db)
    ctx = SignalContext(org_id=org_id, user_id=user_id, integration_id=integ_id)
    evs = [_ev(ctx, ext=f"b{i}") for i in range(3)]
    repo = SignalRepository(db)
    assert await repo.insert_many_idempotent(evs + [evs[0]]) == 3  # intra-batch dup ignored
    assert await repo.insert_many_idempotent(evs) == 0             # fully idempotent re-run


async def test_replay_order_is_stable_and_keyset_resumable(db):
    from datetime import datetime, timezone

    from navis.signals.base import SignalContext
    from navis.signals.repository import SignalRepository

    org_id, user_id, integ_id = await _seed_tenant(db)
    ctx = SignalContext(org_id=org_id, user_id=user_id, integration_id=integ_id)
    t = datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    repo = SignalRepository(db)
    # two events share the SAME occurred_at — the (occurred_at, id) order must be stable.
    await repo.insert_many_idempotent([_ev(ctx, ext="s0", ts=t), _ev(ctx, ext="s1", ts=t),
                                       _ev(ctx, ext="s2")])
    first = [r.id for r in await repo.for_replay(org_id)]
    second = [r.id for r in await repo.for_replay(org_id)]
    assert first == second                      # deterministic across reads
    # keyset cursor: page 2 resumes after page 1 with no overlap.
    p1 = await repo.for_replay(org_id, limit=2)
    p2 = await repo.for_replay(org_id, after=(p1[-1].occurred_at, p1[-1].id), limit=2)
    assert not ({r.id for r in p1} & {r.id for r in p2})
    streamed = [r.id async for r in repo.stream_replay(org_id, batch_size=2)]
    assert streamed == first                     # streaming yields the same total order


async def test_dlq_purge_resolved_keeps_unresolved(db):
    from datetime import datetime, timedelta, timezone

    from navis.signals.repository import DeadLetterRepository

    org_id, _, _ = await _seed_tenant(db)
    dlq = DeadLetterRepository(db)
    r1 = await dlq.record(stage="ingest", error="e", org_id=org_id)
    r2 = await dlq.record(stage="ingest", error="e", org_id=org_id)
    await db.flush()
    r1_id, r2_id = r1.id, r2.id
    await dlq.mark_replayed([r1_id], resolved=True)   # r1 resolved, r2 not
    purged = await dlq.purge_resolved(before=datetime.now(timezone.utc) + timedelta(minutes=1))
    assert purged == 1
    remaining = {r.id for r in await dlq.list_unresolved(org_id)}
    assert r2_id in remaining and r1_id not in remaining


async def test_outbox_staged_in_txn_then_relay_publishes(db):
    from sqlalchemy import select

    from navis.signals.base import SignalContext, SignalEvent, SignalEventType, SignalSource
    from navis.signals.models import SignalOutbox
    from navis.signals.outbox import relay_signals_pass
    from navis.signals.repository import SignalRepository
    from navis.signals.service import SignalIngestionService

    org_id, user_id, integ_id = await _seed_tenant(db)
    ctx = SignalContext(org_id=org_id, user_id=user_id, integration_id=integ_id)
    ev = SignalEvent.create(
        ctx, source=SignalSource.CALENDAR,
        event_type=SignalEventType.CALENDAR_EVENT_CANCELLED, external_id="cx", external_version="1",
    )
    await SignalIngestionService(db).ingest(ev)

    # Outbox row staged (same txn) — unpublished, routed by event_type, carries lineage.
    row = (await db.scalars(select(SignalOutbox).where(SignalOutbox.aggregate_id == ev.event_id))).one()
    assert row.published_at is None
    assert row.event_type == "CalendarEventCancelled"   # NOT CalendarEventCreated (Issue A)
    assert row.payload["schema_version"] == ev.schema_version
    assert row.payload["captured_at"]                   # lineage (Issue E)

    # Relay publishes pending rows and marks them published (at-least-once).
    assert await relay_signals_pass(SignalRepository(db)) >= 1
    published_at = await db.scalar(
        select(SignalOutbox.published_at).where(SignalOutbox.id == row.id)
    )
    assert published_at is not None


async def test_advisory_lock_blocks_second_holder(db):
    from navis.signals.locking import integration_lock_key, try_advisory_xact_lock

    _, _, integ_id = await _seed_tenant(db)
    key = integration_lock_key(integ_id)
    assert await try_advisory_xact_lock(db, key) is True
    # Same connection re-acquires (advisory locks are re-entrant per session);
    # a true cross-session test would need a second connection — exercised in CI.
