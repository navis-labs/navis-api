"""PostgreSQL proof that one inbound message can yield only one sendable reply.

Runs ONLY when `NAVIS_TEST_DATABASE_URL` points at a DISPOSABLE async Postgres —
the fixture drops every table on teardown, so pointing this at a real database
destroys it.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://... pytest tests/integration -q

Why this needs a real server rather than the in-memory fake:

  * the guarantee is a PARTIAL UNIQUE INDEX. A fake dict cannot prove that
    Postgres enforces it, and the whole point of moving the guarantee into the
    database is that application-level "if exists: return existing" loses a
    race.
  * the transition case involves a row written under the OLD idempotency scheme
    whose obligation_key was backfilled. Its primary key does not match the new
    deterministic id, so only a real INSERT ... ON CONFLICT can show that the
    second proposal collides on the obligation index and reuses it.

Nothing here sends mail or approves anything.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL to run reply-obligation Postgres tests",
)

THREAD = "19fdb57e89581d43"
MESSAGE = "<calendar-b7706463-b4ea-4492-b1aa-89c6dfe8@google.com>"
OTHER_MESSAGE = "<CACoxYfgQQVZ1K=2OQOXvYjZ6-ms9Gp4Fxowp=s9N@mail.gmail.com>"


@pytest.fixture()
async def engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401  register all models
    from navis.platform.database import Base

    eng = create_async_engine(os.environ["NAVIS_TEST_DATABASE_URL"])
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


async def _org(engine) -> uuid.UUID:
    from sqlalchemy import text as sql

    org_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            sql("INSERT INTO organizations (id, name, slug) VALUES (:i, :n, :s)"),
            {"i": org_id, "n": "obligation-test", "s": f"obl-{org_id.hex[:12]}"},
        )
    return org_id


def payload(**over) -> dict:
    base = {
        "to": ["mulayamabmrc@gmail.com"],
        "person_email": "mulayamabmrc@gmail.com",
        "thread_id": THREAD,
        "reply_thread_id": THREAD,
        "in_reply_to_message_id": MESSAGE,
        "subject": "Invitation: Meeting",
        "body": "",
        "requires_draft": True,
        "candidate_kind": "waiting_reply",
        "waiting_days": 2.3,
    }
    base.update(over)
    return base


async def _request(session, org_id, *, decision_id=None, watermark="w-1", **over):
    from navis.execution.models import ExecutionOrigin
    from navis.execution.service import ExecutionService

    svc = ExecutionService(session)
    return await svc.request(
        org_id,
        decision_id or uuid.uuid4(),
        "reply_to_email",
        payload(**over),
        reasoning_signal_ids=[],
        reasoning_version=2,
        state_watermark=watermark,
        origin=ExecutionOrigin.PROACTIVE,
    )


async def _count(engine, org_id) -> int:
    from sqlalchemy import text as sql

    async with engine.begin() as conn:
        return (
            await conn.execute(
                sql(
                    "SELECT count(*) FROM execution_requests "
                    "WHERE organization_id = :o AND action_type = 'reply_to_email'"
                ),
                {"o": org_id},
            )
        ).scalar_one()


async def test_three_briefs_one_message_one_execution(engine):
    """The production sequence, against a real database."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    org_id = await _org(engine)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    ids = []
    for watermark, days in (("2e482801", 2.3), ("e06b78b1", 2.3), ("e910a36d", 5.9)):
        async with Session() as s:
            req, _ = await _request(s, org_id, watermark=watermark, waiting_days=days)
            ids.append(req.id)
            await s.commit()

    assert len(set(ids)) == 1
    assert await _count(engine, org_id) == 1


async def test_a_new_inbound_message_creates_a_second_execution(engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    org_id = await _org(engine)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as s:
        first, _ = await _request(s, org_id)
        await s.commit()
    async with Session() as s:
        second, created = await _request(s, org_id, in_reply_to_message_id=OTHER_MESSAGE)
        await s.commit()

    assert first.id != second.id
    assert created is True
    assert await _count(engine, org_id) == 2


async def test_concurrent_proposals_resolve_to_one_execution(engine):
    """Two propose passes racing on the same obligation.

    This is why the guarantee lives in an index and not in a read-then-write:
    both sessions check an empty table, both attempt the insert, and Postgres
    has to be the thing that picks a winner.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    org_id = await _org(engine)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async def propose(watermark: str):
        async with Session() as s:
            req, created = await _request(s, org_id, watermark=watermark)
            await s.commit()
            return req.id, created

    results = await asyncio.gather(
        *[propose(f"w-{i}") for i in range(6)], return_exceptions=True
    )
    ok = [r for r in results if not isinstance(r, Exception)]

    assert ok, f"every concurrent proposal failed: {results}"
    assert len({rid for rid, _ in ok}) == 1, "concurrent proposals diverged"
    assert await _count(engine, org_id) == 1


async def test_the_database_refuses_a_second_row_for_one_obligation(engine):
    """Direct proof of the index, bypassing the service entirely."""
    from sqlalchemy import text as sql
    from sqlalchemy.exc import IntegrityError

    org_id = await _org(engine)
    obligation = f"reply|waiting_reply|{THREAD}|{MESSAGE.strip('<>')}"

    async def insert(row_id: uuid.UUID, idem: str):
        async with engine.begin() as conn:
            await conn.execute(
                sql(
                    "INSERT INTO execution_requests "
                    "(id, organization_id, decision_id, action_type, status, "
                    " idempotency_key, obligation_key, payload, provider_reference, "
                    " attempts, origin, reasoning_signal_ids) "
                    "VALUES (:id, :org, :dec, 'reply_to_email', 'requested', "
                    " :idem, :obl, '{}', '{}', 0, 'proactive', '[]')"
                ),
                {
                    "id": row_id,
                    "org": org_id,
                    "dec": uuid.uuid4(),
                    "idem": idem,
                    "obl": obligation,
                },
            )

    await insert(uuid.uuid4(), "key-one")
    with pytest.raises(IntegrityError):
        # Different primary key, different idempotency_key — the OLD scheme
        # would have allowed this happily. The obligation index must not.
        await insert(uuid.uuid4(), "key-two")


async def test_a_pre_fix_row_is_reused_not_duplicated(engine):
    """The migration case: a row written under the OLD key, then backfilled.

    Its id does NOT match the new deterministic id, so the guarantee has to come
    from the obligation index plus the repository's fallback lookup. Getting
    this wrong is what would have produced a duplicate on the first deploy.
    """
    from sqlalchemy import text as sql
    from sqlalchemy.ext.asyncio import async_sessionmaker

    org_id = await _org(engine)
    legacy_id = uuid.uuid4()
    obligation = f"reply|waiting_reply|{THREAD}|{MESSAGE.strip('<>')}"

    async with engine.begin() as conn:
        await conn.execute(
            sql(
                "INSERT INTO execution_requests "
                "(id, organization_id, decision_id, action_type, status, "
                " idempotency_key, obligation_key, payload, provider_reference, "
                " attempts, origin, reasoning_signal_ids) "
                "VALUES (:id, :org, :dec, 'reply_to_email', 'executed', "
                " 'legacy-payload-hash', :obl, :payload, "
                " '{\"provider\":\"gmail\",\"external_id\":\"19feb69c55c0a6f6\"}', "
                " 1, 'proactive', '[]')"
            ),
            {
                "id": legacy_id,
                "org": org_id,
                "dec": uuid.uuid4(),
                "obl": obligation,
                "payload": __import__("json").dumps(payload()),
            },
        )

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        req, created = await _request(s, org_id, watermark="brand-new")
        await s.commit()

    # The already-SENT row is returned. No second execution, and nothing that
    # could send that reply a second time.
    assert req.id == legacy_id
    assert created is False
    assert req.status == "executed"
    assert await _count(engine, org_id) == 1
