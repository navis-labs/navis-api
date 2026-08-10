"""PostgreSQL proof that `latest_brief` never returns an undated brief first.

Runs ONLY when `NAVIS_TEST_DATABASE_URL` points at a DISPOSABLE async Postgres —
the fixture drops every table on teardown, so pointing this at a real database
destroys it.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://... pytest tests/integration -q

Why this needs a real server: the whole defect is Postgres's own rule that `DESC`
sorts NULLs FIRST. A fake repository sorting a Python list reproduces whatever
the test author assumes, which is exactly the assumption that was wrong.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL to run brief-ordering Postgres tests",
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)


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
    from sqlalchemy.ext.asyncio import AsyncSession

    from navis.identity.models import Organization

    async with AsyncSession(engine, expire_on_commit=False) as s:
        org = Organization(name="Acme", slug=f"acme-{uuid.uuid4().hex[:6]}")
        s.add(org)
        await s.commit()
        return org.id


async def _add_brief(engine, org_id, *, as_of, decisions):
    from sqlalchemy.ext.asyncio import AsyncSession

    from navis.decisions.models import DailyBrief

    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            DailyBrief(
                id=uuid.uuid4(),
                organization_id=org_id,
                reasoning_version=2,
                state_watermark=None,
                as_of=as_of,
                decisions=decisions,
            )
        )
        await s.commit()


async def _latest(engine, org_id):
    from sqlalchemy.ext.asyncio import AsyncSession

    from navis.decisions.service import DecisionService

    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await DecisionService(s).latest_brief(org_id)


# ── 1. the actual production bug ───────────────────────────────────────


async def test_undated_empty_brief_does_not_beat_a_dated_brief(engine):
    """THE regression. The undated onboarding brief is written FIRST and has no
    decisions; a real brief follows. Before NULLS LAST the empty one won."""
    org = await _org(engine)
    await _add_brief(engine, org, as_of=None, decisions=[])
    await _add_brief(engine, org, as_of=NOW, decisions=[{"category": "waiting_reply"}])

    got = await _latest(engine, org)
    assert got is not None
    assert got.as_of is not None, "the undated brief won — NULLS LAST is missing"
    assert len(got.decisions) == 1


# ── 2. the newest dated brief is the one returned ──────────────────────


async def test_returns_the_latest_dated_brief(engine):
    org = await _org(engine)
    await _add_brief(engine, org, as_of=NOW - timedelta(days=2), decisions=[{"n": 1}])
    await _add_brief(engine, org, as_of=NOW, decisions=[{"n": 2}])

    got = await _latest(engine, org)
    assert got.as_of == NOW
    assert got.decisions == [{"n": 2}]


# ── 3. many dated briefs — newest still wins, undated present or not ───


async def test_newest_dated_wins_among_several_including_an_undated_one(engine):
    org = await _org(engine)
    await _add_brief(engine, org, as_of=None, decisions=[])
    await _add_brief(engine, org, as_of=NOW - timedelta(days=3), decisions=[{"n": 1}])
    await _add_brief(engine, org, as_of=NOW, decisions=[{"n": 3}])
    await _add_brief(engine, org, as_of=NOW - timedelta(days=1), decisions=[{"n": 2}])

    got = await _latest(engine, org)
    assert got.as_of == NOW
    assert got.decisions == [{"n": 3}]


# ── 4. shape and existing behaviour unchanged ──────────────────────────


async def test_response_shape_is_unchanged(engine):
    """Still a DailyBrief ORM row with the same fields — the fix touches
    ORDER BY only, so `_serialize` upstream keeps working untouched."""
    from navis.decisions.models import DailyBrief

    org = await _org(engine)
    await _add_brief(engine, org, as_of=NOW, decisions=[{"category": "priority"}])

    got = await _latest(engine, org)
    assert isinstance(got, DailyBrief)
    for field in (
        "id",
        "organization_id",
        "reasoning_version",
        "state_watermark",
        "as_of",
        "decisions",
        "created_at",
    ):
        assert hasattr(got, field), f"DailyBrief lost `{field}`"
    assert got.reasoning_version == 2
    assert got.decisions == [{"category": "priority"}]


async def test_no_briefs_at_all_still_returns_none(engine):
    org = await _org(engine)
    assert await _latest(engine, org) is None


async def test_an_undated_brief_is_still_returned_when_it_is_the_only_one(engine):
    """NULLS LAST demotes the undated brief; it must not DISCARD it. An org whose
    only brief predates its first sync still gets that brief rather than None."""
    org = await _org(engine)
    await _add_brief(engine, org, as_of=None, decisions=[])

    got = await _latest(engine, org)
    assert got is not None
    assert got.as_of is None
