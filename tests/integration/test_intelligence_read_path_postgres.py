"""PostgreSQL proof that Intelligence serves only the CURRENT projection.

Runs ONLY when `NAVIS_TEST_DATABASE_URL` points at a DISPOSABLE async Postgres —
the fixture drops every table on teardown, so pointing this at a real database
destroys it.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://... pytest tests/integration -q

Why this needs a real server: `current_insights` is pure and fully covered
DB-free in tests/intelligence/test_intelligence_v2.py. What is NOT coverable
there is the SQL `get_insights` uses to narrow before calling it — DISTINCT ON,
the newest-watermark pick, and `NULLS LAST` — none of which a Python list
reproduces. That narrowing must stay an optimisation: it has to hand the pure
function a superset of what the function would have chosen from the whole table,
or the optimisation quietly becomes the semantics.
"""
from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL to run Intelligence read-path Postgres tests",
)


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


async def _store(engine, org_id, passes, learning_versions=(30, 31)):
    """Append the real projection rows for each (watermark, reasoning_version)."""
    from sqlalchemy.ext.asyncio import AsyncSession

    from navis.intelligence.models import IntelligenceInsight
    from navis.intelligence.service import project_values
    from tests.intelligence.fakes import make_view, rich_reality, rich_view

    rows: list[dict] = []
    for watermark, version in passes:
        rows += project_values(
            make_view(org_id=org_id, learning_version=0),
            rich_reality(org_id=org_id, state_watermark=watermark,
                         reasoning_version=version),
        )
    # `rich_view`, not `make_view` — the six Learning generators gate on decisive
    # samples, so an empty view stores no Learning rows and DISTINCT ON would
    # never be exercised.
    for learning_version in learning_versions:
        rows += project_values(rich_view(org_id=org_id, learning_version=learning_version))
    assert rows

    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add_all([IntelligenceInsight(**values) for values in rows])
        await s.commit()
    return rows


async def _read(engine, org_id, *, include_superseded=False):
    from sqlalchemy.ext.asyncio import AsyncSession

    from navis.intelligence.repository import IntelligenceRepository

    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await IntelligenceRepository(s).get_insights(
            org_id, include_superseded=include_superseded
        )


async def _adaptive_view(engine, org_id):
    from sqlalchemy.ext.asyncio import AsyncSession

    from navis.adaptive.repository import AdaptiveRepository

    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await AdaptiveRepository(s).build_view(org_id)


async def _decision_view(engine, org_id):
    from sqlalchemy.ext.asyncio import AsyncSession

    from navis.orchestration.brief import _current_intelligence

    async with AsyncSession(engine, expire_on_commit=False) as s:
        return await _current_intelligence(s, org_id)


# ── 1. the narrowing agrees with the rule ──────────────────────────────


async def test_sql_narrowing_agrees_with_the_pure_rule(engine):
    from navis.intelligence.repository import current_insights

    org = await _org(engine)
    stored = await _store(engine, org, [("wm-1", 3), ("wm-1", 4), ("wm-2", 4)])

    served = await _read(engine, org)
    everything = await _read(engine, org, include_superseded=True)

    assert [r.insight_id for r in served] == [
        r.insight_id for r in current_insights(everything)
    ], "the SQL narrowing changed the answer"
    assert len(everything) == len(stored)  # the read deletes nothing


# ── 2. one row per logical insight, deterministically ordered ──────────


async def test_only_the_current_projection_is_served(engine):
    org = await _org(engine)
    await _store(engine, org, [("wm-1", 3), ("wm-1", 4), ("wm-2", 4)])

    served = await _read(engine, org)
    keys = [(r.insight_type, r.key) for r in served]
    assert keys == sorted(keys), "the read order is not deterministic"
    assert len(keys) == len(set(keys)), f"a superseded projection is still served: {keys}"

    reality = [r for r in served if r.state_watermark is not None]
    assert reality and {r.state_watermark for r in reality} == {"wm-2"}
    learning = [r for r in served if r.state_watermark is None]
    assert learning and {r.learning_version for r in learning} == {31}


async def test_two_reasoning_versions_of_one_snapshot_resolve_to_the_newer(engine):
    """THE live defect: v3 and v4 folded the same snapshot, so their rows agree on
    every column and differ only in the version key hashed into `insight_id`."""
    from navis.intelligence.service import project_values
    from tests.intelligence.fakes import make_view, rich_reality

    org = await _org(engine)
    await _store(engine, org, [("wm-1", 3), ("wm-1", 4)], learning_versions=())

    current = {
        v["insight_id"]
        for v in project_values(
            make_view(org_id=org, learning_version=0),
            rich_reality(org_id=org, state_watermark="wm-1", reasoning_version=4),
        )
    }
    served = await _read(engine, org)
    assert served and all(r.insight_id in current for r in served)


# ── 3. isolation, history, and the empty case ──────────────────────────


async def test_organizations_never_share_a_projection(engine):
    org_a = await _org(engine)
    org_b = await _org(engine)
    await _store(engine, org_a, [("wm-a", 4)], learning_versions=())
    await _store(engine, org_b, [("wm-b", 3)], learning_versions=())

    served_a = await _read(engine, org_a)
    served_b = await _read(engine, org_b)
    assert served_a and served_b
    assert all(r.organization_id == org_a for r in served_a)
    assert all(r.organization_id == org_b for r in served_b)
    # B's older reasoning version is current FOR B; A's newer one does not reach it.
    assert {r.state_watermark for r in served_b} == {"wm-b"}
    assert {r.insight_id for r in served_a}.isdisjoint({r.insight_id for r in served_b})


async def test_superseded_rows_stay_stored_and_queryable(engine):
    org = await _org(engine)
    stored = await _store(engine, org, [("wm-1", 3), ("wm-1", 4)], learning_versions=())

    served = await _read(engine, org)
    everything = await _read(engine, org, include_superseded=True)
    assert len(everything) == len(stored) > len(served)
    assert {r.insight_id for r in served} <= {r.insight_id for r in everything}


async def test_an_org_with_one_projection_reads_back_everything(engine):
    org = await _org(engine)
    stored = await _store(engine, org, [("wm-1", 4)], learning_versions=(30,))

    served = await _read(engine, org)
    assert {r.insight_id for r in served} == {v["insight_id"] for v in stored}


async def test_an_org_with_no_insights_reads_back_empty(engine):
    org = await _org(engine)
    assert await _read(engine, org) == []
    assert await _read(engine, org, include_superseded=True) == []


# ── 4. Adaptive consumes the SAME projection ───────────────────────────
#
# AdaptiveRepository used to run `select(IntelligenceInsight)` of its own, which
# returned every projection the org had ever produced. Adaptive proposes POLICY
# from those rows, so a superseded conclusion could govern the next brief while
# GET /intelligence/insights served the corrected one — two Intelligence read
# semantics in one system.


async def test_adaptive_reads_the_same_current_projection_as_the_api(engine):
    org = await _org(engine)
    stored = await _store(engine, org, [("wm-1", 3), ("wm-1", 4), ("wm-2", 4)])

    served = await _read(engine, org)
    view = await _adaptive_view(engine, org)

    assert {i.insight_id for i in view.insights} == {str(r.insight_id) for r in served}
    assert len(stored) > len(served), "the fixture must actually contain superseded rows"


async def test_adaptive_never_sees_a_superseded_insight(engine):
    org = await _org(engine)
    await _store(engine, org, [("wm-1", 3), ("wm-1", 4), ("wm-2", 4)])

    served = {str(r.insight_id) for r in await _read(engine, org)}
    everything = {str(r.insight_id) for r in await _read(engine, org, include_superseded=True)}
    superseded = everything - served
    assert superseded, "the fixture must actually contain superseded rows"

    view = await _adaptive_view(engine, org)
    assert {i.insight_id for i in view.insights}.isdisjoint(superseded)


async def test_adaptive_view_never_crosses_organizations(engine):
    org_a = await _org(engine)
    org_b = await _org(engine)
    await _store(engine, org_a, [("wm-a", 4)], learning_versions=(30,))
    await _store(engine, org_b, [("wm-b", 3)], learning_versions=(31,))

    view_a = await _adaptive_view(engine, org_a)
    view_b = await _adaptive_view(engine, org_b)
    ids_a = {i.insight_id for i in view_a.insights}
    ids_b = {i.insight_id for i in view_b.insights}

    assert ids_a and ids_b and ids_a.isdisjoint(ids_b)
    assert ids_a == {str(r.insight_id) for r in await _read(engine, org_a)}
    assert ids_b == {str(r.insight_id) for r in await _read(engine, org_b)}
    assert view_a.org_id == org_a and view_b.org_id == org_b


async def test_adaptive_version_and_as_of_survive_the_narrower_read(engine):
    """Compatibility: Adaptive's two derived values must not move.

    `adaptive_version` is a max over `learning_version` and `as_of` a max over
    `created_at`. Both are monotone with the projection that produced them, so
    the newest rows carry the maxima and dropping superseded ones cannot lower
    either — but that is an argument, and this is the check.
    """
    org = await _org(engine)
    stored = await _store(engine, org, [("wm-1", 3), ("wm-1", 4), ("wm-2", 4)])

    view = await _adaptive_view(engine, org)
    assert view.adaptive_version == max(r["learning_version"] for r in stored)
    assert view.as_of == max(r["created_at"] for r in stored if r["created_at"])
    # ...and the rest of the view still comes from Learning, untouched.
    assert view.active_policy and "brief_max_items" in view.active_policy


# ── 5. Decision consumes the SAME projection ───────────────────────────
#
# Phase 4.2 gave Decision its second input. Orchestration reads the current
# insights through `IntelligenceRepository` and flattens them to primitives, so
# Decision ranks against what the organization currently knows about itself —
# never against a conclusion Intelligence has already corrected.


async def test_decision_receives_the_current_intelligence_projection(engine):
    org = await _org(engine)
    stored = await _store(engine, org, [("wm-1", 3), ("wm-1", 4), ("wm-2", 4)])

    served = await _read(engine, org)
    view = await _decision_view(engine, org)

    assert {f.insight_id for f in view.facts} == {str(r.insight_id) for r in served}
    assert len(stored) > len(served), "the fixture must actually contain superseded rows"
    # Same rule, same answer as Adaptive and as the HTTP surface.
    adaptive = await _adaptive_view(engine, org)
    assert {f.insight_id for f in view.facts} == {i.insight_id for i in adaptive.insights}


async def test_decision_never_receives_a_superseded_insight(engine):
    org = await _org(engine)
    await _store(engine, org, [("wm-1", 3), ("wm-1", 4), ("wm-2", 4)])

    served = {str(r.insight_id) for r in await _read(engine, org)}
    everything = {str(r.insight_id) for r in await _read(engine, org, include_superseded=True)}
    superseded = everything - served
    assert superseded, "the fixture must actually contain superseded rows"

    view = await _decision_view(engine, org)
    assert {f.insight_id for f in view.facts}.isdisjoint(superseded)


async def test_no_orm_object_reaches_decision(engine):
    """The boundary's whole purpose: primitives cross, rows do not."""
    from navis.intelligence.models import IntelligenceInsight

    org = await _org(engine)
    await _store(engine, org, [("wm-1", 4)], learning_versions=(30,))

    view = await _decision_view(engine, org)
    assert view.facts
    for fact in view.facts:
        assert not isinstance(fact, IntelligenceInsight)
        assert isinstance(fact.insight_id, str)
        assert isinstance(fact.evidence, dict)
        assert type(fact.evidence).__module__ == "builtins"  # not a JSONB proxy
        assert isinstance(fact.supporting_reasoning_signal_ids, tuple)


async def test_the_decision_view_never_crosses_organizations(engine):
    org_a = await _org(engine)
    org_b = await _org(engine)
    await _store(engine, org_a, [("wm-a", 4)], learning_versions=(30,))
    await _store(engine, org_b, [("wm-b", 3)], learning_versions=(31,))

    view_a = await _decision_view(engine, org_a)
    view_b = await _decision_view(engine, org_b)
    ids_a = {f.insight_id for f in view_a.facts}
    ids_b = {f.insight_id for f in view_b.facts}

    assert ids_a and ids_b and ids_a.isdisjoint(ids_b)
    assert view_a.org_id == org_a and view_b.org_id == org_b
    assert ids_a == {str(r.insight_id) for r in await _read(engine, org_a)}
    assert ids_b == {str(r.insight_id) for r in await _read(engine, org_b)}


async def test_an_org_with_no_insights_yields_an_empty_view(engine):
    """`decide` must behave identically for an org that knows nothing yet."""
    org = await _org(engine)
    view = await _decision_view(engine, org)
    assert view.facts == () and view.as_of is None and view.org_id == org
