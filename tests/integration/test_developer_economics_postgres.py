"""The Agent Economics rollup, against real PostgreSQL.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://.../navis_test \
        pytest tests/integration/test_developer_economics_postgres.py -q

WHAT THIS PROVES THAT THE UNIT TESTS CANNOT. `EconomicsRollupOut.of` is pure
and is tested on plain objects in `tests/developer/test_economics_rollup.py`.
Everything BELOW the aggregate is a property of one SQL statement — the tenant
filter on three tables, the two outer joins, the per-run reduction, and the
`EXISTS` used for the model filter. A mistake in any of those produces a
perfectly well-formed aggregate over the wrong rows, which no pure test can
see.

The isolation assertions are the reason this module exists: a cost rollup that
leaks across tenants leaks how much another organization spends and on what.
"""
from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("NAVIS_TEST_DATABASE_URL"),
        reason="set NAVIS_TEST_DATABASE_URL (disposable Postgres) to run these",
    ),
    pytest.mark.asyncio(loop_scope="module"),
]

TENANT_A = uuid.UUID("eeee0000-0000-4000-8000-0000000000a1")
TENANT_B = uuid.UUID("eeee0000-0000-4000-8000-0000000000b1")
AGENT_A1 = uuid.UUID("eeee0000-0000-4000-8000-0000000000a2")
AGENT_A2 = uuid.UUID("eeee0000-0000-4000-8000-0000000000a3")
AGENT_B1 = uuid.UUID("eeee0000-0000-4000-8000-0000000000b2")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401  register every model
    from navis.platform.database import Base

    url = os.environ["NAVIS_TEST_DATABASE_URL"]
    name = url.rsplit("/", 1)[-1].split("?")[0]
    assert name.startswith("navis_test"), (
        f"refusing to run against database {name!r}: point "
        "NAVIS_TEST_DATABASE_URL at a disposable database named navis_test*"
    )

    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def seeded(engine):
    """Two tenants with deliberately similar ledgers.

    Tenant B is given the SAME task descriptions and the same costs as tenant A
    so an isolation bug cannot hide behind a difference in the numbers: if the
    filter were missing, the totals would simply double, which is exactly the
    failure that reads as plausible.

    Per-module row cleanup, never `drop_all` — the database outlives the run,
    and dropping re-oids every PG ENUM and breaks pooled asyncpg codecs.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.developer.run_models import AgentRun, AgentRunOutcome, AgentRunStep, AgentTask
    from navis.identity.enterprise_models import ServiceAccount
    from navis.identity.models import Organization

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM organizations WHERE id = ANY(:ids)").bindparams(
                ids=[TENANT_A, TENANT_B]
            )
        )

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        for org_id, agents in ((TENANT_A, (AGENT_A1, AGENT_A2)), (TENANT_B, (AGENT_B1,))):
            # Slug from the TAIL: both tenant ids deliberately share a prefix
            # so the isolation assertions cannot pass by accident of ordering,
            # and a head-slice collides on the unique constraint.
            s.add(Organization(id=org_id, name=f"Org {org_id.hex[-4:]}",
                               slug=f"econ-{org_id.hex[-8:]}"))
            await s.flush()
            for a in agents:
                s.add(ServiceAccount(id=a, organization_id=org_id,
                                     name=f"agent-{a.hex[:4]}"))
        await s.flush()

        seq = iter(range(1, 10_000))

        def add_run(*, org, agent, task, fetched, version, cost, basis, outcome,
                    currency="USD", model="claude-opus-5", uncosted=False):
            run_id = uuid.uuid4()
            s.add(AgentRun(id=run_id, organization_id=org, task_id=task, agent_id=agent,
                           status="completed", context_version=version,
                           completed_at=__import__("datetime").datetime.now(
                               __import__("datetime").timezone.utc)))
            if fetched:
                s.add(AgentRunStep(
                    id=uuid.uuid4(), organization_id=org, run_id=run_id,
                    sequence=next(seq), step_type="context_fetch", status="completed",
                    label="ctx", context_items=3, context_tokens=500, latency_ms=100,
                ))
            s.add(AgentRunStep(
                id=uuid.uuid4(), organization_id=org, run_id=run_id,
                sequence=next(seq), step_type="model_generation", status="completed",
                label="gen", model_name=model, model_provider="anthropic",
                input_tokens=100, output_tokens=50, latency_ms=900,
                cost_micros=None if uncosted else cost,
                cost_basis=None if uncosted else basis,
                cost_currency=None if uncosted else currency,
                pricing_version=None if uncosted else "2026-09-01",
            ))
            if outcome is not None:
                s.add(AgentRunOutcome(id=uuid.uuid4(), organization_id=org, run_id=run_id,
                                      agent_id=agent, status=outcome, summary="x"))
            return run_id

        # ── Tenant A, task 1: with context, 2 runs, both succeed ────────
        task_a1 = uuid.uuid4()
        s.add(AgentTask(id=task_a1, organization_id=TENANT_A, agent_id=AGENT_A1,
                        description="ship the billing fix", status="open"))
        # ── Tenant A, task 2: WITHOUT context, 3 attempts, 1 success ────
        task_a2 = uuid.uuid4()
        s.add(AgentTask(id=task_a2, organization_id=TENANT_A, agent_id=AGENT_A1,
                        description="ship the billing fix", status="open"))
        # ── Tenant A, task 3: second agent, for agent isolation ─────────
        task_a3 = uuid.uuid4()
        s.add(AgentTask(id=task_a3, organization_id=TENANT_A, agent_id=AGENT_A2,
                        description="other agent work", status="open"))
        # ── Tenant B: identical shape, must never appear in A ───────────
        task_b1 = uuid.uuid4()
        s.add(AgentTask(id=task_b1, organization_id=TENANT_B, agent_id=AGENT_B1,
                        description="ship the billing fix", status="open"))
        await s.flush()

        for _ in range(2):
            add_run(org=TENANT_A, agent=AGENT_A1, task=task_a1, fetched=True, version="v1",
                    cost=1_000, basis="estimated", outcome="success")
        for outcome in ("failure", "failure", "success"):
            add_run(org=TENANT_A, agent=AGENT_A1, task=task_a2, fetched=False, version=None,
                    cost=1_000, basis="estimated", outcome=outcome)
        add_run(org=TENANT_A, agent=AGENT_A2, task=task_a3, fetched=True, version="v2",
                cost=500, basis="provider_reported", outcome="success")
        for _ in range(4):
            add_run(org=TENANT_B, agent=AGENT_B1, task=task_b1, fetched=True, version="v1",
                    cost=1_000, basis="estimated", outcome="success")

        await s.commit()

    yield {"a1": task_a1, "a2": task_a2, "a3": task_a3, "b1": task_b1}

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM organizations WHERE id = ANY(:ids)").bindparams(
                ids=[TENANT_A, TENANT_B]
            )
        )


async def rollup(engine, org, **kw):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.developer.run_schemas import EconomicsRollupOut
    from navis.developer.run_service import AgentLifecycleService

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        rows = await AgentLifecycleService(s).list_run_economics(org, **kw)
    return EconomicsRollupOut.of(rows)


# ── Tenancy: the assertions this module exists for ──────────────────────────


async def test_one_tenants_rollup_never_contains_anothers_runs(engine, seeded):
    """Both tenants hold a task with the SAME description and the same costs,
    so a missing filter shows up as doubled totals rather than as an obviously
    foreign row."""
    a = await rollup(engine, TENANT_A)
    b = await rollup(engine, TENANT_B)

    assert {g.task_id for g in a.groups} == {seeded["a1"], seeded["a2"], seeded["a3"]}
    assert {g.task_id for g in b.groups} == {seeded["b1"]}
    assert a.runs_considered == 6
    assert b.runs_considered == 4

    b_group = b.groups[0]
    assert b_group.runs == 4, "tenant B must see its own four and only its own"
    assert b_group.estimated_cost_micros == 4_000


async def test_an_unknown_organization_sees_nothing(engine, seeded):
    empty = await rollup(engine, uuid.uuid4())
    assert empty.groups == []
    assert empty.runs_considered == 0


async def test_filtering_by_another_tenants_task_returns_nothing(engine, seeded):
    """A cross-tenant id is not an error, it is simply absent — the same answer
    an id that exists nowhere gets."""
    leaked = await rollup(engine, TENANT_A, task_id=seeded["b1"])
    assert leaked.groups == []


async def test_agent_isolation(engine, seeded):
    one = await rollup(engine, TENANT_A, agent_id=AGENT_A2)
    assert {g.task_id for g in one.groups} == {seeded["a3"]}
    assert one.runs_considered == 1


# ── The metric, over real rows ──────────────────────────────────────────────


async def test_context_fetched_and_not_fetched_separate_and_the_cost_differs(engine, seeded):
    """The comparison the rollup exists for, end to end through SQL."""
    a = await rollup(engine, TENANT_A)
    with_ctx = next(g for g in a.groups if g.task_id == seeded["a1"])
    without = next(g for g in a.groups if g.task_id == seeded["a2"])

    assert with_ctx.context_fetched is True
    assert with_ctx.context_version == "v1"
    assert with_ctx.runs == 2
    assert with_ctx.successful_outcomes == 2
    assert with_ctx.success_rate == 1.0
    assert with_ctx.expected_estimated_cost_per_success_micros == 1_000

    assert without.context_fetched is False
    assert without.context_version is None
    assert without.runs == 3
    assert without.successful_outcomes == 1
    # THREE attempts bought ONE success.
    assert without.estimated_cost_micros == 3_000
    assert without.expected_estimated_cost_per_success_micros == 3_000


async def test_retries_are_attempts_not_a_counter(engine, seeded):
    """A retry is another RUN of the same task. The rollup must see three runs
    under one task without anything incrementing a retry field."""
    a = await rollup(engine, TENANT_A, task_id=seeded["a2"])
    assert len(a.groups) == 1
    assert a.groups[0].runs == 3


async def test_the_two_cost_bases_stay_apart_through_sql(engine, seeded):
    a = await rollup(engine, TENANT_A, task_id=seeded["a3"])
    g = a.groups[0]
    assert g.provider_reported_cost_micros == 500
    assert g.estimated_cost_micros is None, "a reported figure must not become an estimate"


async def test_context_version_separates_and_filters(engine, seeded):
    v2 = await rollup(engine, TENANT_A, context_version="v2")
    assert {g.context_version for g in v2.groups} == {"v2"}
    assert {g.task_id for g in v2.groups} == {seeded["a3"]}


async def test_model_filter_keeps_the_whole_run(engine, seeded):
    """Filtering by model must select RUNS, not steps: predicating the join
    would drop the run's other steps and report a fraction of its own cost."""
    hit = await rollup(engine, TENANT_A, task_id=seeded["a1"], model_name="claude-opus-5")
    assert hit.groups[0].runs == 2
    assert hit.groups[0].estimated_cost_micros == 2_000

    miss = await rollup(engine, TENANT_A, model_name="gpt-5")
    assert miss.groups == []


async def test_one_outcome_per_run_is_structural(engine, seeded):
    """`uq_agent_run_outcomes_run` makes a duplicate outcome unrepresentable, so
    the join cannot fan out and inflate the denominator. Asserted rather than
    assumed, because the aggregate would be silently wrong if it ever changed."""
    from sqlalchemy import text

    async with engine.begin() as conn:
        dupes = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM (SELECT run_id FROM agent_run_outcomes "
                    "GROUP BY run_id HAVING count(*) > 1) d"
                )
            )
        ).scalar()
    assert dupes == 0

    a = await rollup(engine, TENANT_A, task_id=seeded["a1"])
    assert a.groups[0].runs == 2, "two runs, two outcomes, no fan-out"


async def test_uncosted_generations_survive_the_reduction(engine, seeded):
    """A model step with no cost must count as uncosted, not as zero — proved
    against a real row rather than a constructed object."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.developer.run_models import AgentRun, AgentRunStep, AgentTask

    maker = async_sessionmaker(engine, expire_on_commit=False)
    task = uuid.uuid4()
    async with maker() as s:
        s.add(AgentTask(id=task, organization_id=TENANT_A, agent_id=AGENT_A1,
                        description="uncosted work", status="open"))
        await s.flush()  # the run's FK needs the task to exist first
        run_id = uuid.uuid4()
        s.add(AgentRun(id=run_id, organization_id=TENANT_A, task_id=task,
                       agent_id=AGENT_A1, status="running"))
        s.add(AgentRunStep(id=uuid.uuid4(), organization_id=TENANT_A, run_id=run_id,
                           sequence=1, step_type="model_generation",
                           status="completed", label="gen", model_name="claude-opus-5",
                           cost_micros=None))
        await s.commit()

    g = (await rollup(engine, TENANT_A, task_id=task)).groups[0]
    assert g.uncosted_generations == 1
    assert g.runs_with_no_cost_recorded == 1
    assert g.estimated_cost_micros is None
    assert g.unknown_outcomes == 1, "no outcome recorded is not a failure"
    assert g.success_rate is None
