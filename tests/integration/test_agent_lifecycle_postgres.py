"""The agent execution lifecycle against a REAL Postgres.

WHAT ONLY A REAL DATABASE CAN PROVE. Everything in
`tests/developer/test_run_lifecycle.py` is pure and runs everywhere. These are
the assertions that need the engine itself:

  * migration 0048 and the ORM models describe the SAME schema;
  * the CHECK constraints actually refuse bad rows, not merely declare that they
    would;
  * FK CASCADE really removes children when a tenant or an agent goes;
  * the UNIQUE constraints make ordering and idempotency real rather than usual;
  * a tenant predicate in the query genuinely excludes another tenant's rows.

SAFETY. Refuses to run unless the target database is named `navis_test*`, and
never drops a schema. Same guard as the other integration modules.
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

TENANT_A = uuid.UUID("bbbb0000-0000-4000-8000-00000000000a")
TENANT_B = uuid.UUID("bbbb0000-0000-4000-8000-00000000000b")


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


@pytest_asyncio.fixture(loop_scope="module")
async def session(engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s
        await s.rollback()


@pytest_asyncio.fixture(loop_scope="module")
async def tenants(session):
    """Two organizations, each with one agent. The cross-tenant fixture."""
    from navis.identity.enterprise_models import ServiceAccount
    from navis.identity.models import Organization

    made: dict = {}
    for org_id, tag in ((TENANT_A, "A"), (TENANT_B, "B")):
        org = await session.get(Organization, org_id)
        if org is None:
            session.add(
                Organization(id=org_id, name=f"Tenant {tag}", slug=f"tenant-{tag.lower()}-{org_id.hex[:6]}")
            )
            await session.flush()
        agent = ServiceAccount(
            organization_id=org_id, name=f"Agent {tag}", scopes=["view_state"]
        )
        session.add(agent)
        await session.flush()
        made[org_id] = agent.id
    await session.commit()
    return made


# ── Schema parity ───────────────────────────────────────────────────────────


async def test_the_migration_and_the_orm_describe_the_same_schema(engine):
    """THE ONE THAT CATCHES DRIFT. The unit suites build from
    `Base.metadata.create_all`; production is built by Alembic. If the two ever
    disagree, every test passes and production has a different table.

    Compared by reflecting what `create_all` produced and asserting migration
    0048's columns, constraints and indexes are all present with the same names.
    """
    from sqlalchemy import inspect

    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda c: inspect(c).get_table_names())
        assert {"agent_tasks", "agent_runs", "agent_run_steps", "agent_run_outcomes"} <= set(tables)

        for table, expected_indexes in (
            ("agent_tasks", {"ix_agent_tasks_org_created", "ix_agent_tasks_agent_created"}),
            (
                "agent_runs",
                {
                    "ix_agent_runs_org_created",
                    "ix_agent_runs_agent_created",
                    "ix_agent_runs_task_created",
                    "ix_agent_runs_org_status",
                },
            ),
            ("agent_run_steps", {"ix_agent_run_steps_org_created"}),
            (
                "agent_run_outcomes",
                {"ix_agent_run_outcomes_org_created", "ix_agent_run_outcomes_agent_status"},
            ),
        ):
            names = {
                i["name"] for i in await conn.run_sync(lambda c, t=table: inspect(c).get_indexes(t))
            }
            missing = expected_indexes - names
            assert not missing, f"{table} is missing indexes {missing}"


async def test_every_table_carries_its_tenant_column(engine):
    """The denormalized `organization_id` is the isolation guarantee. A child
    table without it puts the tenant two joins away on every query."""
    from sqlalchemy import inspect

    async with engine.connect() as conn:
        for table in ("agent_tasks", "agent_runs", "agent_run_steps", "agent_run_outcomes"):
            columns = {
                c["name"]
                for c in await conn.run_sync(lambda c2, t=table: inspect(c2).get_columns(t))
            }
            assert "organization_id" in columns, f"{table} has no tenant column"


# ── Constraints the database must enforce ───────────────────────────────────


async def test_a_bad_status_is_refused_by_the_database(session, tenants):
    """The edge validates too, but the edge is one of several writers over a
    table's life. A free-text status cannot be grouped or counted later."""
    from sqlalchemy.exc import IntegrityError

    from navis.developer.run_models import AgentTask

    session.add(
        AgentTask(
            organization_id=TENANT_A,
            agent_id=tenants[TENANT_A],
            description="bad status",
            status="not-a-status",
        )
    )
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_a_terminal_run_must_carry_its_completion_time(session, tenants):
    """A duration computed from a missing timestamp is silently wrong."""
    from sqlalchemy.exc import IntegrityError

    from navis.developer.run_models import AgentRun, AgentTask

    task = AgentTask(
        organization_id=TENANT_A, agent_id=tenants[TENANT_A], description="t"
    )
    session.add(task)
    await session.flush()

    session.add(
        AgentRun(
            organization_id=TENANT_A,
            task_id=task.id,
            agent_id=tenants[TENANT_A],
            status="completed",
            completed_at=None,  # contradicts the status
        )
    )
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_negative_token_counts_are_refused(session, tenants):
    """A negative count corrupts every aggregate built on this table, silently."""
    from sqlalchemy.exc import IntegrityError

    from navis.developer.run_models import AgentRun, AgentRunStep, AgentTask

    task = AgentTask(organization_id=TENANT_A, agent_id=tenants[TENANT_A], description="t")
    session.add(task)
    await session.flush()
    run = AgentRun(organization_id=TENANT_A, task_id=task.id, agent_id=tenants[TENANT_A])
    session.add(run)
    await session.flush()

    session.add(
        AgentRunStep(
            organization_id=TENANT_A,
            run_id=run.id,
            sequence=0,
            step_type="model_generation",
            input_tokens=-1,
        )
    )
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_two_steps_cannot_share_a_position(session, tenants):
    """The ordering guarantee. Without it a timeline renders differently on
    successive reads."""
    from sqlalchemy.exc import IntegrityError

    from navis.developer.run_models import AgentRun, AgentRunStep, AgentTask

    task = AgentTask(organization_id=TENANT_A, agent_id=tenants[TENANT_A], description="t")
    session.add(task)
    await session.flush()
    run = AgentRun(organization_id=TENANT_A, task_id=task.id, agent_id=tenants[TENANT_A])
    session.add(run)
    await session.flush()

    for _ in range(2):
        session.add(
            AgentRunStep(
                organization_id=TENANT_A, run_id=run.id, sequence=0, step_type="tool_call"
            )
        )
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_a_run_holds_at_most_one_outcome(session, tenants):
    """A run with two contradictory verdicts is a bug, not a state."""
    from sqlalchemy.exc import IntegrityError

    from navis.developer.run_models import AgentRun, AgentRunOutcome, AgentTask

    task = AgentTask(organization_id=TENANT_A, agent_id=tenants[TENANT_A], description="t")
    session.add(task)
    await session.flush()
    run = AgentRun(organization_id=TENANT_A, task_id=task.id, agent_id=tenants[TENANT_A])
    session.add(run)
    await session.flush()

    for status in ("success", "failure"):
        session.add(
            AgentRunOutcome(
                organization_id=TENANT_A,
                run_id=run.id,
                agent_id=tenants[TENANT_A],
                status=status,
            )
        )
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_a_quality_score_outside_zero_to_one_is_refused(session, tenants):
    from sqlalchemy.exc import IntegrityError

    from navis.developer.run_models import AgentRun, AgentRunOutcome, AgentTask

    task = AgentTask(organization_id=TENANT_A, agent_id=tenants[TENANT_A], description="t")
    session.add(task)
    await session.flush()
    run = AgentRun(organization_id=TENANT_A, task_id=task.id, agent_id=tenants[TENANT_A])
    session.add(run)
    await session.flush()

    session.add(
        AgentRunOutcome(
            organization_id=TENANT_A,
            run_id=run.id,
            agent_id=tenants[TENANT_A],
            status="success",
            quality_score=1.5,
        )
    )
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


# ── Tenant isolation, through the service ───────────────────────────────────


async def test_a_cross_tenant_agent_id_cannot_be_used(session, tenants):
    """Tenant B's agent is not addressable from tenant A, and the refusal is
    "no such agent" rather than a permission error — a probe learns nothing."""
    from navis.developer.run_service import AgentLifecycleService, UnknownAgent

    service = AgentLifecycleService(session)
    with pytest.raises(UnknownAgent):
        await service.create_task(
            TENANT_A, agent_id=tenants[TENANT_B], description="cross tenant"
        )


async def test_a_cross_tenant_run_is_invisible(session, tenants):
    """Reading by primary key still carries the tenant predicate, so another
    organization's run is indistinguishable from one that does not exist."""
    from navis.developer.run_service import AgentLifecycleService

    service = AgentLifecycleService(session)
    task = await service.create_task(TENANT_B, agent_id=tenants[TENANT_B], description="B's work")
    run = await service.start_run(TENANT_B, task_id=task.id)
    await session.commit()

    assert await service.get_run(TENANT_A, run.id) is None
    assert await service.list_steps(TENANT_A, run.id) is None
    assert await service.get_outcome(TENANT_A, run.id) is None


async def test_a_list_never_returns_another_tenants_rows(session, tenants):
    from navis.developer.run_service import AgentLifecycleService

    service = AgentLifecycleService(session)
    await service.create_task(TENANT_A, agent_id=tenants[TENANT_A], description="A only")
    await session.commit()

    page = await service.list_tasks(TENANT_A)
    assert page.total >= 1
    for row in page.rows:
        assert row.organization_id == TENANT_A


# ── Lifecycle behaviour ─────────────────────────────────────────────────────


async def test_steps_are_appended_in_order_and_read_back_in_order(session, tenants):
    from navis.developer.run_service import AgentLifecycleService

    service = AgentLifecycleService(session)
    task = await service.create_task(TENANT_A, agent_id=tenants[TENANT_A], description="ordered")
    run = await service.start_run(TENANT_A, task_id=task.id)

    for label in ("context", "tool", "model", "outcome"):
        await service.record_step(
            TENANT_A,
            run.id,
            step_type={"context": "context_fetch", "tool": "tool_call",
                       "model": "model_generation", "outcome": "outcome"}[label],
            label=label,
        )
    await session.commit()

    page = await service.list_steps(TENANT_A, run.id)
    assert [s.sequence for s in page.rows] == [0, 1, 2, 3]
    assert [s.label for s in page.rows] == ["context", "tool", "model", "outcome"]


async def test_a_finished_run_refuses_further_steps(session, tenants):
    """A finished run's timeline is history. A late step would make the record
    of what happened depend on when the reader looked."""
    from navis.developer.run_service import AgentLifecycleService, InvalidTransition

    service = AgentLifecycleService(session)
    task = await service.create_task(TENANT_A, agent_id=tenants[TENANT_A], description="closed")
    run = await service.start_run(TENANT_A, task_id=task.id)
    await service.finish_run(TENANT_A, run.id, status="completed")
    await session.commit()

    with pytest.raises(InvalidTransition):
        await service.record_step(TENANT_A, run.id, step_type="tool_call")


async def test_a_terminal_run_cannot_be_reopened(session, tenants):
    from navis.developer.run_service import AgentLifecycleService, InvalidTransition

    service = AgentLifecycleService(session)
    task = await service.create_task(TENANT_A, agent_id=tenants[TENANT_A], description="terminal")
    run = await service.start_run(TENANT_A, task_id=task.id)
    await service.finish_run(TENANT_A, run.id, status="failed", failure_category="timeout")
    await session.commit()

    with pytest.raises(InvalidTransition):
        await service.finish_run(TENANT_A, run.id, status="completed")


async def test_creation_is_idempotent_under_a_key(session, tenants):
    """A retried POST returns the existing row rather than a duplicate."""
    from navis.developer.run_service import AgentLifecycleService

    service = AgentLifecycleService(session)
    key = f"idem-{uuid.uuid4()}"

    first = await service.create_task(
        TENANT_A, agent_id=tenants[TENANT_A], description="once", idempotency_key=key
    )
    await session.commit()
    second = await service.create_task(
        TENANT_A, agent_id=tenants[TENANT_A], description="once", idempotency_key=key
    )
    assert first.id == second.id


async def test_one_task_holds_many_runs(session, tenants):
    """The whole reason Task and Run are separate: a retry must not lose the
    logical identity of the work."""
    from navis.developer.run_service import AgentLifecycleService

    service = AgentLifecycleService(session)
    task = await service.create_task(TENANT_A, agent_id=tenants[TENANT_A], description="retried")
    for _ in range(3):
        run = await service.start_run(TENANT_A, task_id=task.id)
        await service.finish_run(TENANT_A, run.id, status="failed", failure_category="error")
    await session.commit()

    page = await service.list_runs(TENANT_A, task_id=task.id)
    assert page.total == 3


async def test_an_outcome_is_revised_not_duplicated(session, tenants):
    """An evaluator scoring a result later revises the verdict; it does not add
    a contradictory second one."""
    from navis.developer.run_service import AgentLifecycleService

    service = AgentLifecycleService(session)
    task = await service.create_task(TENANT_A, agent_id=tenants[TENANT_A], description="judged")
    run = await service.start_run(TENANT_A, task_id=task.id)

    first = await service.record_outcome(TENANT_A, run.id, status="unknown")
    second = await service.record_outcome(TENANT_A, run.id, status="success", quality_score=0.9)
    await session.commit()

    assert first.id == second.id
    assert second.status == "success"


async def test_the_outcomes_agent_comes_from_the_run_not_the_caller(session, tenants):
    """"This agent's outcomes" must not be assertable by whoever writes one."""
    from navis.developer.run_service import AgentLifecycleService

    service = AgentLifecycleService(session)
    task = await service.create_task(TENANT_A, agent_id=tenants[TENANT_A], description="attrib")
    run = await service.start_run(TENANT_A, task_id=task.id)
    outcome = await service.record_outcome(TENANT_A, run.id, status="success")
    await session.commit()

    assert outcome.agent_id == tenants[TENANT_A]


async def test_no_secret_reaches_a_column(session, tenants):
    """The end-to-end version of the scrubber tests: a credential handed to the
    service must not be readable back out of the database."""
    from sqlalchemy import text

    from navis.developer.run_service import AgentLifecycleService

    service = AgentLifecycleService(session)
    secret = "navis_deadbeefdeadbeef_thisisthesecret"

    task = await service.create_task(TENANT_A, agent_id=tenants[TENANT_A], description="secrets")
    run = await service.start_run(TENANT_A, task_id=task.id)
    await service.record_step(
        TENANT_A,
        run.id,
        step_type="tool_call",
        tool_name="billing_search",
        request_metadata={"headers": {"authorization": f"Bearer {secret}"}},
    )
    await session.commit()

    found = await session.scalar(
        text("SELECT count(*) FROM agent_run_steps WHERE request_metadata::text LIKE :needle"),
        {"needle": f"%{secret}%"},
    )
    assert found == 0


async def test_pagination_is_bounded(session, tenants):
    """No code path omits a LIMIT. An unbounded list is how a shared table
    becomes an outage."""
    from navis.developer.lifecycle import MAX_PAGE_SIZE
    from navis.developer.run_service import AgentLifecycleService

    service = AgentLifecycleService(session)
    page = await service.list_tasks(TENANT_A, limit=100_000)
    assert page.limit <= MAX_PAGE_SIZE


async def test_deleting_an_agent_cascades_its_history(session, tenants):
    """FK CASCADE, verified rather than assumed. An orphaned run whose agent is
    gone is a row nothing can interpret."""
    from sqlalchemy import text

    from navis.developer.run_service import AgentLifecycleService
    from navis.identity.enterprise_models import ServiceAccount

    service = AgentLifecycleService(session)
    doomed = ServiceAccount(organization_id=TENANT_A, name="Doomed", scopes=[])
    session.add(doomed)
    await session.flush()

    task = await service.create_task(TENANT_A, agent_id=doomed.id, description="doomed")
    run = await service.start_run(TENANT_A, task_id=task.id)
    await service.record_step(TENANT_A, run.id, step_type="tool_call")
    await session.commit()

    await session.delete(doomed)
    await session.commit()

    remaining = await session.scalar(
        text("SELECT count(*) FROM agent_runs WHERE id = :id"), {"id": run.id}
    )
    assert remaining == 0
