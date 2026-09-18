"""The work-key lookup, against real PostgreSQL.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://.../navis_test \
        pytest tests/integration/test_developer_work_key_postgres.py -q

WHY THIS NEEDS A REAL DATABASE AND A LARGE TENANT. The defect being fixed is not
"the lookup is slow". It is that the lookup SILENTLY REPORTS ABSENCE: a client
paging the tenant's open tasks and matching `metadata.navisWorkKey` itself stops
finding the work stream once the tenant holds more open tasks than the client is
willing to read, and nothing distinguishes that from a work stream that was never
opened. A duplicate stream is then created beside the original and the original's
history becomes unreachable.

Reproducing that requires more open tasks than the client window (3 pages of 200
= 600), which makes this a Postgres module rather than a unit test. The target
task is deliberately created FIRST, so it sits at the far end of the
`created_at DESC` ordering — outside the old window by construction, not by luck.

The module also pins the properties a filter must have to be safe at all: it
narrows the tenant predicate rather than replacing it, it matches exactly, and it
is answered by an index rather than a scan of the tenant's tasks.

SAFETY. Refuses to run unless the target database is named `navis_test*`, and
never drops a schema.
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

TENANT_A = uuid.UUID("cccc0000-0000-4000-8000-00000000000a")
TENANT_B = uuid.UUID("cccc0000-0000-4000-8000-00000000000b")

#: The work key the CLI would compute for one worktree. Schema-tagged exactly as
#: `packages/cli/src/workstate/identity.ts` builds it.
TARGET_KEY = "nws1:" + "a" * 32
#: Same tenant, different worktree. Proves the match is on the whole value.
OTHER_KEY = "nws1:" + "b" * 32

#: Comfortably past the old client-side window of 3 pages x 200 rows. Kept as a
#: named constant because the number IS the regression: drop it below 600 and
#: this module passes against the code it was written to fail.
OLD_CLIENT_WINDOW = 600
FILLER_TASKS = 650


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
    """Two tenants; tenant A holds one keyed task buried under 650 others.

    The buried task is inserted BEFORE the filler, so `created_at DESC, id DESC`
    puts it last. That is what makes "outside the old window" a property of the
    fixture rather than an accident of timing.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.developer.run_models import AgentTask
    from navis.identity.enterprise_models import ServiceAccount
    from navis.identity.models import Organization

    from sqlalchemy import text

    # ── Clean this module's tenants FIRST, never `drop_all` ────────────
    # The database outlives the run — that is what makes the schema build
    # once — so a module that only inserts doubles its own fixture on the
    # second run and every count assertion becomes a function of how many
    # times the suite has been run. Deleting the ORGANIZATION cascades to its
    # service accounts, tasks and runs, which is the whole footprint of this
    # module. `drop_all` is avoided for the reason the other Postgres modules
    # document: it re-oids every PG ENUM and breaks pooled asyncpg codecs.
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM organizations WHERE id = ANY(:ids)").bindparams(
                ids=[TENANT_A, TENANT_B]
            )
        )

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        agents: dict[uuid.UUID, uuid.UUID] = {}
        for org_id, tag in ((TENANT_A, "A"), (TENANT_B, "B")):
            if await s.get(Organization, org_id) is None:
                s.add(
                    Organization(
                        id=org_id, name=f"WorkKey {tag}", slug=f"workkey-{tag.lower()}-{org_id.hex[:6]}"
                    )
                )
                await s.flush()
            agent = ServiceAccount(organization_id=org_id, name=f"Agent {tag}", scopes=["view_state"])
            s.add(agent)
            await s.flush()
            agents[org_id] = agent.id

        def task(org_id: uuid.UUID, description: str, metadata: dict, status: str = "open") -> AgentTask:
            return AgentTask(
                organization_id=org_id,
                agent_id=agents[org_id],
                description=description,
                status=status,
                task_metadata=metadata,
            )

        # The one that must survive. First, so it sorts last.
        target = task(TENANT_A, "the buried work stream", {"navisWorkKey": TARGET_KEY})
        s.add(target)
        await s.flush()

        # Tenant B holds a task under the SAME key. If the filter ever replaced
        # the tenant predicate instead of narrowing it, this row is what would
        # leak — so it exists in every run, not only in the isolation test.
        foreign = task(TENANT_B, "another tenant's work stream", {"navisWorkKey": TARGET_KEY})
        s.add(foreign)

        # A neighbouring worktree in the same tenant. Catches a prefix or
        # containment match pretending to be an equality.
        s.add(task(TENANT_A, "a different worktree", {"navisWorkKey": OTHER_KEY}))

        # A closed task under the target key: `status` and `work_key` have to
        # compose, or a closed stream reads as the current one.
        s.add(task(TENANT_A, "finished work", {"navisWorkKey": TARGET_KEY}, status="completed"))

        # The burial. Unkeyed tasks, as an organization doing anything else
        # produces — they must not match, and they must not hide the target.
        for i in range(FILLER_TASKS):
            s.add(task(TENANT_A, f"unrelated open task {i}", {"kind": "filler", "n": i}))

        await s.commit()

        async with engine.begin() as conn:
            from sqlalchemy import text

            await conn.execute(text("ANALYZE agent_tasks"))

        yield {"agents": agents, "target_id": target.id, "foreign_id": foreign.id}


@pytest_asyncio.fixture(loop_scope="module")
async def session(engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s
        await s.rollback()


def _service(session):
    from navis.developer.run_service import AgentLifecycleService

    return AgentLifecycleService(session)


# ── The regression ──────────────────────────────────────────────────────────


async def test_the_target_is_outside_the_old_client_side_window(session, seeded):
    """THE PRECONDITION. Without this the test below proves nothing: a lookup
    that finds the task in the first 600 rows would pass with or without the
    fix."""
    page = await _service(session).list_tasks(TENANT_A, status="open", limit=200, offset=0)
    assert page.total > OLD_CLIENT_WINDOW, (
        f"only {page.total} open tasks — the old 600-row window would have found "
        "the target, so this module cannot detect the regression"
    )

    # Walk the window the client used to walk, and confirm it comes up empty.
    seen: list[str] = []
    for offset in range(0, OLD_CLIENT_WINDOW, 200):
        page = await _service(session).list_tasks(TENANT_A, status="open", limit=200, offset=offset)
        seen.extend(t.task_metadata.get("navisWorkKey", "") for t in page.rows)
    assert TARGET_KEY not in seen, (
        "the target appeared inside the old window; the fixture is not burying it"
    )


async def test_the_scoped_lookup_finds_the_buried_work_stream(session, seeded):
    """THE FIX. One query, no paging, and the row the old scan could not reach."""
    page = await _service(session).list_tasks(TENANT_A, status="open", work_key=TARGET_KEY)

    assert page.total == 1
    assert [t.id for t in page.rows] == [seeded["target_id"]]
    assert page.rows[0].description == "the buried work stream"


# ── The properties that make it safe ────────────────────────────────────────


async def test_a_foreign_tenant_never_reaches_another_tenants_work_key(session, seeded):
    """The filter NARROWS the tenant predicate; it never replaces it. Tenant B
    holds a task under the identical key, and neither tenant can see the
    other's."""
    a = await _service(session).list_tasks(TENANT_A, work_key=TARGET_KEY)
    b = await _service(session).list_tasks(TENANT_B, work_key=TARGET_KEY)

    a_ids = {t.id for t in a.rows}
    b_ids = {t.id for t in b.rows}
    assert seeded["target_id"] in a_ids
    assert seeded["foreign_id"] in b_ids
    assert a_ids & b_ids == set()
    assert seeded["foreign_id"] not in a_ids
    # Tenant B has exactly the one task; a leak would show as more.
    assert b.total == 1


async def test_an_unknown_work_key_returns_nothing_rather_than_everything(session, seeded):
    """The failure that matters: a lookup that matched nothing must not fall
    back to the unfiltered list."""
    page = await _service(session).list_tasks(TENANT_A, work_key="nws1:" + "f" * 32)
    assert page.rows == []
    assert page.total == 0


async def test_the_match_is_exact_and_not_a_prefix(session, seeded):
    """A prefix or containment match would hand a caller its neighbour's work
    stream — the one failure worse than finding nothing."""
    truncated = await _service(session).list_tasks(TENANT_A, work_key=TARGET_KEY[:20])
    assert truncated.rows == []

    extended = await _service(session).list_tasks(TENANT_A, work_key=TARGET_KEY + "x")
    assert extended.rows == []

    other = await _service(session).list_tasks(TENANT_A, status="open", work_key=OTHER_KEY)
    assert [t.description for t in other.rows] == ["a different worktree"]


async def test_a_blank_work_key_matches_nothing_rather_than_the_whole_tenant(session, seeded):
    """A caller that computed an empty key has a bug. Answering it with the
    tenant's entire task list would hide that bug behind a plausible page."""
    page = await _service(session).list_tasks(TENANT_A, work_key="")
    assert page.rows == []
    assert page.total == 0


async def test_work_key_composes_with_status_rather_than_overriding_it(session, seeded):
    """Two filters, both applied. The closed task under the same key is what
    separates 'composes' from 'the last one wins'."""
    every = await _service(session).list_tasks(TENANT_A, work_key=TARGET_KEY)
    assert every.total == 2  # one open, one completed

    open_only = await _service(session).list_tasks(TENANT_A, status="open", work_key=TARGET_KEY)
    assert open_only.total == 1
    assert open_only.rows[0].status == "open"

    closed_only = await _service(session).list_tasks(TENANT_A, status="completed", work_key=TARGET_KEY)
    assert closed_only.total == 1
    assert closed_only.rows[0].status == "completed"


async def test_duplicate_work_keys_are_returned_newest_first_and_paged(session, seeded):
    """Two open tasks CAN carry one key — two sessions racing to open a stream
    is a real sequence, and the ledger does not forbid it. The lookup must
    return them in a defined order rather than an arbitrary one, because the
    client takes the first."""
    from navis.developer.run_models import AgentTask

    extra = AgentTask(
        organization_id=TENANT_A,
        agent_id=seeded["agents"][TENANT_A],
        description="a racing second stream",
        status="open",
        task_metadata={"navisWorkKey": TARGET_KEY},
    )
    session.add(extra)
    await session.commit()
    try:
        page = await _service(session).list_tasks(TENANT_A, status="open", work_key=TARGET_KEY)
        assert page.total == 2
        # Newest first: the row just written leads.
        assert page.rows[0].id == extra.id

        first = await _service(session).list_tasks(
            TENANT_A, status="open", work_key=TARGET_KEY, limit=1, offset=0
        )
        second = await _service(session).list_tasks(
            TENANT_A, status="open", work_key=TARGET_KEY, limit=1, offset=1
        )
        assert first.total == second.total == 2
        assert [t.id for t in first.rows] == [extra.id]
        assert [t.id for t in second.rows] == [seeded["target_id"]]
    finally:
        await session.delete(extra)
        await session.commit()


async def test_the_unfiltered_listing_is_unchanged(session, seeded):
    """The existing endpoint keeps its existing behaviour: every open task in
    the tenant, newest first, unaffected by a filter nobody passed."""
    page = await _service(session).list_tasks(TENANT_A, status="open", limit=200, offset=0)

    # The filler, plus the two OPEN keyed tasks. The completed one under the
    # target key is excluded by `status`, and the other tenant's by the tenant.
    assert page.total == FILLER_TASKS + 2
    assert len(page.rows) == 200
    assert page.limit == 200 and page.offset == 0
    ordered = [(t.created_at, t.id) for t in page.rows]
    assert ordered == sorted(ordered, reverse=True)


# ── The index ───────────────────────────────────────────────────────────────


async def test_the_lookup_is_answered_by_the_index_and_not_a_scan(engine, seeded):
    """THE SCALE PROPERTY, asserted rather than assumed.

    An expression index is only used when the query's expression matches the
    index's EXACTLY. SQLAlchemy's natural spelling renders the JSON key as a
    bind parameter, which does not match under a generic plan — so this asserts
    the plan, not the presence of the index.
    """
    from sqlalchemy import text

    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "EXPLAIN (FORMAT TEXT) SELECT * FROM agent_tasks "
                "WHERE organization_id = :org AND status = 'open' "
                "AND (task_metadata ->> 'navisWorkKey') = :key "
                "ORDER BY created_at DESC, id DESC LIMIT 200"
            ).bindparams(org=TENANT_A, key=TARGET_KEY)
        )
        # EXPLAIN returns the plan ONE ROW PER LINE. Reading only the first
        # scalar returns "Limit ..." and would assert nothing about the scan
        # underneath it.
        plan = "\n".join(row[0] for row in result)

    assert "ix_agent_tasks_org_work_key" in plan, plan
    assert "Seq Scan" not in plan, plan
