"""agent_tasks (organization_id, task_metadata->>'navisWorkKey') — the continuity lookup

Revision ID: 0054_task_work_key_ix
Revises: 0053_developer_access
Create Date: 2026-09-17

WHAT THIS FIXES. A client that records work against a task and then loses the id
— a developer tool restarting in the same worktree, a scheduler resuming after a
deploy — has no way back to that task. A task is addressable by id, and the id is
exactly what a fresh process does not have. Its own key, written into
`task_metadata.navisWorkKey` when the task was created, is the only way back.

Until now that lookup was `GET /developer/tasks?status=open` paged three times
and matched CLIENT-SIDE, and the failure mode is the one that makes this a
correctness fix rather than a performance one: the scan does not error when the
tenant has more open tasks than the client's window. It reports the work stream
as ABSENT. A second stream is then opened beside the first, the original's
history stops being reachable, and nothing anywhere logs a problem. This is the
same defect `ix_agent_runs_task_created` closed one level down, arriving one
level up.

WHY AN EXPRESSION B-TREE AND NOT GIN. The query is a single equality on ONE known
key:

    WHERE organization_id = $1
      AND (task_metadata ->> 'navisWorkKey') = $2

which a b-tree answers in one descent. A GIN index over `task_metadata` would
index every key in every document to serve one of them, cost far more to build
and maintain, and — the deciding reason — could not LEAD WITH THE TENANT. Every
other index on this table leads with `organization_id` so that one organization's
volume stays off another's scan, and that property is worth more here than
generality nobody has asked for.

MEASURED, not assumed. One tenant, one buried work key, PostgreSQL 14 on a
laptop with the table in cache; median of 20 runs of the exact query the service
issues. `tests/integration/test_developer_work_key_postgres.py` builds the same
shape and asserts the plan.

    open tasks      with index      without
           600         0.09 ms      0.16 ms
         5,000         0.09 ms      0.61 ms
        50,000         0.08 ms      5.52 ms
       250,000         0.10 ms     27.04 ms

CONSTANT against LINEAR, which is the whole claim. The absolute numbers are small
because everything is cached on one machine and they are not a production
figure; the shape is what carries over. Without the index the tenant pays for
every task it has ever left open, forever, on a lookup that happens at the start
of every session — and the client-side page window does not grow with the tenant.

NOT PARTIAL. `WHERE (task_metadata ->> 'navisWorkKey') IS NOT NULL` would be
smaller, because tasks created by anything other than a work-state client index
as NULL. It is deliberately not used: a partial index is only usable when the
planner can prove the predicate holds for the query, which makes the plan depend
on a condition no caller states. A NULL entry per unkeyed task is the cheaper
problem.

SAFE TO APPLY ONLINE. Pure-additive, reversible, and created CONCURRENTLY so it
never takes an ACCESS EXCLUSIVE lock on `agent_tasks` — the table is on the
authenticated read path of every Developer client, and a lock held for the length
of a build would stall all of them. CONCURRENTLY requires autocommit and cannot
run inside the migration's transaction, which is what `autocommit_block` is for.
Same pattern as 0028 and 0044.

ONE OPERATIONAL NOTE. A CONCURRENTLY build that is interrupted leaves an INVALID
index behind, which the planner ignores and which must be dropped before
retrying:

    DROP INDEX CONCURRENTLY ix_agent_tasks_org_work_key;

`if_not_exists` below makes the retry itself safe, but it will NOT repair an
invalid index — it sees the name and does nothing. Check
`pg_index.indisvalid` after an interrupted run.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Revision ids stay <= 32 characters: `alembic_version.version_num` is
# varchar(32), and a longer id fails on the version bump AFTER upgrade() has
# already run.
revision: str = "0054_task_work_key_ix"
down_revision: Union[str, None] = "0053_developer_access"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: Spelled exactly as `AgentTask.__table_args__` spells it, and exactly as
#: `run_service._work_key_expression()` renders it. PostgreSQL matches a query
#: to an expression index by comparing the parsed expressions, so a difference
#: in spelling here is a silent sequential scan there.
_EXPRESSION = "(task_metadata ->> 'navisWorkKey')"


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.create_index(
            "ix_agent_tasks_org_work_key",
            "agent_tasks",
            ["organization_id", sa.text(_EXPRESSION)],
            postgresql_concurrently=True,
            if_not_exists=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            "ix_agent_tasks_org_work_key",
            table_name="agent_tasks",
            postgresql_concurrently=True,
            if_exists=True,
        )
