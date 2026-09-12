"""Run telemetry: cache split, reasoning tokens, cost provenance, context counts.

WHY THIS EXISTS WHEN 0048 IS THREE WEEKS OLD. 0048 built the execution ledger to
record WHAT an agent did. This adds the handful of columns needed to answer what
it COST and what it was WORTH — the measurements a baseline-versus-Navis-context
comparison is computed from. Everything here was found missing by trying to write
that comparison against the existing schema, not by speculation about what might
be useful later.

TEN NULLABLE COLUMNS ON THREE EXISTING TABLES. No new table, no new index, no
data rewritten, nothing dropped, no existing column changed. Every addition is
`ALTER TABLE ... ADD COLUMN ... NULL`, which on PostgreSQL 11+ is a catalog-only
change: no table rewrite and no full-table lock held for the duration. An older
process running against a migrated database is unaffected, because every new
column is nullable and nothing reads them yet.

THE FOUR THINGS ADDED, AND WHY EACH IS A COLUMN RATHER THAN JSONB:

  1. `cache_read_tokens` / `cache_write_tokens` on steps. The existing
     `cached_tokens` is a single total, and a total cannot be priced: Anthropic
     bills a cache WRITE at ~1.25x base input and a cache READ at ~0.1x — more
     than twelve to one — and reports them as two separate usage fields for
     exactly that reason. `cached_tokens` is KEPT, not backfilled and not
     derived: a caller whose provider reports only a total is telling the truth,
     and inventing a split for it would be fabricating the expensive half.

  2. `reasoning_tokens` on steps. Billed as output, usually already inside
     `output_tokens`. Recorded alongside rather than added, so "how much of this
     was deliberation" is answerable without double-counting.

  3. `cost_basis` / `pricing_version` on steps, plus the CHECK that a cost must
     carry a basis. A provider returns token counts, not money; every currency
     figure is arithmetic over a price list until a provider says otherwise. The
     constraint is the point of the migration as much as the columns are — it
     makes an unattributed cost UNREPRESENTABLE, so no future writer can leave a
     number that a reader will mistake for billing.

  4. `context_items` / `context_tokens` / `retrieval_count` on steps and
     `context_version` on runs. These are the independent variable of the whole
     comparison — the question is whether supplying organizational context
     changes cost and outcome, which means summing context per run and grouping
     by version. An aggregate over JSONB can use no index and needs a cast per
     row. Counts only: what was supplied, never what it said.

  Plus `human_corrections` on outcomes, nullable so "nobody reviewed it" stays
  distinct from "reviewed, nothing to fix" — the numerator of "human corrections
  per successful task" would otherwise read an unreviewed backlog as flawless.

THE NEW CHECK CONSTRAINTS VALIDATE IMMEDIATELY, and that is safe here only
because it is knowably cheap: `ck_agent_run_steps_cost_has_basis` and the
widened counter check scan `agent_run_steps` once under an ACCESS EXCLUSIVE lock.
On a table this domain has barely begun to fill that is milliseconds. IF THIS
LEDGER HAS GROWN LARGE BY THE TIME YOU APPLY THIS, split each into `ADD
CONSTRAINT ... NOT VALID` followed by a separate `VALIDATE CONSTRAINT`, which
takes only a SHARE UPDATE EXCLUSIVE lock and does not block writes.

THE COUNTER CHECK IS DROPPED AND RECREATED because it is widened to cover the
six new counters. Between the DROP and the ADD, inside this transaction, no
other session can write the table — the lock is held — so there is no window in
which a negative count could land.

DOWNGRADE drops the ten columns and restores the original counter check. It
destroys the telemetry recorded against them, which is the data; it is written
for a forward-fix in staging, never as a routine production operation.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Revision ids stay <= 32 characters: `alembic_version.version_num` is
# varchar(32), and a longer id fails on the version bump AFTER upgrade() has
# already run.
revision: str = "0049_run_telemetry"
down_revision: Union[str, None] = "0048_agent_execution"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: The counter check as 0048 wrote it. Named here so `downgrade` restores
#: exactly what it replaced rather than an approximation of it.
_COUNTERS_BEFORE = (
    "(input_tokens IS NULL OR input_tokens >= 0) AND "
    "(output_tokens IS NULL OR output_tokens >= 0) AND "
    "(cached_tokens IS NULL OR cached_tokens >= 0) AND "
    "(cost_micros IS NULL OR cost_micros >= 0) AND "
    "(latency_ms IS NULL OR latency_ms >= 0)"
)

#: The same check widened to the six counters this migration adds. Kept in one
#: string so it cannot drift from the identical constraint on the ORM model.
_COUNTERS_AFTER = (
    "(input_tokens IS NULL OR input_tokens >= 0) AND "
    "(output_tokens IS NULL OR output_tokens >= 0) AND "
    "(cached_tokens IS NULL OR cached_tokens >= 0) AND "
    "(cache_read_tokens IS NULL OR cache_read_tokens >= 0) AND "
    "(cache_write_tokens IS NULL OR cache_write_tokens >= 0) AND "
    "(reasoning_tokens IS NULL OR reasoning_tokens >= 0) AND "
    "(context_items IS NULL OR context_items >= 0) AND "
    "(context_tokens IS NULL OR context_tokens >= 0) AND "
    "(retrieval_count IS NULL OR retrieval_count >= 0) AND "
    "(cost_micros IS NULL OR cost_micros >= 0) AND "
    "(latency_ms IS NULL OR latency_ms >= 0)"
)

#: Sorted, matching `_one_of` in the ORM models so autogenerate sees no drift.
_COST_BASIS_VALUES = "'estimated', 'provider_reported'"


def upgrade() -> None:
    # ── agent_run_steps: usage the existing columns cannot express ──────
    op.add_column(
        "agent_run_steps", sa.Column("cache_read_tokens", sa.Integer(), nullable=True)
    )
    op.add_column(
        "agent_run_steps", sa.Column("cache_write_tokens", sa.Integer(), nullable=True)
    )
    op.add_column(
        "agent_run_steps", sa.Column("reasoning_tokens", sa.Integer(), nullable=True)
    )

    # ── agent_run_steps: where a cost figure came from ─────────────────
    op.add_column("agent_run_steps", sa.Column("cost_basis", sa.String(24), nullable=True))
    op.add_column(
        "agent_run_steps", sa.Column("pricing_version", sa.String(64), nullable=True)
    )

    # ── agent_run_steps: how much context was supplied ─────────────────
    op.add_column("agent_run_steps", sa.Column("context_items", sa.Integer(), nullable=True))
    op.add_column("agent_run_steps", sa.Column("context_tokens", sa.Integer(), nullable=True))
    op.add_column(
        "agent_run_steps", sa.Column("retrieval_count", sa.Integer(), nullable=True)
    )

    op.drop_constraint(
        "ck_agent_run_steps_counters_non_negative", "agent_run_steps", type_="check"
    )
    op.create_check_constraint(
        "ck_agent_run_steps_counters_non_negative", "agent_run_steps", _COUNTERS_AFTER
    )
    op.create_check_constraint(
        "ck_agent_run_steps_cost_basis",
        "agent_run_steps",
        f"cost_basis IS NULL OR cost_basis IN ({_COST_BASIS_VALUES})",
    )
    # The constraint this migration is really for: no cost without a source.
    op.create_check_constraint(
        "ck_agent_run_steps_cost_has_basis",
        "agent_run_steps",
        "cost_micros IS NULL OR cost_basis IS NOT NULL",
    )

    # ── agent_runs: which context assembly produced the snapshot ───────
    op.add_column("agent_runs", sa.Column("context_version", sa.String(64), nullable=True))

    # ── agent_run_outcomes: how much a human had to fix ────────────────
    op.add_column(
        "agent_run_outcomes", sa.Column("human_corrections", sa.Integer(), nullable=True)
    )
    op.create_check_constraint(
        "ck_agent_run_outcomes_human_corrections",
        "agent_run_outcomes",
        "human_corrections IS NULL OR human_corrections >= 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_agent_run_outcomes_human_corrections", "agent_run_outcomes", type_="check"
    )
    op.drop_column("agent_run_outcomes", "human_corrections")

    op.drop_column("agent_runs", "context_version")

    op.drop_constraint("ck_agent_run_steps_cost_has_basis", "agent_run_steps", type_="check")
    op.drop_constraint("ck_agent_run_steps_cost_basis", "agent_run_steps", type_="check")
    op.drop_constraint(
        "ck_agent_run_steps_counters_non_negative", "agent_run_steps", type_="check"
    )
    op.create_check_constraint(
        "ck_agent_run_steps_counters_non_negative", "agent_run_steps", _COUNTERS_BEFORE
    )

    for column in (
        "retrieval_count",
        "context_tokens",
        "context_items",
        "pricing_version",
        "cost_basis",
        "reasoning_tokens",
        "cache_write_tokens",
        "cache_read_tokens",
    ):
        op.drop_column("agent_run_steps", column)
