"""Agent execution lifecycle: agent_tasks, agent_runs, agent_run_steps, agent_run_outcomes.

WHAT THIS ADDS, AND WHAT IT DELIBERATELY DOES NOT. Four NEW tables recording what
an external agent did: what it was asked (task), each attempt (run), the ordered
timeline of that attempt (steps), and what came of it (outcome). It creates no
identity: `agent_id` is a `service_accounts.id`, the same row
`/api/v1/developer/agents` already serves and the same row
`authz._principal_from_api_key` already resolves a credential to. There is no
second machine identity, no second tenant column, and no change to any existing
table — this migration is purely additive, so an older process running against a
migrated database is unaffected.

WHY NOT `execution_requests`. That aggregate is Navis acting on the
organization's behalf through providers it owns, with a decision, an obligation
and an approval model. An agent run is an account of work done ELSEWHERE, by
software Navis does not run. They share the word "execution" and nothing else,
and putting an external agent's self-reported timeline into the aggregate the
application trusts to decide what it has already done would be a correctness bug
long before it was a modelling one.

WHY NOT THE EXISTING `outcomes` TABLE. `outcomes` is keyed by a deterministic
uuid5 of a provider `source_event_id` and carries `execution_id`/`decision_id`
lineage; it is written by webhooks and the observation service. An agent outcome
has no source event and none of that lineage, so reusing the table would mean
nullable lineage columns and a synthetic event id on a table the application
pipeline depends on. `agent_run_outcomes` is separate and joins to the same
Learning layer later.

`organization_id` IS ON ALL FOUR TABLES, INCLUDING THE CHILDREN, and that
denormalization is the tenant-isolation guarantee rather than an optimization.
With it, `WHERE organization_id = :org` is the first predicate of every query and
the leading column of every index; without it a step's tenant is two joins away
and isolation depends on every call site writing those joins correctly. A tenant
filter that can be forgotten is one that eventually is.

CHECK CONSTRAINTS ARE DECLARED IN SQL, not only at the edge, because the edge is
one of several writers over a table's life (a fixture, a later migration, an
operator's psql session) and a free-text status cannot be grouped or counted.
They are declared identically on the ORM models, so `Base.metadata.create_all`
(what the unit suites build from) and a migrated database produce one schema.

LOCKING AND SAFETY. Four `CREATE TABLE`s and their indexes on tables that do not
yet exist, so nothing is locked, nothing is rewritten, and no existing query
plan changes. Indexes are created inline rather than CONCURRENTLY: the tables are
empty at creation, so there is nothing to build against and CONCURRENTLY would
merely forbid running this inside the migration transaction.

DOWNGRADE drops all four in FK-dependency order. It is destructive by nature —
the execution history is the data — and is written for a forward-fix in staging,
never as a routine production operation.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# Revision ids stay <= 32 characters: `alembic_version.version_num` is
# varchar(32), and a longer id fails on the version bump AFTER upgrade() has
# already run.
revision: str = "0048_agent_execution"
down_revision: Union[str, None] = "0047_cleanup_verification_data"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Closed vocabularies, spelled out here rather than imported from
# `navis.developer.lifecycle`. A migration is a historical record of the schema
# at a point in time: importing today's constant would make this file's meaning
# change when that constant does, and a replay against an old database would
# then produce a schema the original migration never created.
_TASK_STATUSES = "'abandoned', 'completed', 'open'"
_RUN_STATUSES = "'cancelled', 'completed', 'failed', 'queued', 'running'"
_STEP_TYPES = "'context_fetch', 'model_generation', 'outcome', 'tool_call'"
_STEP_STATUSES = "'completed', 'failed', 'running', 'skipped'"
_TOOL_KINDS = "'database', 'handoff', 'mcp', 'navis', 'other', 'rest'"
_ERROR_CATEGORIES = (
    "'cancelled', 'error', 'invalid', 'rate_limited', 'refused', 'timeout'"
)
_OUTCOME_STATUSES = "'failure', 'partial', 'success', 'unknown'"


def upgrade() -> None:
    # ── agent_tasks ─────────────────────────────────────────────────────
    op.create_table(
        "agent_tasks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("service_accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("description", sa.String(2000), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("idempotency_key", sa.String(128), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "task_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.CheckConstraint(f"status IN ({_TASK_STATUSES})", name="ck_agent_tasks_status"),
        # NULL idempotency keys do not collide in a UNIQUE index, so unkeyed
        # creates stay independent while keyed ones are exactly-once per tenant.
        sa.UniqueConstraint(
            "organization_id", "idempotency_key", name="uq_agent_tasks_org_idempotency"
        ),
    )
    op.create_index("ix_agent_tasks_organization_id", "agent_tasks", ["organization_id"])
    op.create_index("ix_agent_tasks_agent_id", "agent_tasks", ["agent_id"])
    op.create_index(
        "ix_agent_tasks_org_created",
        "agent_tasks",
        ["organization_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_agent_tasks_agent_created",
        "agent_tasks",
        ["agent_id", sa.text("created_at DESC")],
    )

    # ── agent_runs ──────────────────────────────────────────────────────
    op.create_table(
        "agent_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("service_accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_category", sa.String(24), nullable=True),
        sa.Column("failure_reason", sa.String(500), nullable=True),
        sa.Column("context_snapshot_id", sa.String(128), nullable=True),
        sa.Column("correlation_id", sa.String(128), nullable=True),
        sa.Column("idempotency_key", sa.String(128), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "run_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.CheckConstraint(f"status IN ({_RUN_STATUSES})", name="ck_agent_runs_status"),
        sa.CheckConstraint(
            f"failure_category IS NULL OR failure_category IN ({_ERROR_CATEGORIES})",
            name="ck_agent_runs_failure_category",
        ),
        # A terminal run states when it ended, and a live run does not claim to
        # have. Enforced here because a duration computed from a missing
        # timestamp is silently wrong rather than loudly absent.
        sa.CheckConstraint(
            "(status IN ('completed','failed','cancelled')) = (completed_at IS NOT NULL)",
            name="ck_agent_runs_completed_at",
        ),
        sa.UniqueConstraint(
            "organization_id", "idempotency_key", name="uq_agent_runs_org_idempotency"
        ),
    )
    op.create_index("ix_agent_runs_organization_id", "agent_runs", ["organization_id"])
    op.create_index("ix_agent_runs_task_id", "agent_runs", ["task_id"])
    op.create_index("ix_agent_runs_agent_id", "agent_runs", ["agent_id"])
    op.create_index("ix_agent_runs_correlation_id", "agent_runs", ["correlation_id"])
    op.create_index(
        "ix_agent_runs_org_created", "agent_runs", ["organization_id", sa.text("created_at DESC")]
    )
    op.create_index(
        "ix_agent_runs_agent_created", "agent_runs", ["agent_id", sa.text("created_at DESC")]
    )
    op.create_index(
        "ix_agent_runs_task_created", "agent_runs", ["task_id", sa.text("created_at DESC")]
    )
    op.create_index("ix_agent_runs_org_status", "agent_runs", ["organization_id", "status"])

    # ── agent_run_steps ─────────────────────────────────────────────────
    op.create_table(
        "agent_run_steps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sequence", sa.Integer, nullable=False),
        sa.Column("step_type", sa.String(24), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("label", sa.String(200), nullable=False, server_default=""),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tool_name", sa.String(200), nullable=True),
        sa.Column("tool_kind", sa.String(16), nullable=True),
        # Provider and model are free-form strings on purpose: Navis does not own
        # the model, and a closed set would need a migration every time a vendor
        # ships a version.
        sa.Column("model_provider", sa.String(64), nullable=True),
        sa.Column("model_name", sa.String(128), nullable=True),
        sa.Column("model_generation_id", sa.String(128), nullable=True),
        sa.Column("input_tokens", sa.Integer, nullable=True),
        sa.Column("output_tokens", sa.Integer, nullable=True),
        sa.Column("cached_tokens", sa.Integer, nullable=True),
        # Millionths of a currency unit as an integer. Never a float: summing
        # floats over millions of rows accumulates error, and money that does not
        # add up is worse than money that is absent.
        sa.Column("cost_micros", sa.BigInteger, nullable=True),
        sa.Column("cost_currency", sa.String(3), nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("error_category", sa.String(24), nullable=True),
        sa.Column("error_message", sa.String(500), nullable=True),
        sa.Column(
            "request_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "response_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.CheckConstraint(f"step_type IN ({_STEP_TYPES})", name="ck_agent_run_steps_type"),
        sa.CheckConstraint(f"status IN ({_STEP_STATUSES})", name="ck_agent_run_steps_status"),
        sa.CheckConstraint(
            f"tool_kind IS NULL OR tool_kind IN ({_TOOL_KINDS})",
            name="ck_agent_run_steps_tool_kind",
        ),
        sa.CheckConstraint(
            f"error_category IS NULL OR error_category IN ({_ERROR_CATEGORIES})",
            name="ck_agent_run_steps_error_category",
        ),
        # A negative token count corrupts every aggregate built on this table,
        # silently. The database refuses instead.
        sa.CheckConstraint(
            "(input_tokens IS NULL OR input_tokens >= 0) AND "
            "(output_tokens IS NULL OR output_tokens >= 0) AND "
            "(cached_tokens IS NULL OR cached_tokens >= 0) AND "
            "(cost_micros IS NULL OR cost_micros >= 0) AND "
            "(latency_ms IS NULL OR latency_ms >= 0)",
            name="ck_agent_run_steps_counters_non_negative",
        ),
        sa.CheckConstraint("sequence >= 0", name="ck_agent_run_steps_sequence"),
        # The ordering guarantee AND the read path. UNIQUE makes a run's timeline
        # deterministic rather than merely usually-ordered.
        sa.UniqueConstraint("run_id", "sequence", name="uq_agent_run_steps_run_sequence"),
    )
    op.create_index("ix_agent_run_steps_organization_id", "agent_run_steps", ["organization_id"])
    op.create_index("ix_agent_run_steps_run_id", "agent_run_steps", ["run_id"])
    op.create_index(
        "ix_agent_run_steps_org_created",
        "agent_run_steps",
        ["organization_id", sa.text("created_at DESC")],
    )

    # ── agent_run_outcomes ──────────────────────────────────────────────
    op.create_table(
        "agent_run_outcomes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("service_accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(16), nullable=False, server_default="unknown"),
        sa.Column("summary", sa.Text, nullable=False, server_default=""),
        sa.Column("result_reference", sa.String(500), nullable=True),
        # NULL until something evaluates it: "not judged" and "judged worthless"
        # are different facts, and a default of 0 would erase the distinction.
        sa.Column("quality_score", sa.Float, nullable=True),
        sa.Column("produced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "outcome_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.CheckConstraint(
            f"status IN ({_OUTCOME_STATUSES})", name="ck_agent_run_outcomes_status"
        ),
        sa.CheckConstraint(
            "quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)",
            name="ck_agent_run_outcomes_quality_score",
        ),
        # One verdict per run. A run with two contradictory outcomes is a bug,
        # not a state worth representing.
        sa.UniqueConstraint("run_id", name="uq_agent_run_outcomes_run"),
    )
    op.create_index(
        "ix_agent_run_outcomes_organization_id", "agent_run_outcomes", ["organization_id"]
    )
    op.create_index("ix_agent_run_outcomes_agent_id", "agent_run_outcomes", ["agent_id"])
    op.create_index(
        "ix_agent_run_outcomes_org_created",
        "agent_run_outcomes",
        ["organization_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_agent_run_outcomes_agent_status", "agent_run_outcomes", ["agent_id", "status"]
    )


def downgrade() -> None:
    # FK-dependency order: children before parents. Indexes and constraints go
    # with their tables, so they are not dropped individually.
    op.drop_table("agent_run_outcomes")
    op.drop_table("agent_run_steps")
    op.drop_table("agent_runs")
    op.drop_table("agent_tasks")
