"""Durable ReasoningSignal provenance.

Execution, Observation, Learning and Intelligence all persist
`reasoning_signal_ids`, but the signals themselves lived only in process memory:
Decision copied the few that became decisions into `daily_briefs.decisions` and
every other signal was discarded. Those stored ids were permanent references to
rows that never existed.

They cannot be recovered by recomputation either — `compute_signal_id` hashes
`reasoning_version` into the UUIDv5, so bumping the version (2 -> 3) changes
every id. The single reasoning reference in production today was written at v2
and can never be re-derived by current code.

This migration adds the table those references resolve to. Additive and
non-destructive: it creates one table, backfills nothing (fabricating provenance
would be worse than admitting the gap), and touches no existing row. Signals are
written from the next brief onward; references older than this table stay
unresolved and are honestly reported as such.

Revision ID: 0036_reasoning_signals
Revises: 0035_execution_obligation
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0036_reasoning_signals"
down_revision = "0035_execution_obligation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "reasoning_signals",
        # Already a deterministic UUIDv5 of the reasoning identity tuple, so it
        # is the natural primary key: replaying the same State at the same
        # reasoning_version converges on the same row instead of duplicating.
        sa.Column("signal_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("signal_type", sa.String(40), nullable=False),
        sa.Column("entity_type", sa.String(40), nullable=True),
        sa.Column("entity_id", sa.String(128), nullable=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("why", sa.Text(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("severity", sa.Float(), nullable=False, server_default="0"),
        sa.Column("engine_name", sa.String(64), nullable=False),
        sa.Column("engine_version", sa.String(16), nullable=False),
        sa.Column("state_as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("state_watermark", sa.String(64), nullable=True),
        sa.Column("reasoning_version", sa.Integer(), nullable=False),
        # Logical time (= state_as_of), never wall clock.
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        if_not_exists=True,
    )
    # Tenant-scoped reads are the only reads: every resolve filters on org_id
    # first so an id replayed across orgs can never resolve the wrong signal.
    op.create_index(
        "ix_reasoning_signals_org", "reasoning_signals", ["organization_id"],
        if_not_exists=True,
    )
    # "What did reasoning say about this org at this watermark?" — the replay
    # question, and the one Intelligence/Decision ask when explaining a brief.
    op.create_index(
        "ix_reasoning_signals_org_watermark",
        "reasoning_signals",
        ["organization_id", "state_watermark"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_reasoning_signals_type", "reasoning_signals", ["signal_type"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_reasoning_signals_version", "reasoning_signals", ["reasoning_version"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_reasoning_signals_version", table_name="reasoning_signals")
    op.drop_index("ix_reasoning_signals_type", table_name="reasoning_signals")
    op.drop_index("ix_reasoning_signals_org_watermark", table_name="reasoning_signals")
    op.drop_index("ix_reasoning_signals_org", table_name="reasoning_signals")
    op.drop_table("reasoning_signals")
