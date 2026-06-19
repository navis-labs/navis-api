"""Learning V2 — append-only events + count aggregates + patterns + inbox

Revision ID: 0010_learning
Revises: 0009_observation
Create Date: 2026-06-19

Replaces the non-deterministic EMA learning with an append-only attribution log
and deterministic count projections. Legacy decision_patterns/behavior_patterns
are (re)created for Intelligence V1 compile-compat; Learning V2 never writes them.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010_learning"
down_revision: Union[str, None] = "0009_observation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "learning_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("outcome_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("execution_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("decision_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("reasoning_signal_ids", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("reasoning_version", sa.Integer(), nullable=True),
        sa.Column("state_watermark", sa.String(length=64), nullable=True),
        sa.Column("outcome_type", sa.String(length=40), nullable=False),
        sa.Column("result", sa.String(length=16), nullable=False),
        sa.Column("schema_version", sa.Integer(), server_default="1"),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_le_org", "learning_events", ["organization_id"])
    op.create_index("ix_le_outcome", "learning_events", ["outcome_id"])
    op.create_index("ix_le_decision", "learning_events", ["decision_id"])
    op.create_index("ix_le_source", "learning_events", ["source_event_id"])

    op.create_table(
        "learning_aggregates",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dimension", sa.String(length=16), nullable=False),
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("successes", sa.Integer(), server_default="0"),
        sa.Column("failures", sa.Integer(), server_default="0"),
        sa.Column("neutrals", sa.Integer(), server_default="0"),
        sa.Column("total", sa.Integer(), server_default="0"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("organization_id", "dimension", "key"),
    )

    op.create_table(
        "learning_patterns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("signal_id", sa.String(length=128), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("samples", sa.Integer(), server_default="0"),
        sa.Column("success_rate", sa.Float(), server_default="0"),
        sa.Column("detail", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_lp_org", "learning_patterns", ["organization_id"])

    op.create_table(
        "learning_inbox",
        sa.Column("consumer", sa.String(length=64), primary_key=True),
        sa.Column("event_id", sa.String(length=128), primary_key=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # Legacy shims (Intelligence V1 compile-compat; never written by V2).
    op.create_table(
        "decision_patterns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("signal_kind", sa.String(length=64), nullable=False),
        sa.Column("samples", sa.Integer(), server_default="0"),
        sa.Column("success_rate", sa.Float(), server_default="0.5"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("organization_id", "signal_kind", name="uq_org_signal_kind"),
    )
    op.create_table(
        "behavior_patterns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scope", sa.String(length=32), server_default="founder"),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("weight", sa.Float(), server_default="0"),
        sa.Column("detail", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )


def downgrade() -> None:
    op.drop_table("behavior_patterns")
    op.drop_table("decision_patterns")
    op.drop_table("learning_inbox")
    op.drop_table("learning_patterns")
    op.drop_table("learning_aggregates")
    op.drop_table("learning_events")
