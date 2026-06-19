"""outcomes (append-only, lineage) + observation_inbox (Observation V2)

Revision ID: 0009_observation
Revises: 0008_execution_hardening
Create Date: 2026-06-19

Append-only outcome log with deterministic outcome_id (idempotent capture), full
decision/reasoning lineage, occurred_at vs recorded_at, and an inbox for
at-least-once idempotent consumption of Execution events.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009_observation"
down_revision: Union[str, None] = "0008_execution_hardening"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "outcomes",
        sa.Column("outcome_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("execution_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("decision_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("reasoning_signal_ids", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("reasoning_version", sa.Integer(), nullable=True),
        sa.Column("state_watermark", sa.String(length=64), nullable=True),
        sa.Column("provider_reference", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("outcome_type", sa.String(length=40), nullable=False),
        sa.Column("payload", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("schema_version", sa.Integer(), server_default="1"),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_outcome_org", "outcomes", ["organization_id"])
    op.create_index("ix_outcome_execution", "outcomes", ["execution_id"])
    op.create_index("ix_outcome_decision", "outcomes", ["decision_id"])
    op.create_index("ix_outcome_type", "outcomes", ["outcome_type"])
    op.create_index("ix_outcome_source_event", "outcomes", ["source_event_id"])
    # Provider-reference lookup (webhooks) — GIN on the JSONB column.
    op.create_index(
        "ix_outcome_provider_ref", "outcomes", ["provider_reference"], postgresql_using="gin"
    )

    op.create_table(
        "observation_inbox",
        sa.Column("consumer", sa.String(length=64), primary_key=True),
        sa.Column("event_id", sa.String(length=128), primary_key=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("observation_inbox")
    op.drop_table("outcomes")
