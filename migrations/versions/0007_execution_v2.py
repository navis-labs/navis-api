"""execution_requests + execution_outbox (Execution V2)

Revision ID: 0007_execution
Revises: 0006_daily_briefs
Create Date: 2026-06-19

Command aggregate with deterministic idempotency (unique on org + key) and full
decision/reasoning lineage, plus a transactional outbox for persist-first /
publish-later events.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_execution"
down_revision: Union[str, None] = "0006_daily_briefs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "execution_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("decision_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("reasoning_signal_ids", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("reasoning_version", sa.Integer(), nullable=True),
        sa.Column("state_watermark", sa.String(length=64), nullable=True),
        sa.Column("action_type", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("idempotency_key", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("approved_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provider_reference", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("attempts", sa.Integer(), server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["approved_by"], ["users.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("organization_id", "idempotency_key", name="uq_execution_idempotency"),
    )
    op.create_index("ix_exec_org", "execution_requests", ["organization_id"])
    op.create_index("ix_exec_status", "execution_requests", ["status"])
    op.create_index("ix_exec_decision", "execution_requests", ["decision_id"])

    op.create_table(
        "execution_outbox",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("execution_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(length=40), nullable=False),
        sa.Column("payload", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["execution_id"], ["execution_requests.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_outbox_org", "execution_outbox", ["organization_id"])
    op.create_index("ix_outbox_execution", "execution_outbox", ["execution_id"])
    op.create_index("ix_outbox_pending", "execution_outbox", ["published_at"])


def downgrade() -> None:
    op.drop_table("execution_outbox")
    op.drop_table("execution_requests")
