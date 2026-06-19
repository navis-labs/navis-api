"""execution_dead_letters + execution_inbox (Execution V2 hardening)

Revision ID: 0008_execution_hardening
Revises: 0007_execution
Create Date: 2026-06-19

DLQ for exhausted executions (replayable) + inbox for idempotent consumers.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_execution_hardening"
down_revision: Union[str, None] = "0007_execution"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "execution_dead_letters",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("execution_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(length=40), nullable=True),
        sa.Column("action_type", sa.String(length=40), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0"),
        sa.Column("payload", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("replayed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["execution_id"], ["execution_requests.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_dlq_org", "execution_dead_letters", ["organization_id"])
    op.create_index("ix_dlq_execution", "execution_dead_letters", ["execution_id"])
    op.create_index("ix_dlq_unresolved", "execution_dead_letters", ["resolved_at"])

    op.create_table(
        "execution_inbox",
        sa.Column("consumer", sa.String(length=64), primary_key=True),
        sa.Column("event_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("execution_inbox")
    op.drop_table("execution_dead_letters")
