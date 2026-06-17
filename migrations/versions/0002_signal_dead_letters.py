"""signal_dead_letters — PostgreSQL-backed dead letter queue

Revision ID: 0002_signal_dead_letters
Revises: 0001_signal_events
Create Date: 2026-06-17

PostgreSQL becomes the source of truth for poison signals (Redis is only a
secondary alert). Replay reads from this table.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_signal_dead_letters"
down_revision: Union[str, None] = "0001_signal_events"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "signal_dead_letters",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("integration_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("dedupe_key", sa.String(length=800), nullable=True),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("replayed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["integration_id"], ["integrations.id"], ondelete="SET NULL"),
    )
    op.create_index(
        "ix_dlq_org_unresolved", "signal_dead_letters", ["organization_id", "resolved_at"]
    )
    op.create_index("ix_dlq_integration", "signal_dead_letters", ["integration_id"])


def downgrade() -> None:
    op.drop_index("ix_dlq_integration", table_name="signal_dead_letters")
    op.drop_index("ix_dlq_org_unresolved", table_name="signal_dead_letters")
    op.drop_table("signal_dead_letters")
