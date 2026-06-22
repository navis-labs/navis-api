"""Signals — transactional outbox for pointer events (Issue B fix)

Revision ID: 0024_signal_outbox
Revises: 0023_signal_models_hardening
Create Date: 2026-06-22

Adds `signal_outbox`. The ingestion path writes a pointer row here IN THE SAME transaction
as the signal_events insert, and a relay worker publishes pending rows
(`published_at IS NULL`) to the event bus at-least-once. Replaces the commit-then-publish
dual write that could drop downstream processing on a crash. Additive; no existing table
changes; reversible.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0024_signal_outbox"
down_revision: Union[str, None] = "0023_signal_models_hardening"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "signal_outbox",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True,
        ),
        sa.Column("aggregate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("source", sa.String(32), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_signal_outbox_organization_id", "signal_outbox", ["organization_id"])
    # Partial index over only the UNPUBLISHED backlog — keeps the relay drain query tiny.
    op.create_index(
        "ix_signal_outbox_unpublished", "signal_outbox", ["created_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_signal_outbox_unpublished", table_name="signal_outbox")
    op.drop_index("ix_signal_outbox_organization_id", table_name="signal_outbox")
    op.drop_table("signal_outbox")
