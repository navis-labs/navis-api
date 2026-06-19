"""signal_events production contract (version-aware, multi-tenant, replayable)

Revision ID: 0001_signal_events
Revises:
Create Date: 2026-06-17

Creates the signal_events log with the full SignalEvent contract: deterministic
event_id PK, the (org, user, integration) tenancy triple, event_type +
external_version for version-aware dedup, payload + metadata, and the indexes
consumers/replay rely on.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_signal_events"
down_revision: Union[str, None] = "0000_identity_base"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "signal_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("integration_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("external_id", sa.String(length=512), nullable=False),
        sa.Column("external_version", sa.String(length=128), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["integration_id"], ["integrations.id"], ondelete="SET NULL"),
        sa.UniqueConstraint(
            "organization_id",
            "source",
            "external_id",
            "external_version",
            name="uq_signal_dedupe",
        ),
    )
    op.create_index(
        "ix_signal_org_source_time",
        "signal_events",
        ["organization_id", "source", "occurred_at"],
    )
    op.create_index("ix_signal_integration", "signal_events", ["integration_id"])
    op.create_index("ix_signal_external", "signal_events", ["external_id"])


def downgrade() -> None:
    op.drop_index("ix_signal_external", table_name="signal_events")
    op.drop_index("ix_signal_integration", table_name="signal_events")
    op.drop_index("ix_signal_org_source_time", table_name="signal_events")
    op.drop_table("signal_events")
