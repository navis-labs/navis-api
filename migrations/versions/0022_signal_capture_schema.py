"""Signals — add captured_at + schema_version to signal_events

Revision ID: 0022_signal_capture_schema
Revises: 0021_org_enum_checks
Create Date: 2026-06-22

Additive contract evolution (SignalEvent gained `captured_at` and `schema_version`).
Both are NON-additive to the dedupe identity, so the deterministic event_id is unchanged
and historical rows replay identically.

  * captured_at — when the connector observed the fact (vs occurred_at = when it happened,
    created_at = when the row was inserted). Backfilled from created_at for existing rows
    (the best available capture proxy), then made NOT NULL.
  * schema_version — defaults to 1; existing rows are implicitly v1.

Reversible.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0022_signal_capture_schema"
down_revision: Union[str, None] = "0021_org_enum_checks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "signal_events",
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
    )
    # Add nullable, backfill from the insert time, then enforce NOT NULL.
    op.add_column(
        "signal_events",
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute("UPDATE signal_events SET captured_at = created_at WHERE captured_at IS NULL")
    op.alter_column(
        "signal_events",
        "captured_at",
        nullable=False,
        server_default=sa.text("now()"),
    )


def downgrade() -> None:
    op.drop_column("signal_events", "captured_at")
    op.drop_column("signal_events", "schema_version")
