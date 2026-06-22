"""Signals DLQ — replay accounting (attempts + last_error)

Revision ID: 0025_dlq_replay_accounting
Revises: 0024_signal_outbox
Create Date: 2026-06-22

Adds `attempts` and `last_error` to signal_dead_letters so DLQ replay can cap retries
(poison-loop guard: after N failed re-ingestions the row is resolved/permanently
dead-lettered) and record the most recent replay error for observability. Additive,
backward-compatible (existing rows default attempts=0), reversible.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0025_dlq_replay_accounting"
down_revision: Union[str, None] = "0024_signal_outbox"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "signal_dead_letters",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "signal_dead_letters",
        sa.Column("last_error", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("signal_dead_letters", "last_error")
    op.drop_column("signal_dead_letters", "attempts")
