"""Separate the two provenance ID namespaces on execution_requests.

`reasoning_signal_ids` must contain ONLY `reasoning_signals.signal_id` values:
Learning aggregates on them (learning/aggregates.py) and Intelligence cites them
as `supporting_reasoning_signal_ids`, so anything from another namespace is a
reference that can never resolve.

The chat scheduling path was writing `signal_events.id` values into that field —
`active_signal_id` is looked up as a SignalEventRecord, and
`source_signal_event_id` is derived from it. This adds the column those values
actually belong in, so source-event provenance is preserved rather than dropped.

Additive and non-destructive: one nullable JSONB column defaulting to `[]`.
Existing rows read back as an empty list, so every current reader is unaffected.

NO BACKFILL, deliberately. Checked against production before writing this: all
four chat-origin rows carry `reasoning_signal_ids = []` (the bug only fires when
an active signal is present), and the single proactive row carries a genuine
ReasoningSignal id. There is nothing mixed to separate, and guessing which
namespace a historical id belonged to would be fabricating provenance.

Revision ID: 0037_execution_source_events
Revises: 0036_reasoning_signals
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0037_execution_source_events"
down_revision = "0036_reasoning_signals"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "execution_requests",
        sa.Column(
            "source_signal_event_ids",
            postgresql.JSONB(),
            nullable=False,
            server_default="[]",
        ),
    )


def downgrade() -> None:
    op.drop_column("execution_requests", "source_signal_event_ids")
