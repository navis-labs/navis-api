"""signal_events.channel_id column + channel-scoped index

Revision ID: 0027_signal_channel_id
Revises: 0026_signal_dedupe_integration
Create Date: 2026-06-23

Adds a nullable `channel_id` column (denormalized from metadata) + an index on
(organization_id, channel_id, occurred_at). Channel-scoped retrieval — "this channel over
time" — is a constant access pattern for Memory/State/Knowledge-Graph/Reasoning; without a
real column it would JSONB-scan the whole table, which is unacceptable at 100B rows. Done
pre-launch while the table is small. Additive + nullable (NULL for non-channel sources), so
existing rows are untouched and it's reversible.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0027_signal_channel_id"
down_revision: Union[str, None] = "0026_signal_dedupe_integration"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("signal_events", sa.Column("channel_id", sa.String(length=128), nullable=True))
    op.create_index(
        "ix_signal_org_channel_time", "signal_events",
        ["organization_id", "channel_id", "occurred_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_signal_org_channel_time", table_name="signal_events")
    op.drop_column("signal_events", "channel_id")
