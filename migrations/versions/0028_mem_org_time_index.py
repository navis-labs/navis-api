"""memory_records (organization_id, event_timestamp, memory_id) index

Revision ID: 0028_mem_org_time_index
Revises: 0027_signal_channel_id
Create Date: 2026-06-24

Serves the State V2 projection's dominant read pattern: stream ALL of an org's memory_records
in (event_timestamp, memory_id) fold order, regardless of entity_type. The existing
ix_mem_org_type_time leads with entity_type and so can't serve that ordering. Pure-additive,
reversible. Created CONCURRENTLY so it never locks the table at scale (requires autocommit;
no-op safe on an empty/new table).
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0028_mem_org_time_index"
down_revision: Union[str, None] = "0027_signal_channel_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.create_index(
            "ix_mem_org_time", "memory_records",
            ["organization_id", "event_timestamp", "memory_id"],
            postgresql_concurrently=True, if_not_exists=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index("ix_mem_org_time", table_name="memory_records", postgresql_concurrently=True, if_exists=True)
