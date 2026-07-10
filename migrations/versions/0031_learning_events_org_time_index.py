"""learning_events (organization_id, occurred_at, id) index

Revision ID: 0031_learning_events_org_time_index
Revises: 0030_merge_context_onboarding
Create Date: 2026-07-08

Serves the Learning replay rebuild's dominant read pattern: keyset-stream ALL of an org's
learning_events in (occurred_at, id) fold order (repository.stream_events). The single-column
ix on organization_id can't provide that ordering, so without this composite the keyset
paginator re-sorts every page (quadratic at scale). Mirrors State's ix_mem_org_time
(migration 0028). Pure-additive, reversible. Created CONCURRENTLY so it never locks the table
at scale (requires autocommit; no-op safe on an empty/new table).
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0031_learning_events_org_time_index"
down_revision: Union[str, None] = "0030_merge_context_onboarding"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.create_index(
            "ix_learning_events_org_time", "learning_events",
            ["organization_id", "occurred_at", "id"],
            postgresql_concurrently=True, if_not_exists=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            "ix_learning_events_org_time", table_name="learning_events",
            postgresql_concurrently=True, if_exists=True,
        )
