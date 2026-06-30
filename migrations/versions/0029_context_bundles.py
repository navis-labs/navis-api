"""context_bundles — content-addressed ContextBundle cache (Context layer)

Revision ID: 0029_context_bundles
Revises: 0028_mem_org_time_index
Create Date: 2026-06-30

A cache, never a source of truth — every bundle is rebuildable from State (rebuildable
from Memory). Keyed by `context_id` (a digest of all assembly inputs), so the same inputs
map to the same row and writes are idempotent. Tenant-scoped via an org FK with ON DELETE
CASCADE. Pure-additive and reversible; touches no existing table.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0029_context_bundles"
down_revision: Union[str, None] = "0028_mem_org_time_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "context_bundles",
        sa.Column("context_id", sa.String(length=64), nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("context_version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("bundle", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("context_id"),
    )
    op.create_index(
        "ix_context_bundles_org", "context_bundles", ["organization_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_context_bundles_org", table_name="context_bundles")
    op.drop_table("context_bundles")
