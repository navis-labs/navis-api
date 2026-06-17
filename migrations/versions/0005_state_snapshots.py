"""state_snapshots — disposable cache of the State V2 projection

Revision ID: 0005_state_snapshots
Revises: 0004_memory_records
Create Date: 2026-06-17

A cache, not a source of truth. Keyed by (organization_id, projection_version);
upserted. Carries the watermark cursor into Memory + the full serialized
projection so State is rebuildable and incrementally updatable. Dropping this
table loses nothing — it rebuilds deterministically from memory_records.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_state_snapshots"
down_revision: Union[str, None] = "0004_memory_records"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "state_snapshots",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("projection_version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("watermark_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("watermark_memory_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("full_projection", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("organization_id", "projection_version"),
    )


def downgrade() -> None:
    op.drop_table("state_snapshots")
