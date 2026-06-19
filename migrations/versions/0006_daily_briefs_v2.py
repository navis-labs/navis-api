"""daily_briefs — Decision V2 (deterministic id + provenance, disposable cache)

Revision ID: 0006_daily_briefs
Revises: 0005_state_snapshots
Create Date: 2026-06-19

Deterministic UUIDv5 id (no uuid4), keyed by (org, reasoning_version,
state_watermark) so replaying Reasoning rebuilds the identical brief. A cache —
drop it and rebuild from Reasoning.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_daily_briefs"
down_revision: Union[str, None] = "0005_state_snapshots"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "daily_briefs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("reasoning_version", sa.Integer(), nullable=False),
        sa.Column("state_watermark", sa.String(length=64), nullable=True),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decisions", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_brief_org", "daily_briefs", ["organization_id"])
    op.create_index("ix_brief_watermark", "daily_briefs", ["state_watermark"])
    op.create_index("ix_brief_as_of", "daily_briefs", ["as_of"])


def downgrade() -> None:
    op.drop_index("ix_brief_as_of", table_name="daily_briefs")
    op.drop_index("ix_brief_watermark", table_name="daily_briefs")
    op.drop_index("ix_brief_org", table_name="daily_briefs")
    op.drop_table("daily_briefs")
