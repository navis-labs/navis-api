"""Intelligence V2 — append-only organizational insights

Revision ID: 0011_intelligence
Revises: 0010_learning
Create Date: 2026-06-19

A deterministic, append-only projection over Learning. Each row is an immutable
fact the organization has learned about itself, keyed by a deterministic UUIDv5
(org|type|key|learning_version) so re-projecting the same Learning is idempotent.
Fully rebuildable: drop the table, re-project Learning, get identical rows.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_intelligence"
down_revision: Union[str, None] = "0010_learning"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "intelligence_insights",
        sa.Column("insight_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("insight_type", sa.String(length=40), nullable=False),
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("sample_size", sa.Integer(), server_default="0"),
        sa.Column("confidence", sa.Float(), server_default="0"),
        sa.Column(
            "supporting_learning_event_ids",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("learning_version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_ii_org", "intelligence_insights", ["organization_id"])
    op.create_index("ix_ii_type", "intelligence_insights", ["insight_type"])


def downgrade() -> None:
    op.drop_index("ix_ii_type", table_name="intelligence_insights")
    op.drop_index("ix_ii_org", table_name="intelligence_insights")
    op.drop_table("intelligence_insights")
