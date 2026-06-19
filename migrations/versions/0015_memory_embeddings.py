"""ML — append-only versioned embeddings over Memory

Revision ID: 0015_ml_embeddings
Revises: 0014_organization
Create Date: 2026-06-19

memory_embeddings is append-only: a deterministic UUIDv5 PK and a unique
(memory_id, model, version) so re-embedding writes new rows, never mutating old
ones. Vectors are JSONB float arrays (portable + testable today); the pgvector ANN
upgrade is a column-type change behind the same service contract.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0015_ml_embeddings"
down_revision: Union[str, None] = "0014_organization"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("embedding", postgresql.JSONB(), nullable=False),
        sa.Column("entity_type", sa.String(length=40), server_default=""),
        sa.Column("source_text", sa.String(length=2000), server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["memory_id"], ["memory_records.memory_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("memory_id", "model", "version", name="uq_mem_model_version"),
    )
    op.create_index("ix_me_org", "memory_embeddings", ["organization_id"])
    op.create_index("ix_me_memory", "memory_embeddings", ["memory_id"])


def downgrade() -> None:
    op.drop_index("ix_me_memory", table_name="memory_embeddings")
    op.drop_index("ix_me_org", table_name="memory_embeddings")
    op.drop_table("memory_embeddings")
