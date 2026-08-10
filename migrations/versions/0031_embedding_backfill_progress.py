"""ML — resumable embedding backfill progress

Revision ID: 0031_embedding_backfill_progress
Revises: 0030_memory_outbox
Create Date: 2026-08-06

Adds `embedding_backfill_progress`: one durable keyset cursor + counters per
(organization, model, version) index.

Re-embedding the corpus is a long, interruptible job. The previous backfill had
no cursor — it issued `SELECT ... LIMIT n` with no ORDER BY and no offset, so it
re-scanned the same n rows on every call and could never advance past its own
page limit (measured: 2,078 of 3,078 records in the reference org were
unreachable). It was also unobservable: nothing recorded how far a migration had
got, or whether it had finished.

`cursor_memory_id` is the keyset position — every memory_id <= cursor has been
considered for that index. It is advanced in the SAME transaction as the vectors
it accounts for, so an interruption can only replay a page, never skip one.

Additive; no existing table is touched; fully reversible.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0031_embedding_backfill_progress"
down_revision: Union[str, Sequence[str], None] = "0030_memory_outbox"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "embedding_backfill_progress",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        # NULL = not started. Every memory_id <= this has been considered.
        sa.Column("cursor_memory_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("embedded_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skipped_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        # NULL with a stale updated_at is the "migration stalled" signal.
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "organization_id", "model", "version", name="uq_backfill_org_model_version"
        ),
    )
    op.create_index(
        "ix_embedding_backfill_progress_organization_id",
        "embedding_backfill_progress",
        ["organization_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_embedding_backfill_progress_organization_id",
        table_name="embedding_backfill_progress",
    )
    op.drop_table("embedding_backfill_progress")
