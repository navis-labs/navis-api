"""Memory — transactional outbox for domain events

Revision ID: 0030_memory_outbox
Revises: 0029_identity_res_dismissals
Create Date: 2026-07-07

Adds `memory_outbox`. The memory ingestion path writes a pointer row here IN THE SAME
transaction as the memory_records append, and a relay worker publishes pending rows
(``published_at IS NULL``) to the event bus at-least-once. This is what advances the
onboarding funnel from FIRST_SIGNAL to MEMORY_READY (and further downstream stages) —
prior to this, memory records were written but their derived domain events
(PersonDetected/CommitmentDetected/…) were never emitted, so the funnel stalled.

Mirrors ``signal_outbox`` (0024). Additive; no existing table changes; reversible.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0030_memory_outbox"
down_revision: Union[str, Sequence[str], None] = "0029_identity_res_dismissals"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_outbox",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True,
        ),
        sa.Column("aggregate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("entity_type", sa.String(40), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_memory_outbox_organization_id", "memory_outbox", ["organization_id"])
    # Partial index over only the UNPUBLISHED backlog — keeps the relay drain query tiny.
    op.create_index(
        "ix_memory_outbox_unpublished", "memory_outbox", ["created_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_memory_outbox_unpublished", table_name="memory_outbox")
    op.drop_index("ix_memory_outbox_organization_id", table_name="memory_outbox")
    op.drop_table("memory_outbox")
