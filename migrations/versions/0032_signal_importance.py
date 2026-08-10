"""Signals — deterministic importance score per signal, per policy version

Revision ID: 0032_signal_importance
Revises: 0031_embedding_backfill_progress
Create Date: 2026-08-06

Adds `signal_importance`: one score + classification per (signal_event, version).

A founder inbox is ~90% newsletters, receipts and marketing, but "automated" is
NOT a proxy for "unimportant" — in the reference corpus every calendar RSVP and
every invoice is automated, so excluding automated mail would delete the
meetings. Importance is therefore scored rather than filtered, and recorded
here.

Kept OUT of `signal_events` on purpose: that table is the immutable, replayable
log, and bolting a recomputed column onto it would break both properties. This
mirrors `memory_embeddings` — a derived, versioned artifact rebuildable from the
source at any time, where two policy versions can coexist so a new policy is
computed and compared before anything consumes it.

Additive; no existing table is touched; fully reversible.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0032_signal_importance"
down_revision: Union[str, Sequence[str], None] = "0031_embedding_backfill_progress"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "signal_importance",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "signal_event_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("signal_events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("classification", sa.String(length=40), nullable=False),
        sa.Column(
            "features",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "signal_event_id", "version", name="uq_importance_signal_version"
        ),
    )
    op.create_index(
        "ix_signal_importance_organization_id", "signal_importance", ["organization_id"]
    )
    op.create_index(
        "ix_signal_importance_signal_event_id", "signal_importance", ["signal_event_id"]
    )
    op.create_index(
        "ix_signal_importance_org_score", "signal_importance", ["organization_id", "score"]
    )


def downgrade() -> None:
    op.drop_index("ix_signal_importance_org_score", table_name="signal_importance")
    op.drop_index("ix_signal_importance_signal_event_id", table_name="signal_importance")
    op.drop_index("ix_signal_importance_organization_id", table_name="signal_importance")
    op.drop_table("signal_importance")
