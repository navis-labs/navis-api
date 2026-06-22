"""Signals models hardening: drop unused ix_signal_external + DLQ user_id (GDPR)

Revision ID: 0023_signal_models_hardening
Revises: 0022_signal_capture_schema
Create Date: 2026-06-22

Two safe, backward-compatible changes scoped to the signals event/DLQ tables:

  1. DROP ix_signal_external — a bare 512-char-key B-tree on signal_events.external_id that
     no query uses (dedup goes through uq_signal_dedupe; reads are by PK or org+time). At
     100B rows it is pure write amplification + storage. Reversible.

  2. ADD signal_dead_letters.user_id — the DLQ payload is a second copy of the failed
     SignalEvent (may contain PII). A user_id column makes that copy targetable for GDPR
     erasure without scanning JSONB. Backfilled from the stored payload (which is the full
     event dict); FK users ON DELETE SET NULL, mirroring signal_events.user_id.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0023_signal_models_hardening"
down_revision: Union[str, None] = "0022_signal_capture_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) Drop the unused index.
    op.drop_index("ix_signal_external", table_name="signal_events")

    # 2) DLQ user_id for GDPR targeting, backfilled from the stored event payload.
    op.add_column(
        "signal_dead_letters",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_signal_dlq_user_id", "signal_dead_letters", "users",
        ["user_id"], ["id"], ondelete="SET NULL",
    )
    # payload is the full event dict (item=event.to_dict()) — pull user_id where it is a
    # well-formed uuid; leave NULL otherwise (fetch-stage rows, malformed legacy payloads).
    op.execute(
        """
        UPDATE signal_dead_letters
           SET user_id = (payload->>'user_id')::uuid
         WHERE payload ? 'user_id'
           AND payload->>'user_id' ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        """
    )


def downgrade() -> None:
    op.drop_constraint("fk_signal_dlq_user_id", "signal_dead_letters", type_="foreignkey")
    op.drop_column("signal_dead_letters", "user_id")
    op.create_index("ix_signal_external", "signal_events", ["external_id"])
