"""Execution — reviewable email drafts for proactive action candidates

Revision ID: 0034_action_drafts
Revises: 0033_execution_origin
Create Date: 2026-08-09

Adds `action_drafts`: one reviewable document per proactive ExecutionRequest.

WHY NOT IN `execution_requests.payload`
---------------------------------------
The execution idempotency key is a hash of (org, decision, action_type,
PAYLOAD). If the draft body lived in the payload, every edit the founder made
would change that hash — and the next propose pass would create a DIFFERENT
ExecutionRequest. The Action Center would grow one card per edit and the
"one obligation, one action" identity would be gone.

Keeping the document in its own row, keyed BY execution_id, means the action's
identity is fixed at proposal time and editing is a plain UPDATE. At approval
the draft is merged INTO the payload, which is safe because `idempotency_key` is
computed once at INSERT and never recomputed — so the merge cannot move the row,
and afterwards the payload is the audit record of exactly what was dispatched.

The draft lifecycle is deliberately separate from ExecutionStatus: "the user is
editing the body" is not a state a machine designed to prevent double side
effects should have to model.

Additive and reversible; no existing table is touched.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0034_action_drafts"
down_revision: Union[str, Sequence[str], None] = "0033_execution_origin"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "action_drafts",
        # The execution row IS the action's identity, so it is also the PK here:
        # one action, at most one draft, no join key to keep in sync.
        sa.Column(
            "execution_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("execution_requests.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("state", sa.String(length=16), nullable=False, server_default="pending"),
        # --- the document ---
        sa.Column("subject", sa.String(length=500), nullable=False, server_default=""),
        sa.Column("body", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "recipients", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        # --- reply threading ---
        sa.Column("thread_id", sa.String(length=256), nullable=True),
        sa.Column("in_reply_to_message_id", sa.String(length=512), nullable=True),
        # --- why it exists (shown to the user; facts only, never reasoning) ---
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "evidence", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        # --- staleness ---
        sa.Column("thread_watermark", sa.String(length=512), nullable=True),
        sa.Column("stale_reason", sa.String(length=40), nullable=True),
        # --- provenance ---
        sa.Column(
            "generated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("edited_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "edited_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("generator", sa.String(length=40), nullable=False, server_default=""),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    # The Action Center asks "which of this org's drafts still need review", so
    # the index is on the pair that query filters by, tenant-first.
    op.create_index(
        "ix_action_drafts_org_state", "action_drafts", ["organization_id", "state"]
    )


def downgrade() -> None:
    op.drop_index("ix_action_drafts_org_state", table_name="action_drafts")
    op.drop_table("action_drafts")
