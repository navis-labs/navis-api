"""Adaptive Organization V1 — append-only policy governance

Revision ID: 0012_adaptive
Revises: 0011_intelligence
Create Date: 2026-06-19

policy_proposals (deterministic projection over Intelligence), policy_snapshots
(versioned governing policy; one ACTIVE per org), policy_approvals (immutable human
governance record). All org-scoped and partition-ready on organization_id.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012_adaptive"
down_revision: Union[str, None] = "0011_intelligence"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "policy_proposals",
        sa.Column("proposal_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("adaptive_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("proposed_changes", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("proposed_payload", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("supporting_insight_ids", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("supporting_learning_ids", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("policy_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("decided_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("schema_version", sa.Integer(), server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_pp_org", "policy_proposals", ["organization_id"])
    op.create_index("ix_pp_status", "policy_proposals", ["status"])

    op.create_table(
        "policy_snapshots",
        sa.Column("policy_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("policy_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("payload", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("source_intelligence_ids", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_learning_event_ids", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_by", sa.String(length=64), nullable=True),
        sa.Column("approved_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("schema_version", sa.Integer(), server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_ps_org", "policy_snapshots", ["organization_id"])
    op.create_index("ix_ps_status", "policy_snapshots", ["status"])
    # At most one ACTIVE policy per org (governance invariant, enforced in DB).
    op.create_index(
        "uq_ps_one_active",
        "policy_snapshots",
        ["organization_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    op.create_table(
        "policy_approvals",
        sa.Column("approval_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("proposal_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("approved_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_pa_org", "policy_approvals", ["organization_id"])
    op.create_index("ix_pa_proposal", "policy_approvals", ["proposal_id"])


def downgrade() -> None:
    op.drop_table("policy_approvals")
    op.drop_index("uq_ps_one_active", table_name="policy_snapshots")
    op.drop_index("ix_ps_status", table_name="policy_snapshots")
    op.drop_index("ix_ps_org", table_name="policy_snapshots")
    op.drop_table("policy_snapshots")
    op.drop_index("ix_pp_status", table_name="policy_proposals")
    op.drop_index("ix_pp_org", table_name="policy_proposals")
    op.drop_table("policy_proposals")
