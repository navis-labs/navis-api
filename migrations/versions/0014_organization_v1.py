"""Organization V1 — profile, lifecycle, onboarding, entitlements, flags, metrics

Revision ID: 0014_organization
Revises: 0013_identity
Create Date: 2026-06-19

Organization owns "what company is this and what can it do". All org-scoped and
partition-ready. Users/roles/keys/SSO stay in Identity.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0014_organization"
down_revision: Union[str, None] = "0013_identity"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "organization_profiles",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("company_name", sa.String(length=255), nullable=True),
        sa.Column("website", sa.String(length=255), nullable=True),
        sa.Column("industry", sa.String(length=128), nullable=True),
        sa.Column("employee_count", sa.Integer(), nullable=True),
        sa.Column("hq_country", sa.String(length=2), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "organization_lifecycles",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="trial"),
        sa.Column("plan", sa.String(length=16), nullable=False, server_default="free"),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("churned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_ol_status", "organization_lifecycles", ["status"])

    op.create_table(
        "organization_onboarding",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("stage", sa.String(length=32), nullable=False, server_default="created"),
        sa.Column("signals_count", sa.Integer(), server_default="0"),
        sa.Column("memory_count", sa.Integer(), server_default="0"),
        sa.Column("stage_timestamps", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_oo_stage", "organization_onboarding", ["stage"])

    op.create_table(
        "organization_entitlements",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("capability", sa.String(length=32), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("organization_id", "capability", name="uq_org_capability"),
    )
    op.create_index("ix_oe_org", "organization_entitlements", ["organization_id"])

    op.create_table(
        "organization_feature_flags",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("flag", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), server_default="release"),
        sa.Column("enabled", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("organization_id", "flag", name="uq_org_flag"),
    )
    op.create_index("ix_off_org", "organization_feature_flags", ["organization_id"])

    op.create_table(
        "organization_metrics",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("ttfi_seconds", sa.Integer(), nullable=True),
        sa.Column("activation_score", sa.Float(), server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )


def downgrade() -> None:
    op.drop_table("organization_metrics")
    op.drop_index("ix_off_org", table_name="organization_feature_flags")
    op.drop_table("organization_feature_flags")
    op.drop_index("ix_oe_org", table_name="organization_entitlements")
    op.drop_table("organization_entitlements")
    op.drop_index("ix_oo_stage", table_name="organization_onboarding")
    op.drop_table("organization_onboarding")
    op.drop_index("ix_ol_status", table_name="organization_lifecycles")
    op.drop_table("organization_lifecycles")
    op.drop_table("organization_profiles")
