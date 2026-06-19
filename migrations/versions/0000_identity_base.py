"""Identity base — the foundational tenancy tables

Revision ID: 0000_identity_base
Revises: None
Create Date: 2026-06-19

Creates organizations/users/teams/memberships/sessions/integrations. These tables
were historically materialized only by Base.metadata.create_all, so `alembic
upgrade head` could not bootstrap an empty database (0001 FK-references
organizations.id before anything created it). This migration is the true base of
the chain; 0001 now depends on it. Shapes mirror the current ORM in
navis/identity/models.py exactly.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0000_identity_base"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_ROLE = sa.Enum(
    "founder", "ceo", "coo", "chief_of_staff", "employee", "admin", name="role"
)
_PROVIDER = sa.Enum(
    "gmail", "google_calendar", "google_meet", "slack", "notion", "drive",
    name="integrationprovider",
)


def _ts(*cols):
    return (
        *cols,
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def upgrade() -> None:
    op.create_table(
        "organizations",
        *_ts(
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("slug", sa.String(length=255), nullable=False, unique=True),
            sa.Column("domain", sa.String(length=255), nullable=True),
        ),
    )

    op.create_table(
        "users",
        *_ts(
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("email", sa.String(length=320), nullable=False),
            sa.Column("full_name", sa.String(length=255), nullable=True),
            sa.Column("hashed_password", sa.String(length=255), nullable=True),
            sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
            sa.Column("is_verified", sa.Boolean(), server_default=sa.false(), nullable=False),
            sa.Column("google_sub", sa.String(length=255), nullable=True),
        ),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_google_sub", "users", ["google_sub"])

    op.create_table(
        "teams",
        *_ts(
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("name", sa.String(length=255), nullable=False),
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_teams_organization_id", "teams", ["organization_id"])

    op.create_table(
        "memberships",
        *_ts(
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("role", _ROLE, nullable=True),
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "organization_id", name="uq_user_org"),
    )
    op.create_index("ix_memberships_user_id", "memberships", ["user_id"])
    op.create_index("ix_memberships_organization_id", "memberships", ["organization_id"])

    op.create_table(
        "sessions",
        *_ts(
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("refresh_token_hash", sa.String(length=255), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("revoked", sa.Boolean(), server_default=sa.false(), nullable=False),
            sa.Column("user_agent", sa.String(length=512), nullable=True),
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    op.create_table(
        "integrations",
        *_ts(
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("provider", _PROVIDER, nullable=False),
            sa.Column("external_account", sa.String(length=320), nullable=True),
            sa.Column("access_token", sa.String(), nullable=True),
            sa.Column("refresh_token", sa.String(), nullable=True),
            sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("scopes", sa.String(), nullable=True),
            sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
            sa.Column("sync_state", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("organization_id", "provider", "external_account", name="uq_org_provider_account"),
    )
    op.create_index("ix_integrations_organization_id", "integrations", ["organization_id"])


def downgrade() -> None:
    op.drop_table("integrations")
    op.drop_table("sessions")
    op.drop_table("memberships")
    op.drop_table("teams")
    op.drop_index("ix_users_google_sub", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    op.drop_table("organizations")
    sa.Enum(name="integrationprovider").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="role").drop(op.get_bind(), checkfirst=True)
