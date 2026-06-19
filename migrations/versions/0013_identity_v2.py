"""Identity V2 — RBAC, API keys, service accounts, audit, token revocation, SSO, SCIM

Revision ID: 0013_identity
Revises: 0012_adaptive
Create Date: 2026-06-19

Adds the enterprise identity surface and the users.tokens_valid_from bulk-revoke
watermark. All org-scoped; audit_logs is append-only.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0013_identity"
down_revision: Union[str, None] = "0012_adaptive"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("tokens_valid_from", sa.DateTime(timezone=True), nullable=True))

    op.create_table(
        "role_permissions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("role", sa.String(length=40), nullable=False),
        sa.Column("permission", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("organization_id", "role", "permission", name="uq_org_role_perm"),
    )
    op.create_index("ix_rp_org", "role_permissions", ["organization_id"])

    op.create_table(
        "service_accounts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("scopes", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_sa_org", "service_accounts", ["organization_id"])

    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("prefix", sa.String(length=16), nullable=False),
        sa.Column("hashed_secret", sa.String(length=128), nullable=False),
        sa.Column("scopes", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("service_account_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["service_account_id"], ["service_accounts.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_ak_org", "api_keys", ["organization_id"])
    op.create_index("ix_ak_prefix", "api_keys", ["prefix"])
    op.create_index("ix_ak_sa", "api_keys", ["service_account_id"])

    op.create_table(
        "revoked_tokens",
        sa.Column("jti", sa.String(length=64), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_rt_user", "revoked_tokens", ["user_id"])

    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("actor_kind", sa.String(length=24), server_default="user"),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("target_id", sa.String(length=128), nullable=True),
        sa.Column("audit_metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_al_org", "audit_logs", ["organization_id"])
    op.create_index("ix_al_actor", "audit_logs", ["actor_id"])
    op.create_index("ix_al_action", "audit_logs", ["action"])

    op.create_table(
        "sso_connections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(length=24), nullable=False),
        sa.Column("config", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("enabled", sa.Boolean(), server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("organization_id", "provider", name="uq_org_sso_provider"),
    )
    op.create_index("ix_sso_org", "sso_connections", ["organization_id"])

    op.create_table(
        "scim_groups",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("members", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("organization_id", "external_id", name="uq_org_scim_group"),
    )
    op.create_index("ix_scim_org", "scim_groups", ["organization_id"])


def downgrade() -> None:
    op.drop_table("scim_groups")
    op.drop_table("sso_connections")
    op.drop_table("audit_logs")
    op.drop_table("revoked_tokens")
    op.drop_table("api_keys")
    op.drop_table("service_accounts")
    op.drop_table("role_permissions")
    op.drop_column("users", "tokens_valid_from")
