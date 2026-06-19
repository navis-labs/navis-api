"""Identity V2 enterprise ORM: RBAC grants, API keys, service accounts, audit,
token revocation, SSO connections, SCIM groups.

All org-scoped and append-only-friendly. Identity imports only platform.* — never
a business domain (enforced by an AST test).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


class RolePermission(Base):
    """Per-org grant of a permission to a role (augments the code defaults).

    org_id NULL = a system-wide default override. Roles/permissions are strings,
    so changing a role's powers is a row, not a migration.
    """

    __tablename__ = "role_permissions"
    __table_args__ = (
        UniqueConstraint("organization_id", "role", "permission", name="uq_org_role_perm"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    role: Mapped[str] = mapped_column(String(40), nullable=False)
    permission: Mapped[str] = mapped_column(String(64), nullable=False)


class ApiKey(UUIDPrimaryKey, Timestamped, Base):
    """A hashed, scoped API credential. Plaintext is shown once at creation only."""

    __tablename__ = "api_keys"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    hashed_secret: Mapped[str] = mapped_column(String(128), nullable=False)
    scopes: Mapped[list] = mapped_column(JSONB, default=list)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    # When set, this key authenticates a machine identity rather than a person.
    service_account_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("service_accounts.id", ondelete="CASCADE"), nullable=True, index=True
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ServiceAccount(UUIDPrimaryKey, Timestamped, Base):
    """A non-human, org-scoped, permission-scoped principal (workers, integrations)."""

    __tablename__ = "service_accounts"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    scopes: Mapped[list] = mapped_column(JSONB, default=list)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RevokedToken(Base):
    """jti denylist for access-token revocation (logout). `expires_at` lets a
    reaper purge entries after the token would have expired anyway."""

    __tablename__ = "revoked_tokens"

    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    revoked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLog(Base):
    """Append-only record of identity-sensitive actions. Never updated/deleted."""

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    actor_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    actor_kind: Mapped[str] = mapped_column(String(24), default="user")  # user|api_key|service_account|system
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    audit_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class SsoConnection(Base):
    """An org's SSO configuration. Provider-agnostic shell; `config` holds the
    provider-specific settings (OIDC issuer/client, SAML metadata, etc.)."""

    __tablename__ = "sso_connections"
    __table_args__ = (UniqueConstraint("organization_id", "provider", name="uq_org_sso_provider"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(24), nullable=False)  # google|oidc|saml
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ScimGroup(Base):
    """A SCIM-provisioned group (maps to a role/cohort within an org)."""

    __tablename__ = "scim_groups"
    __table_args__ = (UniqueConstraint("organization_id", "external_id", name="uq_org_scim_group"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    members: Mapped[list] = mapped_column(JSONB, default=list)  # user ids
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
