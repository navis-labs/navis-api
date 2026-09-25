"""Identity V2 enterprise ORM: RBAC grants, API keys, service accounts, audit,
token revocation, SSO connections, SCIM groups.

All org-scoped and append-only-friendly. Identity imports only platform.* — never
a business domain (enforced by an AST test).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column

from navis.identity.agents import AGENT_ENVIRONMENTS, AGENT_RUNTIMES, AgentEnvironment, AgentRuntime
from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


def _one_of(column: str, values: frozenset[str]) -> str:
    """A SQL `IN (...)` predicate over a closed vocabulary, quoted deterministically.

    Sorted so the constraint's rendered SQL is stable across processes — an
    unordered set would make Alembic's autogenerate see a difference on every run.
    """
    literals = ", ".join(f"'{v}'" for v in sorted(values))
    return f"{column} IN ({literals})"

# SQLAlchemy does NOT detect in-place mutation of a plain JSONB dict/list
# (`row.config["x"] = y` silently won't persist). Wrapping the type registers a
# change listener so in-place edits flush correctly. Today every write reassigns,
# so this is defense-in-depth against a real future foot-gun — no schema change.
_JSONB_DICT = MutableDict.as_mutable(JSONB)
_JSONB_LIST = MutableList.as_mutable(JSONB)


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
    scopes: Mapped[list] = mapped_column(_JSONB_LIST, default=list)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    # When set, this key authenticates a machine identity rather than a person.
    service_account_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("service_accounts.id", ondelete="CASCADE"), nullable=True, index=True
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: When this credential stops authenticating. NULL means it does not expire,
    #: which is what every credential minted before 0052 is.
    #:
    #: SEPARATE FROM `revoked_at` ON PURPOSE. `revoked_at` is a decision somebody
    #: made; this is a limit chosen when the credential was minted. A key can be
    #: both, an audit needs to tell them apart, and authentication refuses
    #: either. See `credential_status_of` for the one derivation both the API and
    #: the console read.
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ServiceAccount(UUIDPrimaryKey, Timestamped, Base):
    """A non-human, org-scoped, permission-scoped principal (workers, integrations).

    THIS IS THE AGENT ROW. `/api/v1/developer/agents` is a product-facing view of
    this table and creates nothing else — see `navis/identity/agents.py` for why
    a separate `agents` table would be a second machine identity rather than a
    new feature. The three descriptive columns below (0046) are what that view
    needs and the authorization path deliberately never reads.
    """

    __tablename__ = "service_accounts"
    __table_args__ = (
        # Closed vocabularies, enforced by the DATABASE and not only by the edge.
        # The edge validates too, but the edge is one of several writers over the
        # life of a table (a migration, a fixture, an operator's psql session),
        # and a free-text tier or runtime cannot be grouped or filtered later.
        CheckConstraint(_one_of("environment", AGENT_ENVIRONMENTS), name="ck_sa_environment"),
        CheckConstraint(_one_of("runtime", AGENT_RUNTIMES), name="ck_sa_runtime"),
        # The hot agent query is "this tenant's agents, newest first" — the list
        # every console page load and every `GET /developer/agents` runs. Leading
        # with organization_id keeps it tenant-keyed, so one organization's fleet
        # size never affects another's scan.
        Index(
            "ix_service_accounts_org_created",
            "organization_id",
            sa_text("created_at DESC"),
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    scopes: Mapped[list] = mapped_column(_JSONB_LIST, default=list)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # ── Descriptive, never authoritative (0046) ─────────────────────────────
    # Server defaults as well as Python defaults: rows written before 0046 are
    # back-filled by the migration, and a raw INSERT that omits them gets the
    # same values the ORM would have supplied rather than a NOT NULL violation.
    description: Mapped[str] = mapped_column(
        String(500), nullable=False, default="", server_default=""
    )
    environment: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=AgentEnvironment.DEVELOPMENT.value,
        server_default=AgentEnvironment.DEVELOPMENT.value,
    )
    runtime: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AgentRuntime.CUSTOM.value,
        server_default=AgentRuntime.CUSTOM.value,
    )


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
    audit_metadata: Mapped[dict] = mapped_column(_JSONB_DICT, default=dict)
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
    config: Mapped[dict] = mapped_column(_JSONB_DICT, default=dict)
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
    members: Mapped[list] = mapped_column(_JSONB_LIST, default=list)  # user ids
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
