"""Identity ORM models: User, Organization, Team, Membership, Session, Integration."""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


class Role(str, enum.Enum):
    """Org-level role. Future-rich set mirrors the founder-led wedge."""

    FOUNDER = "founder"
    CEO = "ceo"
    COO = "coo"
    CHIEF_OF_STAFF = "chief_of_staff"
    EMPLOYEE = "employee"
    ADMIN = "admin"


class IntegrationProvider(str, enum.Enum):
    GMAIL = "gmail"
    GOOGLE_CALENDAR = "google_calendar"
    GOOGLE_MEET = "google_meet"
    SLACK = "slack"
    NOTION = "notion"
    DRIVE = "drive"
    # ---- Microsoft 365 (Entra) family ----
    # One row per RESOURCE, not one per grant, exactly as the Google family
    # works: a single Entra consent covers all of them, but the sync
    # orchestrator dispatches per `Integration.provider` and the frontend's
    # Connections page matches cards on the provider string. Aliasing
    # outlook_calendar back onto the outlook row would make Calendar report
    # the mailbox's sync timestamp as its own.
    #
    # These values are members of a NATIVE PostgreSQL enum — adding one needs
    # `ALTER TYPE ... ADD VALUE` (migration 0046), which is why they must be
    # deployed a release ahead of any code that writes them.
    #
    # TEAMS exists here so the enum, the capability vocabulary and the
    # entitlement wire format are complete and stable. There is deliberately
    # no Teams connector yet: reading channel messages needs Microsoft's
    # Protected APIs approval. An unmapped provider cannot be synced (see
    # `_CAPABILITY` in orchestration/sync.py), so the member is inert.
    OUTLOOK = "outlook"
    OUTLOOK_CALENDAR = "outlook_calendar"
    ONEDRIVE = "onedrive"
    TEAMS = "teams"


class Organization(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    domain: Mapped[str | None] = mapped_column(String(255), nullable=True)

    memberships: Mapped[list["Membership"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    teams: Mapped[list["Team"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )


class User(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    # Set when the user signs in with Google instead of a password.
    google_sub: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    # Set when the user signs in with Microsoft (Entra) instead of a password.
    # `microsoft_sub` is the `oid` claim — stable per user and, unlike the email,
    # not something the account holder can change.
    microsoft_sub: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    # The Entra tenant the user authenticated against (`tid`). Kept because it
    # is the difference between a work/school account and a personal Microsoft
    # account, and only the former can have a domain anybody verified. Not
    # indexed: it is read alongside an already-resolved user, never searched.
    microsoft_tid: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Access tokens issued (iat) before this instant are rejected — the bulk-revoke
    # lever for deactivation and membership removal. NULL = no bulk revocation.
    tokens_valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    memberships: Mapped[list["Membership"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Team(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "teams"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    organization: Mapped["Organization"] = relationship(back_populates="teams")


class Membership(UUIDPrimaryKey, Timestamped, Base):
    """A user's seat in an org, with a role. The tenancy join row."""

    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("user_id", "organization_id", name="uq_user_org"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    # Bind the enum VALUE ('founder'), not the member NAME ('FOUNDER'): the PG `role`
    # enum type is defined with the lowercase values.
    #
    # NOT NULL since migration 0043. The annotation always said `Mapped[Role]` while
    # the column allowed NULL, and that gap let a seat exist with no privilege
    # attached to it — fail-closed for authorization, but a 500 on `/auth/me`.
    # Readers still handle a missing role defensively (a database that predates 0043
    # is a real deployment state, and a role must never be INFERRED); this makes the
    # write side unable to produce one.
    role: Mapped[Role] = mapped_column(
        SAEnum(Role, values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=Role.FOUNDER,
    )

    user: Mapped["User"] = relationship(back_populates="memberships")
    organization: Mapped["Organization"] = relationship(back_populates="memberships")


class Session(UUIDPrimaryKey, Timestamped, Base):
    """Persisted refresh-token session (allows revocation)."""

    __tablename__ = "sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    refresh_token_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)


class Integration(UUIDPrimaryKey, Timestamped, Base):
    """An OAuth-connected signal source for an org (Gmail, Slack, Calendar...).

    Tokens are stored encrypted-at-rest in production; here we keep the shape.
    """

    __tablename__ = "integrations"
    __table_args__ = (
        UniqueConstraint("organization_id", "provider", "external_account", name="uq_org_provider_account"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # Same value-vs-name binding fix as Membership.role (PG enum holds lowercase values).
    provider: Mapped[IntegrationProvider] = mapped_column(
        SAEnum(IntegrationProvider, values_callable=lambda e: [m.value for m in e])
    )
    external_account: Mapped[str | None] = mapped_column(String(320), nullable=True)
    access_token: Mapped[str | None] = mapped_column(String, nullable=True)
    refresh_token: Mapped[str | None] = mapped_column(String, nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scopes: Mapped[str | None] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Provider-specific sync cursors (Gmail historyId, Slack cursor, etc.)
    sync_state: Mapped[dict] = mapped_column(JSONB, default=dict)
