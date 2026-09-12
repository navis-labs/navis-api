"""Identity ORM models: User, Organization, Team, Membership, Session, Integration."""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
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


class IntegrationAuthState(str, enum.Enum):
    """AUTHENTICATION HEALTH — deliberately NOT `is_active`.

    The two answer different questions and must never be merged again:

        is_active   the USER'S INTENT. True from the moment somebody clicks
                    Connect until they click Disconnect. Says nothing about
                    whether the credential still works.

        auth_state  whether the stored grant CAN STILL BE USED. Moved only by
                    the sync path and the OAuth callback, never by a person.

    A source can perfectly well be `is_active=True, auth_state=reconsent_required`
    — the user still wants Gmail connected, Google just rejected the refresh
    token. That is the exact state the old schema could not express, which is why
    a dead grant was retried forever and reported as connected.

    The vocabulary is not new. `RECONSENT_REQUIRED` is named after
    `navis.signals.google.oauth.ReconsentRequired` (raised on `invalid_grant`)
    and `SlackReconsentRequired`; `REVOKED` after Slack's `tokens_revoked`.
    """

    #: The grant works, or has not yet been shown not to.
    OK = "ok"
    #: The refresh token was rejected — revoked, expired, or rotated out. The
    #: user must re-authorise. Terminal until they do: nothing but a new OAuth
    #: grant clears it, which is why the dispatch loop stops retrying it.
    RECONSENT_REQUIRED = "reconsent_required"
    #: The provider told us the token is gone (Slack `tokens_revoked`). Same
    #: remedy as re-consent; kept distinct because the provider SAID so rather
    #: than us inferring it from a refusal, and an operator reading the row
    #: should be able to tell a reported revocation from a deduced one.
    REVOKED = "revoked"


#: The closed vocabulary, as the CHECK constraint spells it. Sorted so Alembic's
#: autogenerate does not see a difference against `0050_integration_health`.
_AUTH_STATE_VALUES = ", ".join(f"'{s.value}'" for s in sorted(IntegrationAuthState, key=lambda s: s.value))


class Integration(UUIDPrimaryKey, Timestamped, Base):
    """An OAuth-connected signal source for an org (Gmail, Slack, Calendar...).

    Tokens are stored encrypted-at-rest in production; here we keep the shape.

    THREE INDEPENDENT FACTS, and conflating any two of them is the bug 0050
    exists to fix:

        is_active                      does the user want this connected?
        auth_state                     does the credential still work?
        last_attempt_at/last_success_at  is data actually arriving?

    Read them together to answer "is this source healthy". Reading `is_active`
    alone answers only whether somebody once clicked Connect.
    """

    __tablename__ = "integrations"
    __table_args__ = (
        UniqueConstraint("organization_id", "provider", "external_account", name="uq_org_provider_account"),
        CheckConstraint(f"auth_state IN ({_AUTH_STATE_VALUES})", name="ck_integrations_auth_state"),
        # The dispatch gate's predicate: active rows that are not known-dead.
        Index("ix_integrations_active_auth_state", "is_active", "auth_state"),
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
    #: THE USER'S INTENT, and nothing else. Set True on connect, False on
    #: disconnect. An auth failure must NOT move this — see `auth_state`.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    #: Authentication health. See `IntegrationAuthState`.
    auth_state: Mapped[str] = mapped_column(
        String(24), nullable=False, default=IntegrationAuthState.OK.value, server_default="ok"
    )
    #: When a sync was last TRIED. Distinguishes "healthy and quiet" from "not
    #: being reached at all" — a source the scheduler has stopped visiting looks
    #: identical to a working one if you only record successes.
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: When data last actually ARRIVED. Staleness is computed from this and from
    #: nothing else. NULL means it has never succeeded, which is a real state for
    #: a freshly-connected source and must not be forged into a timestamp.
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: What went wrong last, in words an operator can act on. Truncated and
    #: scrubbed by the writer — a provider error message is a classic place for a
    #: token or a credentialled URL to appear.
    last_failure_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Provider-specific sync cursors (Gmail historyId, Slack cursor, etc.)
    #
    # HEALTH DOES NOT LIVE HERE, deliberately. This blob is advisory-locked per
    # integration for cursor read-modify-write (`navis/signals/locking.py`), and
    # health is written on the failure path where the cursor does NOT move.
    sync_state: Mapped[dict] = mapped_column(JSONB, default=dict)
