"""LEGACY memory entities — FROZEN. Do not use for new work.

These mutable CRUD tables predate Memory V2 (the event-sourced `memory_records`
log). They are retained ONLY so that `state/` and `intelligence/` keep compiling
until State V2 is built against `memory_records`. Memory V2 NO LONGER WRITES
these tables — they are read-only relics. New facts go to `memory.models.MemoryRecord`.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


class OrgScoped(UUIDPrimaryKey, Timestamped):
    """Mixin: every memory entity belongs to exactly one organization."""

    @staticmethod
    def org_fk() -> Mapped[uuid.UUID]:
        return mapped_column(
            UUID(as_uuid=True),
            ForeignKey("organizations.id", ondelete="CASCADE"),
            index=True,
            nullable=False,
        )


class CommitmentStatus(str, enum.Enum):
    OPEN = "open"
    DELIVERED = "delivered"
    OVERDUE = "overdue"
    DROPPED = "dropped"


class ProjectStatus(str, enum.Enum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    STALLED = "stalled"
    DONE = "done"


class RelationshipKind(str, enum.Enum):
    INTERNAL = "internal"
    CUSTOMER = "customer"
    INVESTOR = "investor"
    CANDIDATE = "candidate"
    VENDOR = "vendor"
    OTHER = "other"


class Person(Base, OrgScoped):
    __tablename__ = "people"

    organization_id: Mapped[uuid.UUID] = OrgScoped.org_fk()
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(320), index=True, nullable=True)
    company: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Soft signals memory accumulates over time (no reasoning here — just storage).
    last_interaction_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    interaction_count: Mapped[int] = mapped_column(default=0)
    attributes: Mapped[dict] = mapped_column(JSONB, default=dict)


class Company(Base, OrgScoped):
    __tablename__ = "companies"

    organization_id: Mapped[uuid.UUID] = OrgScoped.org_fk()
    name: Mapped[str] = mapped_column(String(255), index=True)
    domain: Mapped[str | None] = mapped_column(String(255), nullable=True)
    kind: Mapped[RelationshipKind] = mapped_column(
        SAEnum(RelationshipKind), default=RelationshipKind.OTHER
    )
    attributes: Mapped[dict] = mapped_column(JSONB, default=dict)


class Project(Base, OrgScoped):
    __tablename__ = "projects"

    organization_id: Mapped[uuid.UUID] = OrgScoped.org_fk()
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[ProjectStatus] = mapped_column(SAEnum(ProjectStatus), default=ProjectStatus.ACTIVE)
    owner_person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("people.id", ondelete="SET NULL"), nullable=True
    )
    last_activity_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attributes: Mapped[dict] = mapped_column(JSONB, default=dict)


class Meeting(Base, OrgScoped):
    __tablename__ = "meetings"

    organization_id: Mapped[uuid.UUID] = OrgScoped.org_fk()
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    starts_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    external_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    attendees: Mapped[list] = mapped_column(JSONB, default=list)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)


class Commitment(Base, OrgScoped):
    """Something promised. The heartbeat of "what's slipping"."""

    __tablename__ = "commitments"

    organization_id: Mapped[uuid.UUID] = OrgScoped.org_fk()
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[CommitmentStatus] = mapped_column(
        SAEnum(CommitmentStatus), default=CommitmentStatus.OPEN, index=True
    )
    owner_person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("people.id", ondelete="SET NULL"), nullable=True
    )
    counterparty_person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("people.id", ondelete="SET NULL"), nullable=True
    )
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    promised_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Provenance: which signal this was extracted from.
    source_signal_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("signal_events.id", ondelete="SET NULL"), nullable=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    attributes: Mapped[dict] = mapped_column(JSONB, default=dict)


class Decision(Base, OrgScoped):
    __tablename__ = "decisions"

    organization_id: Mapped[uuid.UUID] = OrgScoped.org_fk()
    title: Mapped[str] = mapped_column(String(512))
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    made_by_person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("people.id", ondelete="SET NULL"), nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    attributes: Mapped[dict] = mapped_column(JSONB, default=dict)


class Relationship(Base, OrgScoped):
    """An edge: the org's standing relationship with an external person/company."""

    __tablename__ = "relationships"

    organization_id: Mapped[uuid.UUID] = OrgScoped.org_fk()
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("people.id", ondelete="CASCADE"), nullable=True
    )
    kind: Mapped[RelationshipKind] = mapped_column(SAEnum(RelationshipKind), default=RelationshipKind.OTHER)
    last_contact_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    health: Mapped[float] = mapped_column(Float, default=0.5)  # 0=cold, 1=warm
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    attributes: Mapped[dict] = mapped_column(JSONB, default=dict)
