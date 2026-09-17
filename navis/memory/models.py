"""Memory V2 ORM — the append-only event-sourced history layer.

`memory_records` is the single, immutable system of record. Every row is one fact
extracted from exactly one SignalEvent by exactly one extractor, with full
provenance and a deterministic primary key (so replay is idempotent).

Legacy mutable entities (Person, Commitment, ...) are re-exported from
`legacy_models` for backward-compat with `state/` and `intelligence/` only —
Memory V2 never writes them.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, UUIDPrimaryKey

# --- Legacy re-exports (FROZEN; state/intelligence still import these) ---
from navis.memory.legacy_models import (  # noqa: F401
    Commitment,
    CommitmentStatus,
    Company,
    Decision,
    Meeting,
    OrgScoped,
    Person,
    Project,
    ProjectStatus,
    Relationship,
    RelationshipKind,
)


class MemoryRecord(Base):
    """One immutable historical fact. Append-only — never updated, never deleted
    by the application. The PK is a deterministic UUIDv5, so re-ingesting or
    replaying the same signal produces the same row (idempotent via ON CONFLICT)."""

    __tablename__ = "memory_records"
    __table_args__ = (
        # History of a single real-world entity (e.g. all facts about one person).
        Index("ix_mem_entity", "organization_id", "entity_type", "entity_id"),
        # Per-(entity_type) time scans.
        Index("ix_mem_org_type_time", "organization_id", "entity_type", "event_timestamp"),
        # The State V2 projection streams ALL of an org's records in (event_timestamp,
        # memory_id) fold order, regardless of entity_type — so it needs an org+time index
        # WITHOUT entity_type in the lead (the type-led index above can't serve that order).
        Index("ix_mem_org_time", "organization_id", "event_timestamp", "memory_id"),
        # ...and it RESUMES in (created_at, memory_id) OBSERVATION order, which is a
        # different question needing a different index. Event time is not
        # observation-monotone — a calendar meeting's event_timestamp is its START —
        # so the fold order cannot double as a cursor. Without this the keyset read
        # in `MemoryReader.observed_since` plans as a seq scan plus a sort on the
        # hot path of every brief. See migration 0044.
        Index("ix_mem_org_observed", "organization_id", "created_at", "memory_id"),
        # Provenance / replay-by-signal.
        Index("ix_mem_signal", "source_signal_event_id"),
    )

    # Deterministic UUIDv5 of (org, signal_event, extractor, entity_type, dedup_key).
    memory_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    entity_type: Mapped[str] = mapped_column(String(40), nullable=False)
    # Stable identity of the real-world entity this fact is about (UUIDv5).
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # --- Provenance (mandatory) ---
    source_signal_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("signal_events.id", ondelete="RESTRICT"), nullable=False
    )
    source_integration_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("integrations.id", ondelete="SET NULL"), nullable=True
    )
    extractor_name: Mapped[str] = mapped_column(String(80), nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(20), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)

    # --- Fact content ---
    event_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    dedup_key: Mapped[str] = mapped_column(String(800), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Ingest time (operational metadata — NOT part of identity, varies across replays).
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MemoryOutbox(UUIDPrimaryKey, Base):
    """Transactional outbox for memory-derived domain events.

    Memory extractors produce facts (PERSON, COMMITMENT, DECISION, ...) that
    downstream domains need to react to — the onboarding funnel advances to
    MEMORY_READY on the first PersonDetected/CommitmentDetected, State V2 folds
    on RelationshipUpdated, and Learning tracks DecisionDetected. Writing straight
    to the bus after commit is a dual write that can lose events on a crash. Instead
    we stage a pointer row here in the SAME transaction as the memory_records append,
    and a relay worker publishes pending rows (`published_at IS NULL`) to the bus
    at-least-once. Consumers dedupe on the payload identity (memory_id) — idempotent.

    Mirrors `signal_outbox`. Same partial-index + retention-purge pattern.
    """

    __tablename__ = "memory_outbox"
    __table_args__ = (
        Index(
            "ix_memory_outbox_unpublished", "created_at",
            postgresql_where=text("published_at IS NULL"),
        ),
    )

    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    # The memory_id this pointer announces — consumer dedupe key.
    aggregate_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)  # platform EventType value
    entity_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)  # pointer + lineage, never the raw body
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
