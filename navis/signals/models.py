"""Persisted signal_events — the raw, immutable, append-only log of reality.

The PK is the deterministic ``event_id``. The dedup uniqueness is the wider
``(organization_id, integration_id, source, external_id, external_version)`` so that a
NEW version of the same external entity (a rescheduled meeting, an edited doc) is a
distinct row rather than a swallowed duplicate — AND so two accounts in one org that
observe the same global external_id (a shared Drive file, a multi-attendee meeting) each
keep their own observation instead of the second being silently dropped.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func, text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


class SignalEventRecord(Timestamped, Base):
    __tablename__ = "signal_events"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "integration_id",
            "source",
            "external_id",
            "external_version",
            name="uq_signal_dedupe",
        ),
        # Replay/consumer scans are by tenant + source + time.
        Index("ix_signal_org_source_time", "organization_id", "source", "occurred_at"),
        Index("ix_signal_integration", "integration_id"),
        # Channel-scoped retrieval (Slack et al.): Memory/State/KG/Reasoning query "this
        # channel over time" constantly. channel_id is denormalized from metadata into a real
        # column (NULL for non-channel sources) so this never JSONB-scans 100B rows.
        Index("ix_signal_org_channel_time", "organization_id", "channel_id", "occurred_at"),
        # NOTE: a bare external_id index was intentionally removed — no query looks up by
        # external_id alone (dedup uses uq_signal_dedupe; reads are by PK or org+time). At
        # 100B rows a 512-char-key B-tree on a never-queried column is pure write
        # amplification + storage. A tenant-scoped (org, external_id) index can be added
        # if/when such a lookup is actually introduced.
    )

    # event_id (deterministic UUIDv5) is the primary key.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    integration_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("integrations.id", ondelete="SET NULL"), nullable=True
    )

    source: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    external_id: Mapped[str] = mapped_column(String(512), nullable=False)
    external_version: Mapped[str] = mapped_column(String(128), nullable=False)
    # Denormalized from metadata["channel_id"] for channel-scoped indexing. NULL for sources
    # without a channel (and for identity snapshots that describe a user, not a channel).
    channel_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # When the connector OBSERVED the fact (capture time), distinct from occurred_at (when
    # it happened) and created_at (when this row was inserted). Drives capture-lag analysis.
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Event-schema version for long-horizon replay safety (matches SIGNAL_SCHEMA_VERSION).
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    extra: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)


class SignalDeadLetter(UUIDPrimaryKey, Timestamped, Base):
    """Poison signals that failed to ingest after retries.

    PostgreSQL is the source of truth for dead letters (the Redis stream is only a
    fast secondary alert carrying ids, never payload). Replay reads from here.
    """

    __tablename__ = "signal_dead_letters"
    __table_args__ = (
        Index("ix_dlq_org_unresolved", "organization_id", "resolved_at"),
        Index("ix_dlq_integration", "integration_id"),
    )

    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True
    )
    integration_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("integrations.id", ondelete="SET NULL"), nullable=True
    )
    # The DLQ payload is a SECOND copy of the failed SignalEvent (may contain PII — email
    # bodies, names). user_id makes that copy targetable for GDPR erasure / data-subject
    # requests without scanning JSONB. Nullable: fetch-stage failures may have no user.
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)  # fetch | ingest
    event_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    dedupe_key: Mapped[str | None] = mapped_column(String(800), nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=False)  # the ORIGINAL failure
    # The full failed SignalEvent (or fetch context) so replay can reconstruct it.
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    # Replay accounting (poison-loop guard + observability): how many times re-ingestion
    # was attempted, and the most recent replay error. After max attempts the row is
    # resolved (permanently dead-lettered) so it leaves the retry set.
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    replayed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SignalOutbox(UUIDPrimaryKey, Base):
    """Transactional outbox for signal pointer events (Issue B fix).

    A pointer row is appended in the SAME transaction as the signal_events insert, so a
    crash after commit can never lose the downstream notification. A relay worker publishes
    pending rows (``published_at IS NULL``) to the bus at-least-once, then marks them
    published; consumers dedupe by ``event_id`` in the payload (idempotent). Replaces the
    fragile commit-then-publish dual write.
    """

    __tablename__ = "signal_outbox"
    __table_args__ = (
        # The relay drains `WHERE published_at IS NULL ORDER BY created_at` — a partial
        # index keeps it tiny (only the unpublished backlog) even as the table grows.
        Index(
            "ix_signal_outbox_unpublished", "created_at",
            postgresql_where=text("published_at IS NULL"),
        ),
    )

    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    # The signal event_id this pointer announces (deterministic UUIDv5) — consumer dedupe key.
    aggregate_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)  # platform EventType value
    source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)  # pointer + lineage, never the body
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SignalImportance(Base):
    """Deterministic importance score for one signal, at one policy version.

    A DERIVED artifact, deliberately kept out of `signal_events`: that log is
    immutable and rebuildable, and adding a mutable scored column to it would
    break both properties. This mirrors `memory_embeddings` — same fact, many
    versions, recomputable from the source at any time.

    Because the id is a pure function of (signal, version), re-scoring is
    idempotent (`ON CONFLICT DO NOTHING`) and two policy versions coexist, so a
    new policy can be computed and compared BEFORE anything consumes it.
    """

    __tablename__ = "signal_importance"
    __table_args__ = (
        UniqueConstraint("signal_event_id", "version", name="uq_importance_signal_version"),
        # Distribution / threshold queries scan by org + score.
        Index("ix_signal_importance_org_score", "organization_id", "score"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True, index=True,
    )
    signal_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("signal_events.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    classification: Mapped[str] = mapped_column(String(40), nullable=False)
    # Why it scored that way — makes a score auditable without re-deriving it.
    features: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


_IMPORTANCE_NAMESPACE = uuid.UUID("c4e8a1b7-5d92-5f38-a067-2b9d4e6f8a13")


def compute_importance_id(signal_event_id: uuid.UUID, version: int) -> uuid.UUID:
    """Deterministic PK — same signal + version always collapses to one row."""
    return uuid.uuid5(_IMPORTANCE_NAMESPACE, f"{signal_event_id}|{version}")
