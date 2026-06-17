"""Persisted signal_events — the raw, immutable, append-only log of reality.

The PK is the deterministic ``event_id``. The dedup uniqueness is the wider
``(organization_id, source, external_id, external_version)`` so that a NEW
version of the same external entity (a rescheduled meeting, an edited doc) is a
distinct row rather than a swallowed duplicate.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


class SignalEventRecord(Timestamped, Base):
    __tablename__ = "signal_events"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "source",
            "external_id",
            "external_version",
            name="uq_signal_dedupe",
        ),
        # Replay/consumer scans are by tenant + source + time.
        Index("ix_signal_org_source_time", "organization_id", "source", "occurred_at"),
        Index("ix_signal_integration", "integration_id"),
        Index("ix_signal_external", "external_id"),
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

    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
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
    source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)  # fetch | ingest
    event_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    dedupe_key: Mapped[str | None] = mapped_column(String(800), nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=False)
    # The full failed SignalEvent (or fetch context) so replay can reconstruct it.
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    replayed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
