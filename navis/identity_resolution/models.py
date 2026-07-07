"""Identity Resolution ORM — two append-only tables above Memory.

`identity_aliases` is one row per (provider, kind, value) alias per org. `identity_link_events`
is an append-only edge log asserting two aliases refer to the same organizational person.
Both are keyed by deterministic UUIDv5, so INSERT ... ON CONFLICT DO NOTHING makes replay
and at-least-once delivery converge to identical rows.

MemoryRecord is UNCHANGED — this module reads Memory-friendly payloads from signal events
and appends its own tables. Downstream (State, Reasoning, Intelligence, KG projectors) reads
these tables directly; Memory is not consulted for identity resolution and never should be.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base


class IdentityAlias(Base):
    """One observed alias — a (provider, kind, value) triple bound to an org.

    Append-only. Idempotent on the deterministic PK. `first_seen_*` are set on insert;
    `last_seen_*` are updated on subsequent observations (the only mutable columns —
    monotonic, so replay converges even under out-of-order delivery)."""

    __tablename__ = "identity_aliases"
    __table_args__ = (
        UniqueConstraint(
            "organization_id", "provider", "kind", "value",
            name="uq_identity_alias_natural",
        ),
        Index("ix_identity_alias_org_value", "organization_id", "kind", "value"),
        Index("ix_identity_alias_org_provider", "organization_id", "provider"),
    )

    # Deterministic UUIDv5 of (org, provider, kind, value).
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(40), nullable=False)   # gmail | slack | calendar | ...
    kind: Mapped[str] = mapped_column(String(40), nullable=False)       # email | slack_user_id | ...
    value: Mapped[str] = mapped_column(String(320), nullable=False)     # normalized alias value

    # Best-effort human-readable label. Never used for identity — only display.
    display_name: Mapped[str | None] = mapped_column(String(320), nullable=True)

    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    first_seen_signal_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("signal_events.id", ondelete="SET NULL"), nullable=True
    )


class IdentityLinkEvent(Base):
    """An append-only edge asserting alias_a and alias_b refer to the same person.

    Provenanced: every link carries the signal it was derived from. Confidence
    reflects the evidence strength — a Slack profile that carries an email is
    STRUCTURAL (self-declared); a shared domain heuristic would be HEURISTIC.

    Canonical resolution is NEVER a destructive merge — it is a BFS over this
    edge log at query time. A wrong link is therefore a recoverable append
    (add an inverse annotation or a governance dispose), not corruption."""

    __tablename__ = "identity_link_events"
    __table_args__ = (
        # Alias-directed lookup for BFS: "give me every link touching this alias".
        Index("ix_identity_link_alias_a", "organization_id", "alias_a_id"),
        Index("ix_identity_link_alias_b", "organization_id", "alias_b_id"),
        # Provenance / replay.
        Index("ix_identity_link_signal", "source_signal_event_id"),
    )

    # Deterministic UUIDv5 of (org, source_signal_event_id, sorted(alias_a, alias_b)).
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    alias_a_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("identity_aliases.id", ondelete="CASCADE"), nullable=False
    )
    alias_b_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("identity_aliases.id", ondelete="CASCADE"), nullable=False
    )
    source_signal_event_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("signal_events.id", ondelete="SET NULL"), nullable=True
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    event_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
