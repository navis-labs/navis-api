"""Learning V2 — append-only attribution log + deterministic count aggregates.

Source of truth: `learning_events` (immutable, one per outcome, with full lineage).
Aggregates (`learning_aggregates`) and patterns (`learning_patterns`) are
deterministic COUNT projections rebuilt from that log — no EMA, no order
dependence, no in-place float drift. Learning answers "what actually works?" and
stores nothing else (no risk/priority/recommendation/intelligence).

Legacy `ConfidenceWeight` (decision_patterns) + `BehaviorPattern` are FROZEN shims
kept only so the not-yet-rebuilt Intelligence V1 keeps compiling — Learning V2
never writes them.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey

# 2: added outcome_surface / failure_kind / provider_reference. Stamped per row,
# so a v1 row stays readable and says exactly what it was recorded knowing.
LEARNING_SCHEMA_VERSION = 2
_LEARNING_NAMESPACE = uuid.UUID("f2a4c6e8-1b3d-5f70-9a2c-4e6b8d0f1a32")


class LearningResult(str, enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"  # attempt recorded, not yet a win/loss (e.g. action_executed)


class OutcomeSurface(str, enum.Enum):
    """WHERE an outcome happened — the axis `result` alone cannot express.

    Not a new taxonomy: this is Observation's own documented one, named. Its
    `OutcomeType` docstring already groups the vocabulary into THREE SOURCES
    ("MACHINE ... HUMAN ... WORLD") and says the distinction "is load-bearing for
    anything that later reads these as evidence". Learning is that reader, and
    until now it flattened all three into one success/failure axis.

    What that flattening cost, concretely: `action_failed` (Gmail was down) and
    `no_response` (the customer ignored us) both incremented `failures` on the
    same reasoning signal. One is a fact about a channel, the other is a fact
    about a judgement, and a signal's success_rate counted them identically —
    which is exactly the "simplistic success score" Learning is supposed not to
    produce.

    DELIVERY   the machine acted, or the channel broke. Says nothing about
               whether acting was the right call.
    JUDGEMENT  a person's verdict on something Navis proposed. The most direct
               evidence the product can get about its own reasoning.
    WORLD      what actually happened afterwards. The only surface that answers
               "did it work?".
    """

    DELIVERY = "delivery"
    JUDGEMENT = "judgement"
    WORLD = "world"


class AggregateDimension(str, enum.Enum):
    SIGNAL = "signal"      # keyed on reasoning signal_id
    DECISION = "decision"  # keyed on decision_id
    OUTCOME = "outcome"    # keyed on outcome_type


def learning_event_id(outcome_id: str) -> uuid.UUID:
    """One learning event per outcome — deterministic, idempotent."""
    return uuid.uuid5(_LEARNING_NAMESPACE, f"outcome:{outcome_id}")


def pattern_id(org_id: uuid.UUID, signal_id: str) -> uuid.UUID:
    return uuid.uuid5(_LEARNING_NAMESPACE, f"pattern:{org_id}:{signal_id}")


class LearningEvent(Base):
    """Immutable attribution fact: an outcome, classified, with full lineage."""

    __tablename__ = "learning_events"
    __table_args__ = (
        # Serves the replay rebuild's dominant read: keyset-stream ALL of an org's events in
        # (occurred_at, id) fold order (repository.stream_events). The single-column
        # organization_id index can't provide that ordering, so without this composite the
        # keyset paginator re-sorts every page — quadratic at scale. Mirrors State's
        # ix_mem_org_time (migration 0028). Pure-additive.
        Index("ix_learning_events_org_time", "organization_id", "occurred_at", "id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # --- Lineage (the audit's required fix) ---
    outcome_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    execution_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    decision_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    reasoning_signal_ids: Mapped[list] = mapped_column(JSONB, default=list)
    reasoning_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state_watermark: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # The external object this outcome is about (gmail_message_id, calendar
    # event id, ...). Observation has always PUBLISHED it; Learning dropped it,
    # so a learning event could not answer "which message was that?" and the
    # only way to find out was to read `outcomes` — which Learning is forbidden
    # from doing. Carried, never interpreted.
    provider_reference: Mapped[dict] = mapped_column(JSONB, default=dict)

    outcome_type: Mapped[str] = mapped_column(String(40), nullable=False)
    # Which of Observation's three sources this came from (see OutcomeSurface).
    outcome_surface: Mapped[str] = mapped_column(String(16), nullable=False)
    result: Mapped[str] = mapped_column(String(16), nullable=False)
    # `execution.contract.FailureKind`, verbatim, for a DELIVERY failure. Stored
    # so "the org lost its entitlement" stays distinguishable from "Gmail was
    # down" without re-reading Observation or sniffing an error string.
    failure_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)
    schema_version: Mapped[int] = mapped_column(Integer, default=LEARNING_SCHEMA_VERSION)
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class LearningAggregate(Base):
    """Deterministic count projection per (dimension, key). success_rate is derived
    on read = successes / (successes + failures). Rebuildable from learning_events."""

    __tablename__ = "learning_aggregates"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    dimension: Mapped[str] = mapped_column(String(16), primary_key=True)
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    successes: Mapped[int] = mapped_column(Integer, default=0)
    failures: Mapped[int] = mapped_column(Integer, default=0)
    neutrals: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)


class LearningPattern(Base):
    """Derived organizational pattern (Intelligence fuel). Stores facts, never
    recommendations. Rebuildable from aggregates."""

    __tablename__ = "learning_patterns"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    signal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # reliable | unreliable
    samples: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)


class LearningInbox(Base):
    """Idempotent-consumer dedupe for OUTCOME_OBSERVED."""

    __tablename__ = "learning_inbox"

    consumer: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# --- FROZEN legacy shims (Intelligence V1 compile-compat; never written by V2) ---

class ConfidenceWeight(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "decision_patterns"
    __table_args__ = (
        UniqueConstraint("organization_id", "signal_kind", name="uq_org_signal_kind"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    signal_kind: Mapped[str] = mapped_column(String(64), index=True)
    samples: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.5)


class BehaviorPattern(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "behavior_patterns"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    scope: Mapped[str] = mapped_column(String(32), default="founder")
    label: Mapped[str] = mapped_column(String(255))
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
