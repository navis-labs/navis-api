"""Observation V2 persistence — the append-only outcome system of record.

An Outcome is one immutable fact: "this happened after execution". Records are
NEVER updated/mutated — a new fact is a new row. Every outcome carries full
lineage back to its Execution -> Decision -> ReasoningSignals -> State watermark,
so Learning and auditors trace cause to effect. The PK is a deterministic UUIDv5
of (source_event_id, outcome_type), so duplicate deliveries collapse to one row.

Observation stores OUTCOME TRUTH only — no success score, no confidence, no
prediction (those are Learning's). The `observation_inbox` gives idempotent
at-least-once consumption.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base

OUTCOME_SCHEMA_VERSION = 1
_OUTCOME_NAMESPACE = uuid.UUID("d3f8a1b6-2c47-5e90-8a3b-1f2c4d6e8a05")


class OutcomeType(str, enum.Enum):
    # From Execution events
    ACTION_EXECUTED = "action_executed"
    ACTION_FAILED = "action_failed"
    # From provider webhooks (real-world results)
    REPLY_RECEIVED = "reply_received"
    MEETING_ACCEPTED = "meeting_accepted"
    MEETING_COMPLETED = "meeting_completed"
    # From pollers (future)
    NO_RESPONSE = "no_response"


def compute_outcome_id(source_event_id: str, outcome_type: str) -> uuid.UUID:
    """Deterministic id — same source event + type => same outcome (idempotent).

    Identity constraint: source_event_id must be GLOBALLY unique (org is deliberately
    not in the key — changing the formula would double-record any event redelivered
    across the deploy). Both producers satisfy it: execution outbox row UUIDs, and
    provider-prefixed webhook ids ("gmail:<msg>:<reply>"). A new source must keep
    that guarantee, or a cross-org id collision silently drops the second outcome."""
    return uuid.uuid5(_OUTCOME_NAMESPACE, f"{source_event_id}|{outcome_type}")


class Outcome(Base):
    __tablename__ = "outcomes"

    outcome_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # --- Lineage (structured; no free-form subject_ref) ---
    execution_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    decision_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    reasoning_signal_ids: Mapped[list] = mapped_column(JSONB, default=list)
    reasoning_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state_watermark: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider_reference: Mapped[dict] = mapped_column(JSONB, default=dict)

    outcome_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    schema_version: Mapped[int] = mapped_column(Integer, default=OUTCOME_SCHEMA_VERSION)

    # The event/webhook that produced this outcome (dedupe + provenance).
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)  # reality time
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)  # ingest time
    # Append-only: created_at only (no updated_at / no onupdate).
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ObservationInbox(Base):
    """Idempotent-consumer dedupe — a (consumer, event_id) is processed once."""

    __tablename__ = "observation_inbox"

    consumer: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
