"""Execution V2 persistence — the command aggregate + the transactional outbox.

PostgreSQL is the source of truth. `execution_requests` is the command/aggregate
with full lineage (decision -> reasoning signals -> state watermark) and a unique
idempotency key. `execution_outbox` is the persisted event log: events are written
in the SAME transaction as the state transition (outbox pattern) and a relay worker
publishes them later (persist-first, publish-later) — no dual-write, no distributed
transaction.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


class ActionType(str, enum.Enum):
    SEND_EMAIL = "send_email"
    CREATE_TASK = "create_task"
    SCHEDULE_MEETING = "schedule_meeting"
    SEND_SLACK = "send_slack"


class ExecutionRequest(Timestamped, Base):
    __tablename__ = "execution_requests"
    __table_args__ = (
        # Idempotency: one request per (org, deterministic key).
        UniqueConstraint("organization_id", "idempotency_key", name="uq_execution_idempotency"),
    )

    # Deterministic UUIDv5 of (org, idempotency_key) — replay/retry reproduce it.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # --- Lineage (structured, queryable: why was this done?) ---
    decision_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    reasoning_signal_ids: Mapped[list] = mapped_column(JSONB, default=list)
    reasoning_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state_watermark: Mapped[str | None] = mapped_column(String(64), nullable=True)

    action_type: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)

    # --- Approval (human-in-the-loop) ---
    approved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # --- Execution result ---
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    provider_reference: Mapped[dict] = mapped_column(JSONB, default=dict)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ExecutionOutbox(UUIDPrimaryKey, Base):
    """Persisted execution events. Written in the request's transaction; the relay
    worker publishes pending rows (published_at IS NULL) to the event bus."""

    __tablename__ = "execution_outbox"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    execution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("execution_requests.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ExecutionDeadLetter(UUIDPrimaryKey, Base):
    """Permanently-failed executions (attempts exhausted). Never lost; replayable."""

    __tablename__ = "execution_dead_letters"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    execution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("execution_requests.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str | None] = mapped_column(String(40), nullable=True)
    action_type: Mapped[str] = mapped_column(String(40), nullable=False)
    error: Mapped[str] = mapped_column(Text, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    replayed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ExecutionInbox(Base):
    """Idempotent-consumer dedupe table. A consumer claims an event_id here once;
    duplicate deliveries (at-least-once outbox) are no-ops."""

    __tablename__ = "execution_inbox"

    # (consumer, event_id) — different consumers each process an event once.
    consumer: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
