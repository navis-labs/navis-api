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
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


class ActionType(str, enum.Enum):
    SEND_EMAIL = "send_email"
    # A REPLY into an existing Gmail thread. Distinct from SEND_EMAIL: it needs
    # the thread it answers, and it is the shape a PROACTIVE proposal takes —
    # proposed before its body exists, and unable to execute until one does.
    REPLY_TO_EMAIL = "reply_to_email"
    # Retained but UNSUPPORTED — there is no task system. See providers/tasks.py.
    CREATE_TASK = "create_task"
    SCHEDULE_MEETING = "schedule_meeting"
    # Moving and cancelling an EXISTING meeting are distinct action types, not
    # variants of SCHEDULE_MEETING: they carry a different payload (an existing
    # `event_id`), a different provider call (PATCH / DELETE vs POST), and a
    # different governance story (cancelling notifies attendees of a removal).
    # `action_type` is a String(40) column, so adding members needs no migration.
    RESCHEDULE_MEETING = "reschedule_meeting"
    CANCEL_MEETING = "cancel_meeting"
    SEND_SLACK = "send_slack"


class ExecutionOrigin(str, enum.Enum):
    """WHO decided this action should exist. Not how it executes — that is
    identical for both — but which product surface owns reviewing it.

    CHAT      the founder asked for it explicitly and is already looking at the
              proposal in the conversation. Chat owns it end to end.
    PROACTIVE Navis discovered it from signals the founder has not seen. The
              Action Center is the review queue for exactly these.

    Without this distinction `GET /actions` returned every REQUESTED row, so a
    meeting proposed in chat ALSO appeared on the Action Center board — two
    surfaces racing to approve one row, and a "proactive" queue full of things
    the user had just typed.
    """

    CHAT = "chat"
    PROACTIVE = "proactive"


class ExecutionRequest(Timestamped, Base):
    __tablename__ = "execution_requests"
    __table_args__ = (
        # Idempotency: one request per (org, deterministic key).
        UniqueConstraint("organization_id", "idempotency_key", name="uq_execution_idempotency"),
        # ONE inbound message = ONE reply obligation = at most ONE execution.
        #
        # THE guarantee, enforced by the database rather than by an
        # if-exists-return-existing read, which two concurrent proposals would
        # both lose. PARTIAL because action types with no message-level
        # identity legitimately carry NULL and must not collide with each
        # other. Mirrored in migration 0035; kept here so `create_all` (used by
        # the Postgres integration suite) builds the same schema.
        Index(
            "uq_execution_obligation",
            "organization_id",
            "obligation_key",
            unique=True,
            postgresql_where=text("obligation_key IS NOT NULL"),
        ),
    )

    # Deterministic UUIDv5 of (org, idempotency_key) — replay/retry reproduce it.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # --- Lineage (structured, queryable: why was this done?) ---
    decision_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    # TWO ID NAMESPACES, deliberately separate.
    #
    # `reasoning_signal_ids` -> reasoning_signals.signal_id ONLY. These are the
    # deterministic ids Learning aggregates on and Intelligence cites, so a value
    # from any other namespace makes those references permanently unresolvable.
    #
    # `source_signal_event_ids` -> signal_events.id — the raw captured event a
    # chat turn was acting on (the email on screen). Real provenance, different
    # question: "which observation prompted this?" rather than "which reasoning
    # concluded it?". Chat used to write these INTO reasoning_signal_ids, which
    # silently mixed the two.
    # SERVER DEFAULTS MIRROR THE MIGRATIONS, for the same reason the partial
    # index above does: `create_all` builds the schema the Postgres integration
    # suite runs against, and without these it built a DIFFERENT schema from
    # production. Migration 0007 gives every JSONB column here a server default
    # and 0037 does the same for `source_signal_event_ids`; the model declared
    # only a PYTHON-side `default=`, which the ORM applies but raw SQL does not.
    #
    # The visible failure: an INSERT that omits `source_signal_event_ids` — which
    # production accepts, because the migrated column defaults to '[]' — raised
    # NotNullViolationError against the test schema. So the one suite that
    # exercises FOR UPDATE SKIP LOCKED and ON CONFLICT was doing it on a table
    # shaped differently from the real one, which is exactly where that suite's
    # value comes from.
    reasoning_signal_ids: Mapped[list] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb")
    )
    source_signal_event_ids: Mapped[list] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb")
    )
    reasoning_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state_watermark: Mapped[str | None] = mapped_column(String(64), nullable=True)

    action_type: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    # Which surface owns reviewing this. See ExecutionOrigin.
    #
    # Server default is 'chat' so pre-existing rows (written before this column
    # existed, by both surfaces) stay OUT of the proactive queue. Wrongly
    # hiding an old proactive row is a missed nudge; wrongly showing a chat row
    # resurrects the duplicate-surface bug this column exists to kill.
    origin: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=ExecutionOrigin.CHAT.value,
        default=ExecutionOrigin.CHAT.value,
        index=True,
    )
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    # The stable natural key of the underlying obligation, when the action has
    # one — currently reply_to_email: "reply|{kind}|{thread_id}|{message_id}".
    #
    # Stored as a readable column rather than left inside the idempotency hash
    # so the guarantee is inspectable ("show me every execution for this
    # message") and so the database itself can enforce it: see the partial
    # unique index in migration 0035. NULL for action types that have no
    # message-level identity, which is why the index is partial.
    obligation_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    payload: Mapped[dict] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )

    # --- Approval (human-in-the-loop) ---
    approved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # --- Execution result ---
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    provider_reference: Mapped[dict] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
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
