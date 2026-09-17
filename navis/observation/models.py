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
    """What kind of fact this outcome records.

    THREE SOURCES, and the distinction is load-bearing for anything that later
    reads these as evidence:

      * MACHINE outcomes — what Execution did. `ACTION_EXECUTED` means the
        provider call succeeded; it does NOT mean the business outcome occurred
        (see the module docstring on Outcome).
      * HUMAN outcomes — what a person decided about a proposal. These are the
        highest-value feedback Navis can receive, because they are a direct
        judgement on something it proposed.
      * WORLD outcomes — what actually happened afterwards, from provider
        webhooks. Only these answer "did it work?".

    `outcome_type` is a String(40) column, so members can be added without a
    migration. Learning maps type -> result and defaults unknown types to
    NEUTRAL, so a new member is inert there until Learning is taught about it.
    """

    # ── Machine: what Execution did ──────────────────────────────────────
    ACTION_EXECUTED = "action_executed"
    ACTION_FAILED = "action_failed"

    # ── Human: what a person decided about a proposal ────────────────────
    #
    # Execution has always EMITTED these (ACTION_REJECTED / ACTION_CANCELLED are
    # in `_EVENT_FOR_STATUS`, written to the outbox in the same transaction as
    # the status change and relayed like every other event). Observation simply
    # had no subscriber for them, so a founder declining a proposal — the
    # clearest negative signal the product can get — was published and dropped.
    #
    # REJECTED and CANCELLED are separate because they mean different things: a
    # rejection is a judgement on the proposal ("no, not this"), a cancellation
    # is usually a withdrawal of a request that has been overtaken by events.
    # Collapsing them would teach a future Learning pass that the two are the
    # same evidence, which they are not.
    ACTION_REJECTED = "action_rejected"
    ACTION_CANCELLED = "action_cancelled"

    # A founder removed a DECISION from their brief. The judgement above are
    # verdicts on a proposed ACTION; this is a verdict on the CONCLUSION itself
    # — "Navis, this did not need me" — and it is the ONLY human choice most
    # decisions can receive: `risk`, `opportunity` and `delegation` map to no
    # action type (`execution.proposer.decision_action`), so they never become
    # an ExecutionRequest and can never be rejected or cancelled.
    #
    # ITS OWN MEMBER, and not ACTION_REJECTED reused. The three existing
    # judgement types all carry an `execution_id`; this one carries a decision
    # and its reasoning signals and nothing else, and folding it into a member
    # whose whole meaning is "about an action" would make `execution_id IS NULL`
    # the only way to tell two different verdicts apart. The same argument the
    # REJECTED/CANCELLED split already documents.
    #
    # WHY the brief-dismissal reason is NOT split into three members: the reason
    # ("explicit" / "stale" / "duplicate") is stored verbatim on the outcome
    # payload, so a later pass can separate them without re-deriving anything,
    # and minting three types before anything reads them would fix a taxonomy to
    # a distinction nothing yet uses.
    DECISION_DISMISSED = "decision_dismissed"

    # ── World: what actually happened (provider webhooks) ────────────────
    REPLY_RECEIVED = "reply_received"
    MEETING_ACCEPTED = "meeting_accepted"
    MEETING_COMPLETED = "meeting_completed"
    # From pollers (future)
    NO_RESPONSE = "no_response"


# `FailureKind` — the vocabulary for WHY an execution failed — lives in
# `navis.execution.contract`, next to `OutcomeReference`, because Execution is
# what knows the cause and that module is already "the hooks Observation will
# consume". Observation stores the value verbatim in
# `ACTION_FAILED.payload["failure_kind"]` and never interprets it.


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
