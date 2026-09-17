"""DailyBrief — the disposable, replayable container of a generation's Decisions.

Keyed deterministically by (org, reasoning_version, state_watermark) so replaying
the same Reasoning rebuilds the identical brief (upserted). A cache, not a source
of truth — drop it and rebuild from Reasoning.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base


class DailyBrief(Base):
    __tablename__ = "daily_briefs"

    # Deterministic UUIDv5 (no uuid4) — replay reproduces the same id.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    reasoning_version: Mapped[int] = mapped_column(Integer, nullable=False)
    state_watermark: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    decisions: Mapped[list] = mapped_column(JSONB, default=list)
    # The Adaptive PolicySnapshot.policy_version that governed this generation,
    # or NULL when the org was running on code defaults. Not part of the id —
    # a policy change replaces the brief for a generation rather than forking it.
    policy_version: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Cache-write metadata (NOT identity; the deterministic content is `decisions`).
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DecisionDismissal(Base):
    """One decision dismissed by a user — an explicit negative signal.

    Consumed by Learning as first-class outcome feedback. Idempotent by (org, decision
    identifier, user): a re-click doesn't create a second row. The identifier is stored
    as an opaque string so it can be a Decision UUIDv5, a reasoning signal id, or any
    other stable event id the frontend passes — no cross-domain FK coupling.
    """

    __tablename__ = "decision_dismissals"
    __table_args__ = (
        UniqueConstraint(
            "organization_id", "decision_ref", "user_id",
            name="uq_decision_dismissal_natural",
        ),
        Index("ix_decision_dismissal_org_time", "organization_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    # Opaque identifier the frontend passed — decision id, event id, whatever.
    # No FK; the identifier space is deliberately open so removed decisions still
    # carry their dismissal history for Learning.
    decision_ref: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(String(32), nullable=False)  # explicit | stale | duplicate
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
