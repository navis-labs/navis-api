"""Persisted learning: raw events + the running confidence weights they produce."""
from __future__ import annotations

import uuid

from sqlalchemy import Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


class LearningEvent(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "learning_events"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    signal_kind: Mapped[str] = mapped_column(String(64), index=True)
    success: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)


class ConfidenceWeight(UUIDPrimaryKey, Timestamped, Base):
    """Running success-rate per signal kind, per org. The learned dial reasoning
    and decisions read to scale their confidence."""

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
    """Founder/team behavior patterns (e.g. "ignores hiring alerts on Mondays")."""

    __tablename__ = "behavior_patterns"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    scope: Mapped[str] = mapped_column(String(32), default="founder")  # founder|team|company
    label: Mapped[str] = mapped_column(String(255))
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
