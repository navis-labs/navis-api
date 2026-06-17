"""Persisted outcome observations."""
from __future__ import annotations

import enum
import uuid

from sqlalchemy import Enum as SAEnum, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


class OutcomeKind(str, enum.Enum):
    REPLY_RECEIVED = "reply_received"
    MEETING_BOOKED = "meeting_booked"
    COMMITMENT_COMPLETED = "commitment_completed"
    NO_RESPONSE = "no_response"
    IGNORED = "ignored"


class Outcome(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "outcomes"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[OutcomeKind] = mapped_column(SAEnum(OutcomeKind))
    # What this outcome relates back to (action:<id>, commitment:<id>, brief_item ...).
    subject_ref: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    success: Mapped[float] = mapped_column(Float, default=0.0)  # 0..1
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)
