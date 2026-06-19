"""DailyBrief — the disposable, replayable container of a generation's Decisions.

Keyed deterministically by (org, reasoning_version, state_watermark) so replaying
the same Reasoning rebuilds the identical brief (upserted). A cache, not a source
of truth — drop it and rebuild from Reasoning.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
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

    # Cache-write metadata (NOT identity; the deterministic content is `decisions`).
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
