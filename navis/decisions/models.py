"""Persisted daily briefs (one per org per generation)."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


class DailyBrief(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "daily_briefs"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    state_summary: Mapped[dict] = mapped_column(JSONB, default=dict)
    items: Mapped[list] = mapped_column(JSONB, default=list)
