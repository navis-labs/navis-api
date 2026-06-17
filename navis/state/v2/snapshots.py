"""State snapshot cache table.

A snapshot is a CACHE of the projection, not a source of truth — it can be dropped
and rebuilt from Memory at any time. Keyed by (org, projection_version); upserted.
No UUID4 PK (determinism law): identity is the natural key.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base


class StateSnapshot(Base):
    __tablename__ = "state_snapshots"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    projection_version: Mapped[int] = mapped_column(Integer, primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)

    # Cursor into Memory: everything up to (and including) this point is folded in.
    watermark_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    watermark_memory_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    full_projection: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Cache-write metadata (NOT part of the projection; deterministic content lives
    # entirely in full_projection).
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
