"""Persisted state snapshots. Each build is stored so State has history too."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


class OrganizationalStateSnapshot(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "organizational_states"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    built_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    summary: Mapped[dict] = mapped_column(JSONB, default=dict)
    # Full serialized projection for replay/debugging.
    projection: Mapped[dict] = mapped_column(JSONB, default=dict)
