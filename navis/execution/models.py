"""Persisted actions. The audit trail of everything Navis proposed and did."""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey


class ActionType(str, enum.Enum):
    SEND_EMAIL = "send_email"
    CREATE_TASK = "create_task"
    SCHEDULE_MEETING = "schedule_meeting"
    CREATE_WORKFLOW = "create_workflow"


class ActionStatus(str, enum.Enum):
    REQUESTED = "requested"   # proposed by a decision, awaiting approval
    APPROVED = "approved"
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"


class Action(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "actions"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[ActionType] = mapped_column(SAEnum(ActionType))
    status: Mapped[ActionStatus] = mapped_column(
        SAEnum(ActionStatus), default=ActionStatus.REQUESTED, index=True
    )
    # Why this action exists — the brief item / signal that proposed it.
    origin_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    result: Mapped[dict] = mapped_column(JSONB, default=dict)
    approved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
