"""Intelligence V2 — append-only insights, a deterministic projection over Learning.

Intelligence answers "what has this organization learned about itself?" — facts,
never recommendations/actions/scores. Each IntelligenceInsight is immutable, fully
traceable to the learning events that support it, and reproducible (deterministic
UUIDv5 keyed on the learning version). Rebuildable: drop the table, re-project
Learning, get identical insights.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base

INTELLIGENCE_SCHEMA_VERSION = 1
_INSIGHT_NAMESPACE = uuid.UUID("a7d2f4b6-8c91-5e03-9b2a-4d6f8a0c2e15")


class InsightType(str, enum.Enum):
    SIGNAL = "signal_intelligence"
    DECISION = "decision_intelligence"
    TEAM = "team_intelligence"
    CUSTOMER = "customer_intelligence"
    ORGANIZATIONAL = "organizational_intelligence"
    ADAPTATION = "adaptation_intelligence"


def compute_insight_id(
    org_id: uuid.UUID, insight_type: str, key: str, learning_version: int
) -> uuid.UUID:
    """Deterministic — same org + type + key + learning version => same insight."""
    return uuid.uuid5(_INSIGHT_NAMESPACE, f"{org_id}|{insight_type}|{key}|{learning_version}")


class IntelligenceInsight(Base):
    __tablename__ = "intelligence_insights"

    insight_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    insight_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(128), nullable=False)  # subject (signal_id / outcome_type / "org")

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Lineage: the learning events that support this insight (capped, sorted).
    supporting_learning_event_ids: Mapped[list] = mapped_column(JSONB, default=list)
    learning_version: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, default=INTELLIGENCE_SCHEMA_VERSION)

    # Logical time (= max occurred_at of the learning consumed) — NOT wall clock.
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
