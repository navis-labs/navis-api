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
    # ── Learning-derived: what the organization has learned about ITSELF from
    # the outcomes of its own actions. Versioned by `learning_version`.
    SIGNAL = "signal_intelligence"
    DECISION = "decision_intelligence"
    TEAM = "team_intelligence"
    CUSTOMER = "customer_intelligence"
    ORGANIZATIONAL = "organizational_intelligence"
    ADAPTATION = "adaptation_intelligence"

    # ── Reality-derived: what is TRUE of the organization right now, folded
    # from current State and the conclusions Reasoning already reached about it.
    # Versioned by `state_watermark` — reality moves when State moves, not when
    # an outcome is recorded.
    #
    # Deliberately NOT named `*_intelligence` where the honest word is narrower:
    # `status_intelligence` would imply a health verdict, and Navis has no basis
    # for scoring organizational health. See `status_intelligence()`.
    RISK = "risk_intelligence"
    OPPORTUNITY = "opportunity_intelligence"
    RELATIONSHIP = "relationship_intelligence"
    COMMITMENT = "commitment_intelligence"
    # What the organization currently OWES a named counterparty — an unanswered
    # person, a judgment somebody asked for. Its OWN family rather than a member
    # of RISK, deliberately: `_RISK_SIGNAL_TYPES` is byte-identical to the set
    # Decision maps to `DecisionCategory.RISK`, so the two layers mean the same
    # thing by "risk" and `organizational_status`'s open-risk count matches the
    # brief's. A debt with a name on it is not a decaying relationship.
    OBLIGATION = "obligation_intelligence"
    STATUS = "organizational_status"
    TREND = "trend_intelligence"
    # ONE COUNTERPARTY, and every conclusion Reasoning reached about them,
    # connected through the edges State persisted. The only insight family that
    # is not a GROUP BY over signal type: every other reality type answers "how
    # many of this kind of thing?", and this one answers "what is going on with
    # this person?". See `insights.situation_intelligence`.
    SITUATION = "situation_intelligence"


def compute_insight_id(
    org_id: uuid.UUID, insight_type: str, key: str, version_key: int | str
) -> uuid.UUID:
    """Deterministic — same org + type + key + version => same insight.

    `version_key` is the dimension along which an insight is ALLOWED to change.
    It used to be `learning_version` and nothing else, which is correct for an
    insight derived from the Learning log and wrong for one derived from current
    organizational reality: reality moves when State moves, not when an outcome
    is recorded. A reality insight keyed on `learning_version` would collapse
    onto one id per org and — the table being append-only with ON CONFLICT DO
    NOTHING — freeze at whatever was true the first time it was projected.

    So the parameter is now the version ITSELF rather than one particular
    version's name:
        Learning-derived  -> str(learning_version)   (an int renders identically)
        reality-derived   -> state_watermark         (the memory id State folded to)

    BYTE-IDENTICAL FOR EVERY EXISTING ROW. The key string is unchanged and an
    int still formats to the same characters, so every id minted before this
    generalisation is reproduced exactly. `_INSIGHT_NAMESPACE` is untouched.
    """
    return uuid.uuid5(_INSIGHT_NAMESPACE, f"{org_id}|{insight_type}|{key}|{version_key}")


# The reality version key is `f"{state_watermark}|r{reasoning_version}"`; rows
# minted before reasoning became part of the identity used the bare watermark.
# See `service._version_key`.
PRE_VERSIONED_REALITY = -1
"""Rank for a reality row minted under the BARE watermark, before the reasoning
version joined the identity. Strictly older than any reasoning version — including
version 0, which is a real (if defaulted) version rather than the absence of one."""

# uuid5 is not invertible, so recovery is a bounded probe. REASONING_VERSION is a
# hand-bumped constant that has moved four times in the product's life; 64 is far
# past any plausible value and costs ~65 hashes (tens of microseconds) per row.
# Intelligence may not import Reasoning (the Core Law, pinned by an AST test), so
# the ceiling cannot be read from the constant itself.
_MAX_PROBED_REASONING_VERSION = 64


def recover_reasoning_version(row) -> int | None:
    """Which reasoning version produced this stored reality insight.

    `compute_insight_id` hashes the version key into `insight_id` and no column
    stores the key, so the row cannot SAY which projection it came from. It can
    still PROVE it: recompute the id under each candidate key and see which one
    the row's own id equals. A match is a proof rather than an inference — uuid5
    is a function of the exact key string, and every input to that string
    (`organization_id`, `insight_type`, `key`, `state_watermark`) is a column.

    Load-bearing at the read boundary because reasoning v3 and v4 folded the SAME
    State snapshot live: their rows agree on organization, type, key,
    learning_version, state_watermark and created_at, and differ ONLY in the
    version key hashed into the id. Nothing else can tell the corrected
    projection from the one it corrected.

    Returns `PRE_VERSIONED_REALITY` for a bare-watermark row, the version for a
    versioned one, and None when the id is reproducible under neither — a
    Learning-derived row, a row with no watermark, or one whose key predates this
    id scheme entirely. None means "unknown", never "old".
    """
    watermark = row.state_watermark
    if not watermark:
        return None
    args = (row.organization_id, row.insight_type, row.key)
    if compute_insight_id(*args, watermark) == row.insight_id:
        return PRE_VERSIONED_REALITY
    for version in range(_MAX_PROBED_REASONING_VERSION + 1):
        if compute_insight_id(*args, f"{watermark}|r{version}") == row.insight_id:
            return version
    return None


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

    # The State snapshot this insight describes — a real column, not a key inside
    # `evidence`, because it is now the version dimension for reality-derived
    # insights (see compute_insight_id) and has to be queryable and indexable.
    #
    # NULLABLE and never backfilled. Rows written before this column existed do
    # not know which snapshot they described; writing a guess would be
    # fabricated provenance. `evidence["state_watermark"]` still carries the
    # Learning-inherited value for those rows and is left exactly as it was.
    state_watermark: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # WHICH CODE produced this row, at what version. ReasoningSignalRecord has
    # carried this since it existed; Intelligence never did, so "would this
    # generator still emit this today?" was unanswerable. It becomes load-bearing
    # the moment a second generator family exists, because without it a
    # projection change silently rewrites the meaning of historical rows.
    #
    # Not part of the insight id: an engine version bump is a reason to re-derive
    # and compare, not a reason to mint a parallel row for the same fact.
    engine_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    engine_version: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Logical time (= max occurred_at of the learning consumed) — NOT wall clock.
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
