"""Decision V2 contract.

A Decision is a prioritized, action-worthy interpretation derived from
ReasoningSignals. It is deterministic and fully traceable: every Decision points
back at the exact `reasoning_signal_ids` that produced it, the `reasoning_version`,
and the `state_watermark` of the State the reasoning ran over.

Decision does NOT compute risk/confidence/state/intelligence and does NOT propose
actions (no emails/tasks). It converts signals into ranked priorities. Categories
are the four first-class decision types.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from navis.reasoning.v2.contract import SignalType

_DECISION_NAMESPACE = uuid.UUID("c4f1e2d3-5a6b-5c8d-9e0f-1a2b3c4d5e6f")


class DecisionCategory(str, enum.Enum):
    PRIORITY = "priority"
    # Its OWN category, not RISK: it is the only one the proposer can
    # currently turn into a real, targeted action, and collapsing it into
    # RISK would re-inherit the blank-email mapping that category used to
    # carry. Distinct category => distinct, explicit handling.
    WAITING_REPLY = "waiting_reply"
    # Each obligation category is its OWN, for the same reason WAITING_REPLY
    # is: the proposer can author a real, targeted follow-up for these, and
    # folding them into RISK would re-inherit that category's (removed)
    # blank-email mapping.
    FOLLOW_UP = "follow_up"
    DECISION_OWED = "decision_owed"
    RISK = "risk"
    DELEGATION = "delegation"
    OPPORTUNITY = "opportunity"


# Deterministic map from a reasoning signal type to a decision category.
_CATEGORY_FOR_SIGNAL = {
    SignalType.COMMITMENT_SLIPPING: DecisionCategory.RISK,
    SignalType.RELATIONSHIP_COOLING: DecisionCategory.RISK,
    SignalType.PROJECT_STALLED: DecisionCategory.RISK,
    SignalType.DECISION_UNRESOLVED: DecisionCategory.RISK,
    SignalType.WAITING_ON_USER_REPLY: DecisionCategory.WAITING_REPLY,
    SignalType.COMMITMENT_FOLLOW_UP: DecisionCategory.FOLLOW_UP,
    SignalType.DECISION_OWED: DecisionCategory.DECISION_OWED,
    SignalType.CUSTOMER_ENGAGEMENT_DROP: DecisionCategory.RISK,
    SignalType.MEETING_OVERLOAD: DecisionCategory.RISK,
    SignalType.OPPORTUNITY: DecisionCategory.OPPORTUNITY,
    SignalType.LEVERAGE_POINT: DecisionCategory.DELEGATION,
    SignalType.PRIORITY_LOAD: DecisionCategory.PRIORITY,
    SignalType.ORG_DRIFT: DecisionCategory.PRIORITY,
    SignalType.CONTEXT_SUMMARY: DecisionCategory.PRIORITY,
}


def category_for(signal_type: SignalType) -> DecisionCategory:
    return _CATEGORY_FOR_SIGNAL.get(signal_type, DecisionCategory.PRIORITY)


def compute_decision_id(
    org_id: uuid.UUID, reasoning_version: int, state_watermark: str | None, signal_id: uuid.UUID
) -> uuid.UUID:
    key = f"{org_id}|{reasoning_version}|{state_watermark or '-'}|{signal_id}"
    return uuid.uuid5(_DECISION_NAMESPACE, key)


def compute_brief_id(
    org_id: uuid.UUID, reasoning_version: int, state_watermark: str | None
) -> uuid.UUID:
    key = f"brief|{org_id}|{reasoning_version}|{state_watermark or '-'}"
    return uuid.uuid5(_DECISION_NAMESPACE, key)


@dataclass(frozen=True, slots=True)
class Decision:
    decision_id: uuid.UUID
    org_id: uuid.UUID
    title: str
    summary: str
    why: str
    evidence: dict[str, Any]
    priority_rank: int
    category: DecisionCategory
    reasoning_signal_ids: list[str]
    reasoning_version: int
    state_watermark: str | None
    created_at: datetime | None  # logical time = state.as_of (deterministic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": str(self.decision_id),
            "org_id": str(self.org_id),
            "title": self.title,
            "summary": self.summary,
            "why": self.why,
            "evidence": self.evidence,
            "priority_rank": self.priority_rank,
            "category": self.category.value,
            "reasoning_signal_ids": self.reasoning_signal_ids,
            "reasoning_version": self.reasoning_version,
            "state_watermark": self.state_watermark,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
