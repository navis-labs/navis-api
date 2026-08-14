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


# A SIGNAL IS NOT A DECISION.
#
# Reasoning emits two kinds of signal, and only one of them is a decision:
#
#   * DECISION POINTS — something is owed, slipping, unanswered or at risk.
#     A person has to exercise judgment: reply, approve, prioritize,
#     delegate, or commit. These become Decisions.
#
#   * INTERPRETATION — true, useful observations about the organization
#     that ask nothing of anyone. Engines/meta.py says so in its own
#     docstring: "These produce interpretation, NOT decisions ... it does
#     NOT rank actions or recommend (that is Decision's job)."
#     These stay ReasoningSignals: available as context and evidence,
#     never promoted to the brief.
#
# This map is the boundary, and membership in it is the ONLY test. A
# signal type absent here is not a decision — see `category_for`.
#
# ── Why this map used to be wrong ─────────────────────────────────────
#
# It previously listed all 14 signal types, and `category_for` fell back
# to PRIORITY for anything unmapped. Two consequences, both visible in
# production briefs:
#
#   * CONTEXT_SUMMARY ("Organizational context") is emitted with
#     confidence=1.0 and, as a PRIORITY, drew the largest category bias
#     in the ranker — so a bare recitation of entity counts outranked
#     genuine obligations in essentially every brief.
#   * OPPORTUNITY ("Active inbound engagement") fires for any
#     relationship with an inbound direction in the last two days, with
#     no automation filter. Promotional mail, invoices, newsletters and
#     calendar notifications each produced an identically-titled
#     "decision" that asked nothing of the founder.
#
# The fallback also meant any NEW signal type silently became a PRIORITY
# decision the moment it was added. Absence now excludes.
_CATEGORY_FOR_SIGNAL = {
    # Something was promised and is drifting.
    SignalType.COMMITMENT_SLIPPING: DecisionCategory.RISK,
    SignalType.COMMITMENT_FOLLOW_UP: DecisionCategory.FOLLOW_UP,
    # A relationship or project is decaying and someone must intervene.
    SignalType.RELATIONSHIP_COOLING: DecisionCategory.RISK,
    SignalType.PROJECT_STALLED: DecisionCategory.RISK,
    SignalType.CUSTOMER_ENGAGEMENT_DROP: DecisionCategory.RISK,
    # A judgment is explicitly owed.
    SignalType.DECISION_UNRESOLVED: DecisionCategory.RISK,
    SignalType.DECISION_OWED: DecisionCategory.DECISION_OWED,
    # A person is waiting on a human reply. engines/waiting.py already
    # excludes automated, promotional and no-reply senders before this
    # ever fires — that filter is the org's single notion of "is this a
    # real person", and nothing here second-guesses it.
    SignalType.WAITING_ON_USER_REPLY: DecisionCategory.WAITING_REPLY,
    # Calendar load the founder has to actively prune.
    SignalType.MEETING_OVERLOAD: DecisionCategory.RISK,
}

# Interpretation-only. Listed explicitly so the exclusion is a decision
# someone made, not an oversight, and so a reader can see the whole
# vocabulary in one place. Keeping these OUT of the brief is what makes
# a quiet Today trustworthy.
_INTERPRETATION_ONLY = frozenset(
    {
        SignalType.CONTEXT_SUMMARY,   # "Organizational context" — a state recitation
        SignalType.PRIORITY_LOAD,     # aggregate of the risk signals already listed above
        SignalType.ORG_DRIFT,         # "Rising commitment volume" — a trend, not an ask
        SignalType.OPPORTUNITY,       # "Active inbound engagement" — an observation
        SignalType.LEVERAGE_POINT,    # "Hub company: X" — a structural fact
    }
)

# Every signal type must be deliberately classified as one or the other.
# A new type added to SignalType without a home here fails fast in tests
# rather than silently appearing in founders' briefs.
assert set(_CATEGORY_FOR_SIGNAL) | _INTERPRETATION_ONLY == set(SignalType), (
    "every SignalType must be classified as decision-worthy or interpretation-only"
)


def category_for(signal_type: SignalType) -> DecisionCategory | None:
    """The decision category for a signal, or None if it is not a decision.

    None is a real answer, not a failure: interpretation signals are
    valid reasoning output that simply do not belong in a brief.
    """
    return _CATEGORY_FOR_SIGNAL.get(signal_type)


def is_decision_worthy(signal_type: SignalType) -> bool:
    """Does this signal represent a point where a person must decide?"""
    return signal_type in _CATEGORY_FOR_SIGNAL


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
