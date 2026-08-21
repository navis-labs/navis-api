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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from navis.reasoning.v2.contract import SignalType

_DECISION_NAMESPACE = uuid.UUID("c4f1e2d3-5a6b-5c8d-9e0f-1a2b3c4d5e6f")


# ── What the organization knows about itself (the second input) ───────────────
#
# Decision has only ever consumed ReasoningSignal[], so it could rank what is
# true of the organization RIGHT NOW and nothing about what the organization has
# LEARNED about itself. Intelligence holds that answer, and Decision may not
# import it — the dependency would invert the pipeline and make the ranker a
# reader of an append-only projection it has no business querying.
#
# So it arrives the way every other cross-layer input in this system arrives: as
# a flat, primitive-only projection assembled by the composition root and handed
# in. `navis.orchestration.brief` is that root; it already reads State for
# Reasoning, builds `OrganizationalRealityView` for Intelligence, and resolves
# Adaptive policy for Decision. This is the fourth join at the same seam.
#
# THE DIRECTION IS DELIBERATE and mirrors `intelligence.insights`'s reality
# contract exactly: the CONSUMER owns the shape it accepts, the producer owns
# nothing about it. Nothing below carries an ORM row, a session, a repository or
# an IntelligenceInsight, so `navis/decisions/**` stays free of Intelligence.
#
# THIS PHASE IS THE CONTRACT AND THE WIRING. No ranking reads a fact yet, and
# that is the point: `compute_decision_id`, the ordering, and the brief payload
# are all provably unchanged by handing Decision this view. See
# `DecisionService.decide`.


@dataclass(frozen=True, slots=True)
class IntelligenceFact:
    """One current organizational insight, flattened to primitives.

    Exactly the fields an insight already carries — nothing is recomputed,
    re-derived or enriched, and a fact that recorded no supporting signals says
    so with an empty tuple rather than an invented one.

    `supporting_reasoning_signal_ids` is hoisted out of `evidence` because it is
    the JOIN between the two things Decision now holds: the signals it ranks and
    the insights folded from them. It is the one field a consumer would otherwise
    have to reach into a JSONB blob to find.
    """

    insight_id: str
    insight_type: str
    key: str
    summary: str
    confidence: float
    sample_size: int
    # The engine's own measurements, verbatim (named entities included). A dict
    # rather than a frozen mapping for the same reason `RealitySignalRow` uses
    # one: the builder hands over a copy, so the ORM row's JSONB can never be
    # mutated through this view.
    evidence: dict[str, Any] = field(default_factory=dict)
    supporting_reasoning_signal_ids: tuple[str, ...] = ()
    state_watermark: str | None = None
    learning_version: int = 0


@dataclass(frozen=True, slots=True)
class IntelligenceView:
    """The organization's CURRENT insights, as Decision is allowed to see them.

    `facts` is a tuple, not a list: a frozen dataclass wrapped around a mutable
    list is not actually frozen, and this is an input to a deterministic ranker.

    `as_of` is logical time — the newest `created_at` among the folded insights,
    which is State's clock for a reality fold and Learning's for a Learning fold.
    Never wall clock, so the same stored insights always build the same view.
    """

    org_id: uuid.UUID
    as_of: datetime | None = None
    facts: tuple[IntelligenceFact, ...] = ()


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
    # WHICH ORGANIZATIONAL FACTS informed this decision, sorted. Empty when
    # Intelligence contributed nothing — which is the normal case and stays
    # honestly empty rather than being filled with everything the org knows.
    #
    # Rides in the existing serialized representation rather than in a new
    # column: `daily_briefs.decisions` is JSONB, so the contract widens with no
    # migration, and `execution/proposer.py` reads only named keys so it is
    # unaffected. Deliberately NOT written into `evidence` — that dict is the
    # ReasoningSignal's own, shared by reference and already persisted, and
    # injecting decision-scoped provenance into it would mutate a stored signal.
    supporting_insight_ids: tuple[str, ...] = ()
    # THE ORIGINATING SIGNAL'S OWN confidence and severity, passed through
    # unchanged. Both are the 0..1 floats the emitting engine assigned (see
    # reasoning/v2/contract.py); `weight` is their product and is what the
    # ranker sorts on.
    #
    # They are carried because dropping them left `priority_rank` as the only
    # thing a client could order or judge by, and rank is an ORDINAL — the
    # position a decision landed in after sorting. Frontends filled the gap by
    # mapping rank back into a percentage and calling it "confidence", which is
    # the same number wearing a disguise. The real values existed all along on
    # the signal this decision was built from; they simply stopped here.
    #
    # NOT a calibrated probability. An engine sets confidence by how directly it
    # OBSERVED the thing (a quoted promise with a quoted deadline scores 0.85;
    # an inference from silence scores lower), and severity by how much it
    # matters. Present them as the engine's own values, never as a percentage
    # likelihood.
    #
    # Optional, and defaulted, so every existing construction of this dataclass
    # keeps working and a decision built without a signal reports null rather
    # than a fabricated 0.0.
    confidence: float | None = None
    severity: float | None = None

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
            # Always present, never conditional: a reader must be able to tell
            # "no insight informed this" from "this build predates provenance".
            "supporting_insight_ids": list(self.supporting_insight_ids),
            # Same rule, same reason: always present, null when this decision
            # has no originating signal to take them from. A client must be able
            # to distinguish "the engine stated no confidence" from "this build
            # does not report it" — the second is what forces a client to invent
            # one from `priority_rank`.
            #
            # Rounded to match ReasoningSignal.to_dict(), so the same underlying
            # value is byte-identical wherever it is read.
            "confidence": (
                round(self.confidence, 4) if self.confidence is not None else None
            ),
            "severity": (
                round(self.severity, 4) if self.severity is not None else None
            ),
        }
