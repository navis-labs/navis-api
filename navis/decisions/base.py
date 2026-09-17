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
from collections.abc import Mapping
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
# STATUS: the view is now READ, not just wired. `what_matters_today` resolves
# the facts relevant to each signal (`_relevant_facts`), lets them nudge that
# signal's score (`_intelligence_bias`, bounded by the category bias the ranker
# already applies), and records exactly those facts on the decision as
# `supporting_insight_ids` — so the stored provenance and the applied nudge can
# never disagree. `compute_decision_id` is still untouched by any of it:
# corroboration changes a decision's RANK, never its identity. An absent or
# empty view still produces byte-identical output. See `DecisionService.decide`.


class InsightRelation(str, enum.Enum):
    """HOW an organizational insight relates to a decision. NEVER "caused".

    `what_matters_today._relevant_facts` has always resolved insights by two
    structurally different relationships and then flattened both into one
    undifferentiated `supporting_insight_ids` list. A surface reading that list
    has no way to tell them apart, so it renders both the same way — and the
    natural rendering of "supporting insight" is "this insight is why Navis
    raised this", which NEITHER relationship supports:

      * DERIVED_FROM_SAME_SIGNAL — the insight was folded FROM the very
        ReasoningSignal this decision ranks (`intelligence.service._provenance`
        recorded the signal id on the insight). That is a SHARED ORIGIN, not
        causation: the signal produced both, and the insight is corroborating
        evidence that the same finding survived aggregation.

      * SHARES_ENTITY — the insight names this decision's entity among the
        entities in its evidence. That is TOPICAL OVERLAP and nothing more. An
        org-wide footprint insight naming 110 entities shares an entity with
        almost every decision the org will ever produce.

      * LEARNED_FROM_OUTCOMES — the insight also names this signal, but it is a
        LEARNING-derived insight: a measurement over what happened when the
        organization ACTED on signals like this one, folded from
        `learning_events` and versioned by `learning_version`. It is a statement
        about outcome history, not about what is true right now, and the two are
        not interchangeable. Collapsing it into DERIVED_FROM_SAME_SIGNAL was the
        same flattening one level down: both matched on a signal id, and the id
        was the only thing they had in common.

        The distinction is decidable from stored data alone — an insight either
        belongs to `intelligence.insights.REALITY_INSIGHT_TYPES` or it does not
        — and it is decided by the composition root, which is the only layer
        allowed to know both vocabularies. See `IntelligenceFact.describes_state`.

    NOT IMPLEMENTED, deliberately: `supported_by_state`, `contextual_evidence`,
    `predicted_by_ml`, `contradicts`. No layer in this system stores any of them.
    Nothing computes contradiction between insights; no model contributes to
    Intelligence at all (`navis/intelligence/**`, `navis/reasoning/v2/**` and
    `navis/decisions/**` import no `navis.ml`, pinned by AST tests); and a
    state/context link would be a re-derivation of `entity_id`, which every
    decision already carries. Inventing any of them here would be exactly the
    fabricated provenance this enum exists to prevent.

    The distinction is preserved on the wire (`Decision.supporting_insights`) so
    a client can say "also observed across the organization", "about the same
    entity", or "here is how this went last time", rather than asserting a
    causal chain the backend does not store.
    """

    DERIVED_FROM_SAME_SIGNAL = "derived_from_same_signal"
    SHARES_ENTITY = "shares_entity"
    LEARNED_FROM_OUTCOMES = "learned_from_outcomes"


@dataclass(frozen=True, slots=True)
class SupportingInsight:
    """One insight, the RELATIONSHIP by which it reached this decision, and
    WHETHER IT IS STILL ABOUT THE SAME ORGANIZATION.

    `describes_current_snapshot` is the field that stops stale intelligence from
    reading as current corroboration, and the gap it closes is structural rather
    than hypothetical:

      * `compute_signal_id` hashes (org, reasoning_version, engine, type, entity,
        dedup_key) and NOT the state watermark, so the same cooling relationship
        produces the SAME signal id at every snapshot;
      * `orchestration.brief._project_intelligence` is deliberately non-fatal —
        "a projection failure costs an insight and never the brief" — so a brief
        can be built at watermark W2 while the newest stored reality generation
        is still at W1;
      * `_current_intelligence` then hands Decision the W1 generation, whose
        `supporting_reasoning_signal_ids` still match this pass's signals.

    The insight is then attached to a W2 decision as though it corroborated it,
    while its CONTENT ("6 relationships are cooling") describes a different
    organization. Both watermarks were already in hand and both were discarded.

    TRI-STATE, like `state.v2.models._SenderClassifiedView`:
      True  — a reality insight describing exactly this decision's snapshot
      False — a reality insight describing an EARLIER snapshot (stale corroboration)
      None  — a Learning-derived insight, which makes no claim about a snapshot
              at all; its `state_watermark` is whatever an execution happened to
              record on its way past and must not be read as one.
    """

    insight_id: str
    relation: InsightRelation
    # The insight's own type string (`intelligence.models.InsightType`), so a
    # client can name the FAMILY of evidence without a second lookup.
    insight_type: str
    # The insight's own published headline and how many things it folded,
    # verbatim. A citation nobody can read is a UUID; these are what let a
    # surface show the organizational context this decision sits in without a
    # second request, and without any layer rewording Intelligence's claim.
    title: str = ""
    sample_size: int = 0
    # The snapshot the insight itself recorded. Null when it recorded none.
    state_watermark: str | None = None
    describes_current_snapshot: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "relation": self.relation.value,
            "insight_type": self.insight_type,
            "title": self.title,
            "sample_size": self.sample_size,
            "state_watermark": self.state_watermark,
            "describes_current_snapshot": self.describes_current_snapshot,
        }


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
    # WHICH FAMILY this insight belongs to, as one primitive.
    #
    # True  — a REALITY insight: folded from the current State snapshot and the
    #         ReasoningSignals computed over it, versioned by `state_watermark`.
    #         Its watermark is a claim about which organization it describes.
    # False — a LEARNING insight: folded from `learning_events`, versioned by
    #         `learning_version`. It measures what happened when the organization
    #         acted, and its watermark is inherited from whichever execution
    #         recorded one — never a claim about a snapshot.
    #
    # Set by the composition root (`orchestration.brief._intelligence_view`) from
    # `intelligence.insights.REALITY_INSIGHT_TYPES`, because that is the only
    # layer permitted to know both vocabularies — `navis/decisions/**` may not
    # import `navis.intelligence` (AST-pinned). It arrives here as a bool for the
    # same reason every other cross-layer input does: the consumer owns the shape
    # it accepts, and nothing about the producer leaks through it.
    #
    # Defaults to False so a view built by an older caller keeps working: a fact
    # that does not claim to describe State cannot make a false claim about it.
    describes_state: bool = False
    # The insight's own published headline — "1 outstanding reply obligation".
    # Carried because a citation a founder can read is the difference between
    # organizational context and a bare UUID, and because REWORDING it anywhere
    # downstream would give one organizational fact two voices. Verbatim or
    # nothing.
    #
    # Deliberately the TITLE and not `summary`: a reality summary quotes the
    # engine's own measurements ("peak severity 0.65"), which is provenance for
    # a trace view and machine vocabulary on a founder's screen.
    title: str = ""
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


def obligation_counts(by_signal_type: Mapping[str, int]) -> dict[str, int]:
    """How many obligations of each kind Reasoning found. PURE.

    HOW MANY, versus the brief's HOW MANY WE ARE SHOWING YOU. A brief is
    ranked, deduped by entity and truncated to `brief_max_items`, so counting
    its decisions answers "how big is the card list", which is not the question
    "how many people are waiting on me". This counts the signals themselves,
    one per stored row, at the snapshot the brief was built from.

    INTERPRETATION IS EXCLUDED, by exactly the same map the brief uses
    (`category_for`) and no other rule. CONTEXT_SUMMARY, PRIORITY_LOAD,
    ORG_DRIFT, OPPORTUNITY and LEVERAGE_POINT ask nothing of anyone, so folding
    them into an obligation count would inflate it with observations — the same
    overclaim `_INTERPRETATION_ONLY` exists to keep out of the brief.

    Keyed by the raw `signal_type`, not by category: RISK covers six types, and
    "3 risks" cannot tell a founder whether a customer went quiet or a promise
    slipped. A type with no signals is ABSENT rather than zero — the caller
    reports whether the snapshot was readable at all, and a present key is
    always a real, resolvable count.

    An unrecognised type string (a signal stored under an older vocabulary) is
    skipped rather than counted: this function cannot know whether it was an
    obligation, and guessing would be the fallback-to-PRIORITY defect that
    `_CATEGORY_FOR_SIGNAL` was rewritten to remove.
    """
    counts: dict[str, int] = {}
    for raw, count in by_signal_type.items():
        try:
            signal_type = SignalType(raw)
        except ValueError:
            continue
        if category_for(signal_type) is None:
            continue
        if count > 0:
            counts[signal_type.value] = int(count)
    return counts


# ── Decision ↔ reasoning-observation agreement (P0-3) ─────────────────────────


@dataclass(frozen=True, slots=True)
class ProvenanceCheck:
    """Whether a decision and the reasoning conclusion it cites describe the SAME
    snapshot — proven field by field, never assumed.

    THE DEFECT THIS CLOSES. A decision serialises the in-memory signal from the
    pass it was ranked in, while its `reasoning_signal_ids` pointer resolves to a
    stored row. Those were the same row for every conclusion and the identity row
    kept only its FIRST detection, so live decision 4373a0b3 reports 11.85 days /
    confidence 0.8951 while the signal it cites says 7.16 / 0.7386. Both numbers
    were computed by the engines; the pointer simply led to the older one.

    `reasoning_signal_observations` now records the conclusion per snapshot, so
    the agreement is CHECKABLE. This reports the result rather than repairing it:
    a decision whose observation is missing is marked unverified, never quietly
    backfilled from an adjacent snapshot or from the identity row. Fail closed.

    `verified is False` is a real, actionable answer — it means the brief is
    serving a decision whose reasoning cannot be reproduced at its own watermark.
    """

    verified: bool
    # "" when verified. Otherwise the FIRST field that disagreed, or why the
    # observation could not be resolved at all.
    reason: str = ""
    signal_id: str | None = None
    state_watermark: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "reason": self.reason,
            "signal_id": self.signal_id,
            "state_watermark": self.state_watermark,
        }


# Every value a decision copies off its originating signal, paired with the
# observation attribute it must equal. `summary` is deliberately absent: the
# ranker sets it from the same `why`, so checking it twice would report one
# disagreement as two.
_PROVENANCE_FIELDS = ("confidence", "severity", "why", "evidence")


def check_decision_provenance(
    decision: Mapping[str, Any], observation: Any | None
) -> ProvenanceCheck:
    """PURE. Does `observation` reproduce what this decision claims?

    `observation` is duck-typed — anything carrying `state_watermark`,
    `confidence`, `severity`, `why` and `evidence`. That keeps this function free
    of any ORM import, so `navis/decisions/**` stays infrastructure-free (AST-pinned
    by `test_the_ranker_performs_no_io_and_imports_no_infrastructure`). `None`
    means the caller could not resolve one, which is itself a failure to verify.

    Floats are compared at the precision the decision PUBLISHES (4dp, set by
    `Decision.to_dict`), because that is the number a client actually reads. A
    stricter comparison would report a rounding artefact as a lineage break.
    """
    signal_ids = list(decision.get("reasoning_signal_ids") or ())
    signal_id = signal_ids[0] if signal_ids else None
    watermark = decision.get("state_watermark")

    if signal_id is None:
        return ProvenanceCheck(
            False, "decision cites no reasoning signal", None, watermark
        )
    if observation is None:
        return ProvenanceCheck(
            False,
            "no reasoning observation recorded at this decision's watermark",
            signal_id,
            watermark,
        )
    if observation.state_watermark != watermark:
        return ProvenanceCheck(
            False,
            f"observation describes watermark {observation.state_watermark!r}, "
            f"decision claims {watermark!r}",
            signal_id,
            watermark,
        )

    for field_name in _PROVENANCE_FIELDS:
        claimed = decision.get(field_name)
        recorded = getattr(observation, field_name, None)
        if field_name in ("confidence", "severity"):
            recorded = None if recorded is None else round(recorded, 4)
        elif field_name == "evidence":
            claimed = claimed or {}
            recorded = recorded or {}
        if claimed != recorded:
            return ProvenanceCheck(
                False,
                f"{field_name} disagrees: decision {claimed!r} vs observation {recorded!r}",
                signal_id,
                watermark,
            )

    return ProvenanceCheck(True, "", signal_id, watermark)


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
    # THE SAME INSIGHTS, each carrying HOW it got here. See `InsightRelation`:
    # neither relationship is causal, and collapsing them into the bare id list
    # above is what let a surface present corroboration as cause. Ordered
    # identically to `supporting_insight_ids` so the two are trivially zippable.
    supporting_insights: tuple[SupportingInsight, ...] = ()
    # WHICH REASONING CONCLUSION THIS IS — the originating ReasoningSignal's
    # `signal_type`, verbatim.
    #
    # `category` is not a substitute and never was: RISK covers five distinct
    # signal types (COMMITMENT_SLIPPING, RELATIONSHIP_COOLING, PROJECT_STALLED,
    # CUSTOMER_ENGAGEMENT_DROP, DECISION_UNRESOLVED, MEETING_OVERLOAD), so a
    # client asking "what did Navis actually conclude?" could only read the
    # engine's free-form `why` string and guess. The value was right there on
    # the signal the ranker holds; like `entity_type`/`entity_id` before it, it
    # simply never left this engine.
    #
    # Null only when a decision was built without an originating signal.
    signal_type: str | None = None
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
    # WHAT ORGANIZATIONAL THING THIS DECISION IS ABOUT.
    #
    # The originating ReasoningSignal's `entity_type` / `entity_id`, passed
    # through unchanged. Same id space as `state/v2/scope.py::EntityRef` — both
    # read `.entity_id` off the same State entity view — so this is the join
    # between a decision and the entity a client already has on screen.
    #
    # They were dropped here, and only here. The ranker holds both (it dedupes
    # on `(entity_type, entity_id)` so two engines firing on one customer
    # produce one card), Reasoning persists both, Intelligence carries both in
    # `RealitySignalRow` — and then the Decision this all produced named no
    # entity at all. A client could therefore not answer "which person is this
    # decision about?" except by string-matching `evidence`, which is the
    # engine's own free-form dict and not a contract.
    #
    # Same additive shape as `supporting_insight_ids` above, for the same
    # reasons: `daily_briefs.decisions` is JSONB so the contract widens with no
    # migration; `execution/proposer.py` reads only named keys; identity is
    # unaffected because `compute_decision_id` hashes (org, reasoning_version,
    # watermark, signal_id) and never these.
    #
    # Nullable because a signal can legitimately have no entity — `meta.py` and
    # `risk.py` emit org-wide signals with `entity_id=None`, and the ranker
    # deliberately does not collapse those. Null means "org-wide", which is a
    # real answer and must stay distinguishable from "not recorded".
    entity_type: str | None = None
    entity_id: str | None = None

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
            # The same insights, each stating the relationship it actually has.
            # Always present — an empty list means "no insight is about this",
            # which a client must be able to tell from "this build predates the
            # relation". Read this in preference to the bare id list above: it
            # is the only form that does not invite a causal reading.
            "supporting_insights": [s.to_dict() for s in self.supporting_insights],
            # Which reasoning conclusion this decision carries, verbatim. Always
            # present; null means the decision has no originating signal.
            "signal_type": self.signal_type,
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
            # Always present, never conditional — same rule as the two above.
            # Null means the originating signal was org-wide, not that this
            # build predates entity provenance.
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
        }
