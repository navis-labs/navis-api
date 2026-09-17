"""The six intelligence derivations — pure functions over a LearningView.

Each emits FACTS (what the org has learned about itself), never recommendations.
Deterministic: sorted iteration, no clock/random/uuid4/LLM. Input is Learning only.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from navis.intelligence.features import compute_features
from navis.intelligence.models import InsightType

MIN_SAMPLES = 5


@dataclass(frozen=True, slots=True)
class AggRow:
    key: str
    successes: int
    failures: int
    neutrals: int
    total: int

    @property
    def decisive(self) -> int:
        return self.successes + self.failures

    @property
    def rate(self) -> float | None:
        return round(self.successes / self.decisive, 4) if self.decisive else None


@dataclass(frozen=True, slots=True)
class PatternRow:
    signal_id: str
    kind: str
    success_rate: float
    samples: int


@dataclass(frozen=True, slots=True)
class LearningView:
    org_id: uuid.UUID
    learning_version: int
    as_of: datetime | None
    signal_aggs: list[AggRow]
    decision_aggs: list[AggRow]
    outcome_aggs: list[AggRow]
    patterns: list[PatternRow]
    # key -> sorted supporting learning_event_id strings
    lineage: dict[str, list[str]] = field(default_factory=dict)
    # The State snapshot this learning was true against, carried from
    # LearningEvent.state_watermark. An insight without it can be traced to the
    # events that support it but not to the organizational state they described,
    # which is the difference between "we observed this" and "this was true of
    # the org as of X". Still Learning-only — the watermark is a value Learning
    # already records, not a new read of State.
    state_watermark: str | None = None
    # key -> the ReasoningSignal ids whose outcomes fed this key. Answers WHY
    # something was surfaced, one hop below the learning events.
    signal_lineage: dict[str, list[str]] = field(default_factory=dict)


# ── Current organizational reality (the second input) ─────────────────────────
#
# Intelligence consumed Learning alone, so it could only ever describe HISTORICAL
# execution outcomes — "N actions were attempted" — never "this is true of the
# organization right now". State and Reasoning hold that answer and Intelligence
# may not import either of them (see the Core Law in repository.py, pinned by an
# AST test), so it arrives the same way `evidence` arrives at Reasoning and
# `Grounding` arrives at ML: as a flat, primitive-only projection assembled by
# the composition root and handed in.
#
# `navis.orchestration.brief` is that root — it already reads State, runs
# Reasoning and resolves Adaptive policy for exactly this reason. Nothing below
# carries an ORM row, an OrganizationalState, a ReasoningSignal or a session,
# so this module stays a pure function of its inputs.
#
# PHASE 4.0 IS THE CONTRACT ONLY. No generator reads a reality view yet; the one
# thing it changes today is that an insight's `state_watermark` comes from the
# State snapshot it describes instead of being inherited from a LearningEvent
# (see service._provenance).


@dataclass(frozen=True, slots=True)
class RealitySignalRow:
    """One ReasoningSignal, flattened to primitives.

    Carries the engine's own numbers and its own `evidence` verbatim. Nothing is
    recomputed, re-derived or enriched here — a signal that recorded no
    `thread_id` says nothing rather than saying null.
    """

    signal_id: str
    signal_type: str
    entity_type: str | None
    entity_id: str | None
    confidence: float
    severity: float
    engine_name: str
    engine_version: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextLink:
    """One persisted edge from a context entity to another entity.

    `state.v2.scope.EntityLink`, flattened to primitives — same three strings, no
    State import. The role is State's own link key ("owner", "commitments",
    "meetings") and the type is resolved by Context from its closed role map,
    never inferred here.

    WHY INTELLIGENCE NEEDS IT. Without edges, the only relation an insight can
    express between two entities is "both produced a signal of the same type",
    which is a GROUP BY and not an organizational situation. With them, an
    insight can say that the cooling relationship and the slipping commitment are
    the same counterparty — because State recorded that they are.
    """

    role: str
    entity_type: str
    entity_id: str


@dataclass(frozen=True, slots=True)
class ContextEntityRow:
    """One entity CONTEXT selected as relevant, with where it was learned.

    THE GAP THIS CLOSES. `service._provenance` says it in its own words: "NOT
    AVAILABLE HERE: memory_ids / signal_event_ids ... the chain stops at the
    ReasoningSignal. A reality view does not close that gap either — it carries
    each signal's own `evidence` verbatim and nothing more." So an insight could
    name an affected entity and could not say which remembered fact, or which
    captured business-system event, it was known from. The provenance chain
    Navis is built on broke one hop short of the source.

    Context already holds exactly that. `state.v2.scope.ScopedView` ranks the
    entities relevant to an intent and carries `citations[entity_id] =
    {memory_ids, signal_ids}` for every one — read straight off each entity's own
    `provenance`, never re-derived. It simply had no consumer inside the
    pipeline: `scope_for` was called by the `/context` endpoint and by Ask Navis,
    and by nothing between State and Intelligence.

    A SELECTION, NOT A SUBSTITUTE FOR STATE. Context returns the top-N per entity
    type, so an insight may legitimately name an entity Context did not select —
    that entity simply carries no lineage here, which is honest absence rather
    than a gap to fill. Reasoning deliberately keeps reading the FULL State: an
    exhaustive detector fed a top-N selection would silently stop finding
    obligations, which is the opposite of what a selection is for.

    Primitives only, like every other row in this contract: no ORM object, no
    `ScopedView`, no session. `navis/intelligence/**` still imports no State.
    """

    entity_type: str
    entity_id: str
    # Where Context placed it and why — `scope_for`'s own rank and reason
    # strings ("open, oldest activity", "stalest interaction"), verbatim.
    rank: int
    reason: str
    # The memory records this entity's current state was folded from, and the
    # captured business-system events those records came from. Bounded by the
    # composition root; empty when the entity recorded no provenance.
    memory_ids: tuple[str, ...] = ()
    signal_event_ids: tuple[str, ...] = ()
    # The entity's persisted edges (see `ContextLink`). Defaulted so every
    # existing construction site stays valid; an empty tuple means State recorded
    # no edge for this entity, never that edges are unavailable.
    links: tuple[ContextLink, ...] = ()


@dataclass(frozen=True, slots=True)
class OrganizationalRealityView:
    """What is true of the organization NOW, as Intelligence is allowed to see it.

    `state_watermark` is the AUTHORITATIVE one — `OrganizationalState.watermark`,
    the memory record State actually folded to. It is deliberately not the value
    Learning carries: `LearningEvent.state_watermark` is whatever an execution
    happened to record (live, both calendar executions recorded NULL), so the
    newest-non-null pick can stamp an insight with an unrelated action's
    watermark.

    `signals` is a tuple, not a list: this is a versioning key's input and a
    frozen dataclass around a mutable list is not actually frozen.

    `reasoning_version` is part of the identity of a reality insight, not
    decoration. State and Reasoning are TWO independent clocks over the same
    facts, and an insight is a function of both: reasoning v4 over an unchanged
    snapshot reaches different conclusions than v3 did. Versioning on the
    watermark alone made those two collide — live, the v4 relevance fix computed
    the same `insight_id`s as v3, and because the table is append-only with
    ON CONFLICT DO NOTHING the corrected rows were silently discarded and the
    stale "4 open relationship cooling risks" stayed served until a new memory
    record happened to move the watermark.
    """

    org_id: uuid.UUID
    as_of: datetime | None          # logical time = state.as_of, never wall clock
    state_watermark: str | None
    entity_counts: dict[str, int] = field(default_factory=dict)
    signals: tuple[RealitySignalRow, ...] = ()
    # Defaulted so every existing construction site stays valid; the composition
    # root supplies the real value.
    reasoning_version: int = 0
    # CONTEXT — the relevant slice of State, with its citations. See
    # `ContextEntityRow`. Defaulted to empty so every existing construction site
    # stays valid and produces byte-identical insights; an org whose Context is
    # unavailable loses lineage depth and never gains a fabricated id.
    context: tuple[ContextEntityRow, ...] = ()


@dataclass(frozen=True, slots=True)
class InsightDraft:
    insight_type: InsightType
    key: str
    title: str
    summary: str
    evidence: dict[str, Any]
    sample_size: int
    confidence: float
    supporting: list[str]
    # Which engine produced this draft. Defaulted to the Learning projection so
    # the six original generators are untouched; reality engines set their own.
    engine_name: str = "intelligence.learning"
    engine_version: str = "1"
    # The ReasoningSignal ids this draft was folded from. Empty for every
    # Learning-derived draft — those are folds over outcomes, not over signals —
    # and that emptiness is honest rather than a gap to fill.
    supporting_signals: tuple[str, ...] = ()


def _confidence(decisive: int) -> float:
    return round(min(1.0, decisive / 100.0), 4)


# 1. Signal Intelligence -------------------------------------------------------

def signal_intelligence(view: LearningView) -> list[InsightDraft]:
    out = []
    for agg in sorted(view.signal_aggs, key=lambda a: a.key):
        if agg.decisive < MIN_SAMPLES:
            continue
        rate = agg.rate or 0.0
        out.append(
            InsightDraft(
                insight_type=InsightType.SIGNAL,
                key=agg.key,
                title=f"Signal preceded success {rate:.0%}",
                summary=(
                    f"When this reasoning signal fired, the outcome was positive "
                    f"{agg.successes}/{agg.decisive} times."
                ),
                evidence={"successes": agg.successes, "failures": agg.failures, "success_rate": rate},
                sample_size=agg.decisive,
                confidence=_confidence(agg.decisive),
                supporting=view.lineage.get(agg.key, []),
            )
        )
    return out


# 2. Decision Intelligence -----------------------------------------------------

def decision_intelligence(view: LearningView) -> list[InsightDraft]:
    out = []
    for agg in sorted(view.decision_aggs, key=lambda a: a.key):
        if agg.decisive < MIN_SAMPLES:
            continue
        rate = agg.rate or 0.0
        out.append(
            InsightDraft(
                insight_type=InsightType.DECISION,
                key=agg.key,
                title=f"Decision produced success {rate:.0%}",
                summary=(
                    f"Outcomes attributed to this decision were positive "
                    f"{agg.successes}/{agg.decisive} times."
                ),
                evidence={"successes": agg.successes, "failures": agg.failures, "success_rate": rate},
                sample_size=agg.decisive,
                confidence=_confidence(agg.decisive),
                supporting=view.lineage.get(agg.key, []),
            )
        )
    return out


# 3. Team Intelligence (from learning_patterns) --------------------------------

def team_intelligence(view: LearningView) -> list[InsightDraft]:
    f = compute_features(view)
    if f.total_patterns == 0:
        return []
    samples = sum(p.samples for p in view.patterns)
    return [
        InsightDraft(
            insight_type=InsightType.TEAM,
            key="signal_reliability",
            title=f"Signal judgment reliable in {f.reliable_patterns}/{f.total_patterns} patterns",
            summary=(
                f"The team's reasoning signals proved reliable in {f.reliable_patterns} of "
                f"{f.total_patterns} recurring patterns."
            ),
            evidence={"reliable": f.reliable_patterns, "total_patterns": f.total_patterns},
            sample_size=samples,
            confidence=_confidence(f.total_patterns * 10),
            supporting=[],
        )
    ]


# 4. Customer Intelligence (from outcome aggregates) ---------------------------

def customer_intelligence(view: LearningView) -> list[InsightDraft]:
    f = compute_features(view)
    if f.executed_attempts == 0:
        return []
    engaged, attempts = f.positive_outcomes, f.executed_attempts
    rate = f.engagement_rate
    assert rate is not None  # executed_attempts > 0 guarantees a rate
    return [
        InsightDraft(
            insight_type=InsightType.CUSTOMER,
            key="engagement",
            title=f"Customer engagement followed {rate:.0%} of actions",
            summary=(
                f"Replies/meetings followed {engaged} of {attempts} executed actions — "
                f"a {rate:.0%} engagement rate."
            ),
            evidence={"engaged": engaged, "attempts": attempts, "engagement_rate": rate},
            sample_size=attempts,
            confidence=_confidence(attempts),
            supporting=[],
        )
    ]


# 5. Organizational Intelligence -----------------------------------------------

def organizational_intelligence(view: LearningView) -> list[InsightDraft]:
    f = compute_features(view)
    decisive = f.decisive_outcomes
    if decisive == 0:
        return []
    rate = f.outcome_positive_rate
    assert rate is not None  # decisive > 0 guarantees a rate
    return [
        InsightDraft(
            insight_type=InsightType.ORGANIZATIONAL,
            key="outcome_profile",
            title=f"{rate:.0%} of decisive outcomes were positive",
            summary=(
                f"Across {decisive} decisive outcomes, {rate:.0%} were positive; "
                f"{f.total_patterns} recurring signal patterns have emerged."
            ),
            evidence={"positive": f.positive_outcomes, "negative": f.negative_outcomes, "patterns": f.total_patterns},
            sample_size=decisive,
            confidence=_confidence(decisive),
            supporting=[],
        )
    ]


# 6. Adaptation Intelligence (the single strongest learned fact) ---------------

def adaptation_intelligence(view: LearningView) -> list[InsightDraft]:
    candidates = [a for a in view.signal_aggs if a.decisive >= MIN_SAMPLES]
    if not candidates:
        return []
    # Deterministic pick: best success_rate, then most samples, then key.
    best = sorted(candidates, key=lambda a: (-(a.rate or 0.0), -a.decisive, a.key))[0]
    rate = best.rate or 0.0
    return [
        InsightDraft(
            insight_type=InsightType.ADAPTATION,
            key=best.key,
            title="Learned organizational strength",
            summary=(
                f"This organization has learned it consistently succeeds when this signal "
                f"fires — {rate:.0%} positive over {best.decisive} outcomes."
            ),
            evidence={"signal_key": best.key, "success_rate": rate, "samples": best.decisive},
            sample_size=best.decisive,
            confidence=_confidence(best.decisive),
            supporting=view.lineage.get(best.key, []),
        )
    ]


_GENERATORS = (
    signal_intelligence,
    decision_intelligence,
    team_intelligence,
    customer_intelligence,
    organizational_intelligence,
    adaptation_intelligence,
)

# Engine identity for the six generators above, stamped on every row they
# produce (IntelligenceInsight.engine_name / engine_version).
#
# ONE identity for all six, not one each — the same discipline as
# REASONING_VERSION, which versions fourteen engines together. They share an
# input, a threshold (MIN_SAMPLES) and a confidence function, so they are one
# projection that happens to emit six insight types; versioning them apart would
# claim an independence they do not have.
#
# Bump when any generator's LOGIC or thresholds change. That does NOT change an
# insight_id (engine identity is not in the id key, on purpose): a version bump
# is a reason to re-derive and compare against the old row, not a reason to mint
# a parallel row for the same fact.
LEARNING_ENGINE_NAME = "intelligence.learning"
LEARNING_ENGINE_VERSION = "1"


# ══════════════════════════════════════════════════════════════════════════════
# REALITY INTELLIGENCE — organizational interpretation over CURRENT state
# ══════════════════════════════════════════════════════════════════════════════
#
# These fold `OrganizationalRealityView` — State's entity counts plus the
# ReasoningSignals already computed over that State — into organizational
# interpretations. They answer "what is true of this organization right now",
# where the Learning generators above answer "what has it learned from its own
# outcomes".
#
# THE CONCLUSIONS ARE NOT RECOMPUTED HERE. Reasoning has already decided that a
# relationship is cooling, a commitment is slipping, a window is warm — with its
# own thresholds, its own confidence and its own evidence, versioned by
# REASONING_VERSION. These engines aggregate and interpret those conclusions;
# they never re-derive them, never re-threshold them and never second-guess them.
# That is what keeps one notion of "is this relationship cooling" in the codebase.
#
# NO FABRICATION. Every engine is gated on evidence actually being present:
#   * no signals of a kind          -> no insight of that kind
#   * no entities of a type         -> no insight about them
#   * no prior snapshot             -> no trend (a single snapshot is not a trend)
#   * no temporal series / model    -> no forecast, at all
#   * no behavioural evidence       -> no behaviour insight, at all
# An organization with nothing to say about it gets no rows, and that is the
# correct output rather than a failure.
#
# SIGNAL TYPE LITERALS. These are `navis.reasoning.v2.contract.SignalType`
# values written as strings, because `navis/intelligence/**` may not import
# `navis.reasoning` (an AST invariant — see tests/intelligence). Exactly the
# pattern `reasoning/v2/engines/meta.py` uses for the Signals importance
# vocabulary. Drift is not left to luck: a test asserts every literal below is a
# real SignalType and fails if either side moves.

# Decay + duty signals that describe organizational RISK. Grouped, not merged:
# each keeps its own insight so "6 relationships cooling" never hides inside a
# single undifferentiated risk count.
_RISK_SIGNAL_TYPES = frozenset({
    "commitment_slipping",
    "relationship_cooling",
    "project_stalled",
    "customer_engagement_drop",
    "decision_unresolved",
    "meeting_overload",
})
# Warm windows and structural advantages. Reasoning emits both and Decision
# discards both as interpretation — they ask nothing of anyone, which is exactly
# what makes them Intelligence's material rather than a brief's.
_OPPORTUNITY_SIGNAL_TYPES = frozenset({"opportunity", "leverage_point"})
# Obligations attached to a promise. `commitment_follow_up` is the only signal
# in the system that carries a real, stated deadline (`due_at` is quoted from
# the message); `commitment_slipping` covers UNDATED promises by age alone.
_COMMITMENT_SIGNAL_TYPES = frozenset({"commitment_slipping", "commitment_follow_up"})
# Debts the organization owes a NAMED counterparty right now: an unanswered
# person, and a judgment somebody asked for and has not received. Both are
# observed rather than inferred — `engines/waiting.py` reads the thread's own
# folded `direction`, and `engines/obligations.py` reads a recorded request — and
# both are decision-worthy without being RISK. See `obligation_intelligence`.
_OBLIGATION_SIGNAL_TYPES = frozenset({"waiting_on_user_reply", "decision_owed"})
# What is owed, in the organization's language rather than the engine's. Keyed by
# the same literals so a new obligation type cannot be added to the set without
# deciding what to call it.
_OBLIGATION_NOUNS = {
    "waiting_on_user_reply": "reply",
    "decision_owed": "decision",
}
_RELATIONSHIP_COOLING = "relationship_cooling"
_OPPORTUNITY = "opportunity"
_ORG_DRIFT = "org_drift"
_PRIORITY_LOAD = "priority_load"

REALITY_ENGINE_VERSION = "1"

# Every insight type below is versioned by `state_watermark` rather than by
# `learning_version`. `service.draft_to_values` reads this set to pick the
# version key, so membership here is the ONLY thing that decides which clock an
# insight is versioned against.
REALITY_INSIGHT_TYPES = frozenset({
    InsightType.RISK,
    InsightType.OPPORTUNITY,
    InsightType.RELATIONSHIP,
    InsightType.COMMITMENT,
    InsightType.OBLIGATION,
    InsightType.STATUS,
    InsightType.TREND,
    InsightType.SITUATION,
})

# How many affected entities an insight names in its evidence. Bounded so one
# pathological org cannot write an unbounded JSONB blob; the COUNT is always
# exact and never truncated, only the enumeration is.
_MAX_NAMED_ENTITIES = 20


def _by_type(reality, wanted: frozenset[str] | str) -> list:
    """Contributing signals of the wanted type(s), in deterministic id order."""
    types = {wanted} if isinstance(wanted, str) else wanted
    return sorted(
        (s for s in reality.signals if s.signal_type in types),
        key=lambda s: s.signal_id,
    )


def _label(signal) -> str:
    """A human-readable name for what a signal is about, or its entity id.

    Reads ONLY keys the reasoning engines actually write (see their `evidence`
    dicts). Never constructs a name, never guesses one from an id, and falls
    back to the entity id — which is a real identifier — rather than to a
    placeholder that would read as a name.

    `speaker_email` is load-bearing: it is the ONLY human-readable field
    `commitment_slipping` carries, and without it every slipping commitment
    labelled itself with a raw UUID.
    """
    e = signal.evidence or {}
    for key in (
        "person_email", "speaker_email", "domain",
        "subject", "commitment", "decision",
    ):
        value = e.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return signal.entity_id or signal.signal_type


def _context_index(reality) -> dict[tuple[str | None, str | None], Any]:
    """(entity_type, entity_id) -> the ContextEntityRow for it, if any.

    Rebuilt per call rather than threaded through every generator: Context is
    capped at a few dozen rows and the generators run once per pass, so this is
    a handful of dict inserts against a signature change in six places.
    """
    return {(c.entity_type, c.entity_id): c for c in getattr(reality, "context", ())}


def _entities(signals: list, reality=None) -> list[dict[str, Any]]:
    """The affected entities, named and bounded, sorted for determinism.

    Each entity carries its CONTEXT lineage when Context selected it: the memory
    records its current state was folded from, and the business-system events
    those came from. That is the hop the provenance chain was missing — an
    insight could name an entity and not say where it was learned.

    Nothing is invented. An entity Context did not select, or one that recorded
    no provenance, simply carries no lineage keys at all: absence is honest,
    where an empty list would read as "we looked and there was none".

    `reality` is optional so every existing caller and test keeps working and
    keeps producing the same entity dicts it did before.
    """
    index = _context_index(reality) if reality is not None else {}
    out: list[dict[str, Any]] = []
    for s in signals[:_MAX_NAMED_ENTITIES]:
        entity: dict[str, Any] = {
            "entity_type": s.entity_type,
            "entity_id": s.entity_id,
            "label": _label(s),
            "severity": round(s.severity, 4),
            "signal_id": s.signal_id,
        }
        ctx = index.get((s.entity_type, s.entity_id))
        if ctx is not None:
            # Why CONTEXT considered this entity relevant, in its own words.
            entity["context_rank"] = ctx.rank
            entity["context_reason"] = ctx.reason
            if ctx.memory_ids:
                entity["memory_ids"] = list(ctx.memory_ids)
            if ctx.signal_event_ids:
                entity["signal_event_ids"] = list(ctx.signal_event_ids)
        out.append(entity)
    return out


def _composite_confidence(signals: list) -> float:
    """Confidence in the COMPOSITE claim "all N of these hold".

    The minimum, not the mean. Each contributing signal carries its own
    detection confidence; a statement about all of them together can be no
    stronger than its weakest member. Averaging would let one certain detection
    launder several shaky ones into a confident-looking organizational claim.
    """
    return round(min(s.confidence for s in signals), 4)


def _peak_severity(signals: list) -> float:
    """The org's exposure is set by its WORST instance, so severity is the max."""
    return round(max(s.severity for s in signals), 4)


def _signal_ids(signals: list) -> tuple[str, ...]:
    return tuple(s.signal_id for s in signals)


# Entity-type nouns whose plural is not "+s". Bounded by State's eleven entity
# types, so this is a closed list rather than an English-pluralisation problem.
_IRREGULAR_PLURALS = {
    "person": "people",
    "company": "companies",
    "entity": "entities",
    "counterparty": "counterparties",
}


def _plural(n: int, one: str, many: str | None = None) -> str:
    if n == 1:
        return one
    return many or _IRREGULAR_PLURALS.get(one) or one + "s"


# 1. Risk Intelligence ---------------------------------------------------------

def risk_intelligence(reality) -> list[InsightDraft]:
    """One insight per KIND of risk the organization currently carries.

    Keyed on the signal type rather than on the entity, so the row count is
    bounded by the vocabulary (at most six) instead of by how many entities are
    affected — and so the insight is about the organization ("six relationships
    are cooling") rather than a restatement of a single card the brief already
    shows.
    """
    out: list[InsightDraft] = []
    for signal_type in sorted(_RISK_SIGNAL_TYPES):
        signals = _by_type(reality, signal_type)
        if not signals:
            continue  # no evidence of this risk => no claim about it
        noun = signal_type.replace("_", " ")
        n = len(signals)
        out.append(
            InsightDraft(
                insight_type=InsightType.RISK,
                key=signal_type,
                title=f"{n} open {noun} {_plural(n, 'risk')}",
                summary=(
                    f"Reasoning currently identifies {n} "
                    f"{_plural(n, 'entity', 'entities')} with {noun}, "
                    f"peak severity {_peak_severity(signals):.2f}."
                ),
                evidence={
                    "signal_type": signal_type,
                    "affected": n,
                    "peak_severity": _peak_severity(signals),
                    "entities": _entities(signals, reality),
                },
                sample_size=n,
                confidence=_composite_confidence(signals),
                supporting=[],
                engine_name="intelligence.reality.risk",
                engine_version=REALITY_ENGINE_VERSION,
                supporting_signals=_signal_ids(signals),
            )
        )
    return out


# 2. Opportunity Intelligence --------------------------------------------------

def opportunity_intelligence(reality) -> list[InsightDraft]:
    """Warm windows and structural leverage the organization holds right now.

    Both inputs are signals Decision deliberately drops as interpretation, so
    without this engine they were computed on every brief and thrown away.
    """
    out: list[InsightDraft] = []
    for signal_type in sorted(_OPPORTUNITY_SIGNAL_TYPES):
        signals = _by_type(reality, signal_type)
        if not signals:
            continue
        n = len(signals)
        if signal_type == _OPPORTUNITY:
            title = f"{n} warm {_plural(n, 'relationship')}"
            summary = (
                f"{n} {_plural(n, 'counterparty', 'counterparties')} made inbound "
                "contact within the engine's recency window — an open window to "
                "act on."
            )
        else:
            title = f"{n} hub {_plural(n, 'company')}"
            summary = (
                f"{n} {_plural(n, 'company')} "
                f"{'connects' if n == 1 else 'connect'} multiple known people, "
                "concentrating reach through a small number of accounts."
            )
        out.append(
            InsightDraft(
                insight_type=InsightType.OPPORTUNITY,
                key=signal_type,
                title=title,
                summary=summary,
                evidence={
                    "signal_type": signal_type,
                    "count": n,
                    "entities": _entities(signals, reality),
                },
                sample_size=n,
                confidence=_composite_confidence(signals),
                supporting=[],
                engine_name="intelligence.reality.opportunity",
                engine_version=REALITY_ENGINE_VERSION,
                supporting_signals=_signal_ids(signals),
            )
        )
    return out


# 3. Relationship Intelligence -------------------------------------------------

def relationship_intelligence(reality) -> list[InsightDraft]:
    """The shape of the organization's relationship portfolio.

    Needs relationships to exist in State — an org with none gets no insight,
    not a zero. `cooling` and `warm` are counted from the signals Reasoning
    emitted, never re-derived from timestamps here.
    """
    total = int(reality.entity_counts.get("relationship", 0) or 0)
    if total <= 0:
        return []
    cooling = _by_type(reality, _RELATIONSHIP_COOLING)
    warm = _by_type(reality, _OPPORTUNITY)
    contributing = sorted(cooling + warm, key=lambda s: s.signal_id)
    if not contributing:
        # Relationships exist but Reasoning reached no conclusion about any of
        # them. Reporting "0 cooling, 0 warm" would dress an absence of analysis
        # up as a finding.
        return []
    return [
        InsightDraft(
            insight_type=InsightType.RELATIONSHIP,
            key="portfolio",
            title=(
                f"{len(cooling)} of {total} tracked relationships are cooling"
                if cooling
                else f"{len(warm)} of {total} tracked relationships are active"
            ),
            summary=(
                f"Of {total} tracked {_plural(total, 'relationship')}, "
                f"{len(cooling)} {'has' if len(cooling) == 1 else 'have'} gone "
                f"quiet past the cooling threshold and {len(warm)} had recent "
                "inbound contact."
            ),
            evidence={
                "tracked": total,
                "cooling": len(cooling),
                "warm": len(warm),
                "cooling_entities": _entities(cooling, reality),
                "warm_entities": _entities(warm, reality),
            },
            sample_size=total,
            confidence=_composite_confidence(contributing),
            supporting=[],
            engine_name="intelligence.reality.relationship",
            engine_version=REALITY_ENGINE_VERSION,
            supporting_signals=_signal_ids(contributing),
        )
    ]


# 4. Commitment Intelligence ---------------------------------------------------

def commitment_intelligence(reality) -> list[InsightDraft]:
    """Open obligations, and how far past their stated deadline they are.

    DEADLINES ARE NEVER INVENTED. `overdue_days` and `due_at` are read only from
    `commitment_follow_up` evidence, where Reasoning quoted them from the
    message that stated them. An undated promise contributes to the count and
    contributes no deadline, because it has none.
    """
    signals = _by_type(reality, _COMMITMENT_SIGNAL_TYPES)
    if not signals:
        return []
    overdue = [
        s.evidence.get("overdue_days")
        for s in signals
        if isinstance(s.evidence.get("overdue_days"), (int, float))
    ]
    n = len(signals)
    evidence: dict[str, Any] = {
        "open_obligations": n,
        "dated": len(overdue),
        "undated": n - len(overdue),
        "entities": _entities(signals, reality),
    }
    if overdue:
        evidence["max_overdue_days"] = round(max(overdue), 1)
    return [
        InsightDraft(
            insight_type=InsightType.COMMITMENT,
            key="open_obligations",
            title=f"{n} open {_plural(n, 'commitment')} at risk",
            summary=(
                f"{n} {_plural(n, 'commitment')} {'is' if n == 1 else 'are'} open "
                f"past the age threshold"
                + (
                    f"; the furthest past a stated deadline is "
                    f"{max(overdue):.0f} days."
                    if overdue
                    else (
                        " — it carries no stated deadline."
                        if n == 1
                        else " — none of them carries a stated deadline."
                    )
                )
            ),
            evidence=evidence,
            sample_size=n,
            confidence=_composite_confidence(signals),
            supporting=[],
            engine_name="intelligence.reality.commitment",
            engine_version=REALITY_ENGINE_VERSION,
            supporting_signals=_signal_ids(signals),
        )
    ]


# 4b. Obligation Intelligence --------------------------------------------------

def obligation_intelligence(reality) -> list[InsightDraft]:
    """What the organization currently OWES a named counterparty.

    THE BLIND SPOT THIS CLOSES. Of the nine SignalTypes that can become a
    Decision, these two reached Intelligence through no generator at all:
    `waiting_on_user_reply` and `decision_owed`. A brief card built from either
    travelled Signal -> Memory -> State -> Reasoning -> Decision -> Today with
    Intelligence contributing nothing — no insight, no corroboration, an empty
    `supporting_insights` — and they are, between them, two of the three
    categories `execution.proposer` can turn into a real action. The most
    actionable cards on Today were exactly the ones Intelligence was silent
    about.

    A THIRD FAMILY, NOT AN EXTENSION OF RISK. `_RISK_SIGNAL_TYPES` is
    byte-identical to the set `decisions.base` maps to `DecisionCategory.RISK`
    (asserted in tests), so the two layers mean the same thing by "risk" and
    `status_intelligence`'s "N carrying open risk" counts exactly what the brief
    calls risk. Folding obligations in would silently redefine that number for
    every organization and make the two layers disagree. An unanswered person is
    not a decaying relationship; it is a debt with a name on it.

    THE ORGANIZATIONAL CLAIM, not the message. This does not say "an email
    arrived" — that is the Signal, and Memory already holds it. It says how many
    counterparties are currently owed something by this organization, which kind
    of debt each is, and how bad the worst one has become. Keyed per signal type
    so "4 people are waiting on a reply" never hides inside an undifferentiated
    total with "1 decision is owed".

    NOTHING IS RECOMPUTED. Reasoning already decided that a thread is unanswered
    (`engines/waiting.py` reads `ThreadState.direction` — an observed fact, not
    an inference) and that a judgment is owed. This aggregates those conclusions
    with their own confidence, severity and evidence, exactly as the other
    reality engines do.
    """
    out: list[InsightDraft] = []
    for signal_type in sorted(_OBLIGATION_SIGNAL_TYPES):
        signals = _by_type(reality, signal_type)
        if not signals:
            continue  # nothing owed of this kind => no claim about it
        n = len(signals)
        noun = _OBLIGATION_NOUNS[signal_type]
        out.append(
            InsightDraft(
                insight_type=InsightType.OBLIGATION,
                key=signal_type,
                title=f"{n} outstanding {noun} {_plural(n, 'obligation')}",
                summary=(
                    f"{n} {_plural(n, 'counterparty')} "
                    f"{'is' if n == 1 else 'are'} currently owed a {noun} by this "
                    f"organization, peak severity {_peak_severity(signals):.2f}."
                ),
                evidence={
                    "signal_type": signal_type,
                    "outstanding": n,
                    "peak_severity": _peak_severity(signals),
                    "entities": _entities(signals, reality),
                },
                sample_size=n,
                confidence=_composite_confidence(signals),
                supporting=[],
                engine_name="intelligence.reality.obligation",
                engine_version=REALITY_ENGINE_VERSION,
                supporting_signals=_signal_ids(signals),
            )
        )
    return out


# 5. Organizational Status -----------------------------------------------------

def status_intelligence(reality) -> list[InsightDraft]:
    """The organization's current factual footprint and open-risk load.

    THERE IS NO HEALTH SCORE, deliberately. Navis can count what it tracks and
    count what Reasoning flagged; it has no basis for compressing those into
    "83% healthy", and a number like that would be read as a measurement rather
    than the arbitrary weighting it would actually be. This states the counts.
    """
    counts = {k: int(v) for k, v in sorted(reality.entity_counts.items()) if v}
    if not counts:
        return []  # nothing tracked yet => nothing to describe
    risk_signals = _by_type(reality, _RISK_SIGNAL_TYPES)
    load = _by_type(reality, _PRIORITY_LOAD)
    contributing = sorted(risk_signals + load, key=lambda s: s.signal_id)
    headline = ", ".join(f"{v} {_plural(v, k)}" for k, v in sorted(counts.items()))
    evidence: dict[str, Any] = {"counts": counts, "open_risks": len(risk_signals)}
    if load:
        # PriorityEngine already computed where pressure concentrates; carry its
        # breakdown verbatim rather than recounting it a second way.
        evidence["pressure"] = dict(sorted((load[0].evidence or {}).items()))
    return [
        InsightDraft(
            insight_type=InsightType.STATUS,
            key="footprint",
            title=(
                f"Tracking {sum(counts.values())} organizational entities, "
                f"{len(risk_signals)} carrying open risk"
            ),
            summary=f"Currently tracking {headline}.",
            evidence=evidence,
            sample_size=sum(counts.values()),
            # A count of what State holds is observed, not inferred. When
            # Reasoning has flagged risk, the weakest of those detections bounds
            # the claim about how much risk is open.
            confidence=_composite_confidence(contributing) if contributing else 1.0,
            supporting=[],
            engine_name="intelligence.reality.status",
            engine_version=REALITY_ENGINE_VERSION,
            supporting_signals=_signal_ids(contributing),
        )
    ]


# 6. Trend Intelligence --------------------------------------------------------

def trend_intelligence(reality) -> list[InsightDraft]:
    """Direction of travel — emitted ONLY from real temporal evidence.

    A trend needs two snapshots. `TrendEngine` is the only thing in the system
    that has ever seen two, and it emits ORG_DRIFT only when handed a prior
    State. This engine consumes that signal and nothing else: it will not
    difference a single snapshot against a guess, and it produces nothing at all
    until the orchestrator passes `prior_state` to `run_reasoning_v2`.

    That is why it is silent in production today, and being silent is correct.

    KNOWN LIMITATION, stated rather than papered over: this cannot name WHICH
    entity type moved. `TrendEngine` puts that only in its title and dedup_key
    ("Rising commitment volume" / "drift:commitment"), and `RealitySignalRow`
    carries neither — it carries the engine's `evidence`, which holds just
    `delta` and `current`. Naming it would mean adding the key to TrendEngine's
    evidence, which is a Reasoning change and out of scope here. So the insight
    reports the movement it can actually see and claims nothing more.
    """
    signals = _by_type(reality, _ORG_DRIFT)
    if not signals:
        return []
    out: list[InsightDraft] = []
    for s in signals:
        e = s.evidence or {}
        delta, current = e.get("delta"), e.get("current")
        if not isinstance(delta, int) or not isinstance(current, int):
            continue  # a drift signal that cannot say how much it moved says nothing
        direction = "up" if delta > 0 else "down"
        out.append(
            InsightDraft(
                insight_type=InsightType.TREND,
                # Keyed on the signal id: ORG_DRIFT carries no entity, so the
                # signal IS the identity of this particular movement.
                key=s.entity_id or s.signal_id,
                title=f"Tracked volume moved {direction} by {abs(delta)}",
                summary=(
                    f"Reasoning compared this snapshot against the previous one "
                    f"and found a change of {delta:+d}, to a current total of "
                    f"{current}."
                ),
                evidence={"delta": delta, "current": current, "direction": direction},
                sample_size=abs(delta),
                confidence=round(s.confidence, 4),
                supporting=[],
                engine_name="intelligence.reality.trend",
                engine_version=REALITY_ENGINE_VERSION,
                supporting_signals=(s.signal_id,),
            )
        )
    return out


# 7. Situation Intelligence ----------------------------------------------------

# WHAT EACH CONCLUSION IS CALLED WHEN IT IS ONE CLAUSE OF A SENTENCE ABOUT A
# COUNTERPARTY. Closed and keyed by the same literals the sets above use, so a
# signal type cannot join a situation without a decision about how to say it. A
# type absent from this map contributes its SIGNAL to the cluster (it is still
# real pressure on the same counterparty) but never a clause to the headline —
# the alternative is a sentence containing a raw engine identifier.
_SITUATION_CLAUSES = {
    "commitment_slipping":      "an unresolved commitment",
    "commitment_follow_up":     "a commitment past its stated deadline",
    "relationship_cooling":     "a cooling relationship",
    "waiting_on_user_reply":    "an unanswered message",
    "decision_owed":            "a decision they are waiting on",
    "customer_engagement_drop": "falling engagement",
}

# WHICH EDGES A SITUATION MAY TRAVERSE, and the whole reason this engine cannot
# fabricate a connection.
#
# A relationship's own commitments and meetings belong to ONE counterparty, so
# following them can never merge two of them. `people` (company -> person,
# channel -> person) is deliberately ABSENT: a company with twenty contacts
# would otherwise collapse twenty unrelated counterparties into one "situation",
# which is exactly the false join this engine exists to avoid. `documents`,
# `projects` and `channels` are absent for the same reason — they are shared
# workspaces, not counterparty facts.
_SITUATION_FORWARD_ROLES = frozenset({"commitments", "meetings"})

# Edges that point AT the anchor and name the same counterparty from the other
# end: a person names its relationship, a commitment names its owner. Reading a
# persisted directed edge from its target is not an inference — it is the same
# stored fact, read backwards.
_SITUATION_REVERSE_ROLES = frozenset({"relationships", "owner"})

# One counterparty's situation names at most this many linked entities. A
# relationship with fifty recorded meetings is real; enumerating all of them in a
# JSONB blob is not useful. The COUNTS stay exact — only the enumeration is cut.
_MAX_SITUATION_ENTITIES = 12

# HOW EACH TRAVERSED EDGE READS IN A SENTENCE. `via_role` keeps State's own key
# for the trace ("commitments", "relationships_of"); this is the same edge said
# in words, so `organizational_implication` cannot leak an internal token like
# "relationships_of" into a string a human reads. Closed, like every other map
# here: a role with no phrase contributes none.
_ROLE_PHRASES = {
    "commitments":      "a commitment they made",
    "meetings":         "a recorded meeting",
    "relationships_of": "the person this relationship is with",
    "owner_of":         "the person who made the commitment",
    "owner":            "the person who made the commitment",
}


def _situation_label(signals: list, entity_id: str) -> str:
    """The counterparty's name, from the signals' own evidence.

    `_label` reads only keys the reasoning engines actually write
    (`person_email`, `speaker_email`, …). The signals in one cluster are about
    one counterparty by construction, so the first readable label is that
    counterparty's — and the entity id is the honest fallback when no engine in
    the cluster recorded a human-readable field.

    Signals arrive already sorted by signal_id, so the choice is deterministic.
    """
    for s in signals:
        label = _label(s)
        if label and label != s.entity_id and label != s.signal_type:
            return label
    return entity_id


def _clause_list(phrases: list[str]) -> str:
    """"a and b" / "a, b and c" — Oxford-free, deterministic, no trailing comma."""
    if len(phrases) == 1:
        return phrases[0]
    return f"{', '.join(phrases[:-1])} and {phrases[-1]}"


def situation_intelligence(reality) -> list[InsightDraft]:
    """ONE COUNTERPARTY, EVERYTHING REASONING CONCLUDED ABOUT THEM, CONNECTED.

    THIS IS THE ENGINE THAT IS NOT A GROUP BY. Every other reality generator
    buckets signals by TYPE — "2 open commitment slipping risks" — which restates
    Reasoning at a higher altitude and connects nothing. Live, that is why one
    slipping commitment and the cooling relationship OF THE SAME PERSON reached
    the brief as two unrelated cards, while State had recorded
    `relationship --commitments--> commitment` all along.

    This engine reads that edge. The connection is not inferred, scored, modelled
    or guessed: `folds.compute_links` computed it from data the extractors
    recorded (the commitment's `speaker_email` matching the relationship's
    `person_email`), Context carries it verbatim, and this engine follows it.

    THE ANCHOR IS THE RELATIONSHIP, because a relationship IS a counterparty in
    State's vocabulary — it is keyed by one person's email and it is the only
    entity whose links are all about that same person (`_SITUATION_FORWARD_ROLES`).
    Anchoring anywhere else would either miss the connection (a commitment names
    only its owner) or invite a false one (a company names every contact at it).

    IT EMITS ONLY WHEN THERE IS GENUINELY A CONNECTION: at least two distinct
    conclusions, spread across at least two distinct entities. One conclusion
    about one entity is a ReasoningSignal, and repeating it here as a "situation"
    would be the same restatement in a new costume. Silence is the correct output
    for an organization whose pressures do not overlap, and it is the common case.

    NOTHING IS INVENTED. No company, customer, project, role, importance or
    causality appears unless a persisted edge put it there. The headline is
    assembled from a closed clause map over signal types the engines emitted; the
    counts are cardinalities of link lists; the lineage is Context's own
    citations. `organizational_implication` states what the GRAPH says — that
    these conclusions share a counterparty — and never why, which no layer here
    can know.

    Pure and deterministic: sorted inputs, sorted outputs, no clock, no I/O.
    """
    context = tuple(getattr(reality, "context", ()) or ())
    if not context:
        return []

    by_entity: dict[tuple[str | None, str | None], list] = {}
    for s in reality.signals:
        if s.entity_type and s.entity_id:
            by_entity.setdefault((s.entity_type, s.entity_id), []).append(s)
    if not by_entity:
        return []

    rows = {(c.entity_type, c.entity_id): c for c in context}
    # Reverse index: which context entities point AT this one, and by which role.
    # Built once for the pass — Context is capped at a few dozen rows.
    pointing_at: dict[str, list[tuple[str, Any]]] = {}
    for row in context:
        for link in row.links or ():
            if link.role in _SITUATION_REVERSE_ROLES:
                pointing_at.setdefault(link.entity_id, []).append((link.role, row))

    out: list[InsightDraft] = []
    anchors = sorted(
        (c for c in context if c.entity_type == "relationship"),
        key=lambda c: c.entity_id,
    )
    for anchor in anchors:
        # ── Build the cluster from persisted edges only ──────────────────────
        # (entity_type, entity_id) -> the role that put it here, for the trace.
        members: dict[tuple[str, str], str] = {(anchor.entity_type, anchor.entity_id): "anchor"}
        for link in anchor.links or ():
            if link.role in _SITUATION_FORWARD_ROLES:
                members.setdefault((link.entity_type, link.entity_id), link.role)
        # The person who names this relationship, and the owner of any commitment
        # in it — the same stored edges, read from their target end.
        for role, row in pointing_at.get(anchor.entity_id, ()):
            members.setdefault((row.entity_type, row.entity_id), f"{role}_of")
        for (member_type, member_id), _role in list(members.items()):
            if member_type != "commitment":
                continue
            member_row = rows.get((member_type, member_id))
            for link in (member_row.links if member_row else ()) or ():
                if link.role == "owner":
                    members.setdefault((link.entity_type, link.entity_id), "owner")

        # ── The conclusions Reasoning reached about that cluster ─────────────
        cluster_signals = sorted(
            (s for key in members for s in by_entity.get(key, ())),
            key=lambda s: s.signal_id,
        )
        signal_entities = {(s.entity_type, s.entity_id) for s in cluster_signals}
        # A situation is a CONNECTION. Two conclusions about one entity is that
        # entity's own story and every other generator already tells it.
        if len(cluster_signals) < 2 or len(signal_entities) < 2:
            continue

        label = _situation_label(cluster_signals, anchor.entity_id)
        # Clause per DISTINCT signal type, in the order the types were first seen
        # across id-sorted signals — deterministic, and it reads as one sentence.
        seen_types: list[str] = []
        for s in cluster_signals:
            if s.signal_type not in seen_types:
                seen_types.append(s.signal_type)
        clauses = [_SITUATION_CLAUSES[t] for t in seen_types if t in _SITUATION_CLAUSES]
        if not clauses:
            # Every conclusion here is a type this engine has no words for. Say
            # nothing rather than publish a headline containing engine literals.
            continue

        linked_counts: dict[str, int] = {}
        for (member_type, _mid), role in members.items():
            if role != "anchor":
                linked_counts[member_type] = linked_counts.get(member_type, 0) + 1
        linked_phrase = _clause_list(
            [
                f"{n} {_plural(n, t)}"
                for t, n in sorted(linked_counts.items())
            ]
        ) if linked_counts else ""

        related = [
            {
                "entity_type": member_type,
                "entity_id": member_id,
                # WHICH persisted edge put this entity in the situation. The
                # trace answer to "why is this meeting here?".
                "via_role": role,
                "has_signal": (member_type, member_id) in signal_entities,
            }
            for (member_type, member_id), role in sorted(
                members.items(), key=lambda kv: (kv[0][0], kv[0][1])
            )
            if role != "anchor"
        ][:_MAX_SITUATION_ENTITIES]

        # Context's own citations for every entity in the cluster — the memory
        # records these facts were folded from and the business-system events
        # behind them. Pointers, never bodies: Intelligence may not read Memory,
        # and `Citation` is explicit that a ScopedView stays pointer-carrying.
        memory_ids: list[str] = []
        signal_event_ids: list[str] = []
        for key in sorted(members):
            row = rows.get(key)
            if row is None:
                continue
            memory_ids.extend(row.memory_ids or ())
            signal_event_ids.extend(row.signal_event_ids or ())

        out.append(
            InsightDraft(
                insight_type=InsightType.SITUATION,
                # Keyed on the counterparty, so the same situation keeps one
                # identity across snapshots and the id changes only with the
                # version key (watermark + reasoning version).
                key=f"relationship:{anchor.entity_id}",
                title=f"{label} has {_clause_list(clauses)}",
                summary=(
                    f"Reasoning reached {len(cluster_signals)} separate conclusions "
                    f"about this counterparty at this snapshot"
                    + (f"; State links {linked_phrase} to them" if linked_phrase else "")
                    + "."
                ),
                evidence={
                    # ── The grounded representation (see the module docstring on
                    # the ML boundary). Every field below is a persisted fact or a
                    # cardinality of one; nothing is generated prose.
                    "situation_type": "+".join(sorted(seen_types)),
                    "primary_entity": {
                        "entity_type": anchor.entity_type,
                        "entity_id": anchor.entity_id,
                        "label": label,
                        "context_rank": anchor.rank,
                        "context_reason": anchor.reason,
                    },
                    "related_entities": related,
                    "signals": [
                        {
                            "signal_id": s.signal_id,
                            "signal_type": s.signal_type,
                            "entity_type": s.entity_type,
                            "entity_id": s.entity_id,
                            "severity": round(s.severity, 4),
                            "confidence": round(s.confidence, 4),
                        }
                        for s in cluster_signals
                    ],
                    # Same shape every other reality insight publishes, so a
                    # consumer that already reads `entities` keeps working.
                    "entities": _entities(cluster_signals, reality),
                    "linked_entity_counts": linked_counts,
                    "peak_severity": _peak_severity(cluster_signals),
                    # WHAT THE GRAPH SAYS — structural, never causal. It states
                    # that these conclusions share a counterparty, which is what
                    # was persisted; it does not claim one caused the other.
                    "organizational_implication": (
                        f"{len(cluster_signals)} conclusions that Reasoning reached "
                        f"independently are about one counterparty. State connects them "
                        f"through "
                        + (
                            _clause_list(
                                sorted({
                                    _ROLE_PHRASES[r["via_role"]]
                                    for r in related
                                    if r["via_role"] in _ROLE_PHRASES
                                })
                            )
                            or "no recorded link"
                        )
                        + "."
                    ),
                    "memory_ids": sorted(set(memory_ids))[:_MAX_SITUATION_ENTITIES],
                    "signal_event_ids": sorted(set(signal_event_ids))[:_MAX_SITUATION_ENTITIES],
                },
                sample_size=len(cluster_signals),
                confidence=_composite_confidence(cluster_signals),
                supporting=[],
                engine_name="intelligence.reality.situation",
                engine_version=REALITY_ENGINE_VERSION,
                # BOTH signals. This is what makes the situation reach BOTH
                # decisions: `what_matters_today._relevant_facts` matches an
                # insight to a decision when the insight names that decision's
                # signal, so naming both is what lets one organizational sentence
                # corroborate the commitment card and the relationship card alike.
                supporting_signals=_signal_ids(cluster_signals),
            )
        )
    return out


_REALITY_GENERATORS = (
    risk_intelligence,
    opportunity_intelligence,
    relationship_intelligence,
    commitment_intelligence,
    obligation_intelligence,
    status_intelligence,
    trend_intelligence,
    situation_intelligence,
)

# DELIBERATELY ABSENT, and each for a reason rather than an oversight:
#
#   FORECAST  — needs a temporal series and a model. Heuristics dressed as
#               forecasts ("3 more will slip next week") are the single easiest
#               way to make this product untrustworthy. No forecast until there
#               is something to forecast from.
#   BEHAVIOR  — nothing in the pipeline observes founder behaviour. Approval and
#               dismissal events are published to zero consumers, so there is no
#               behavioural evidence to interpret. An empty behaviour model is
#               the honest output, and the frontend already renders one.
#   CHANGE    — the same missing input as TREND. Folded into TREND rather than
#               given a second name for the same evidence.
#   PATTERN   — Learning already owns outcome patterns (`learning_patterns`,
#               surfaced by `team_intelligence`). A second pattern vocabulary
#               over the same evidence would be two answers to one question.
#   CUSTOMER  — `customer_engagement_drop` is already carried by RISK, keyed by
#               its own signal type. Both live orgs currently hold zero
#               CustomerState entities, so a dedicated engine would emit nothing
#               anyway.


def project(view: LearningView, reality=None) -> list[InsightDraft]:
    """Run the Learning generators, and the reality generators when reality is
    supplied. Pure + deterministic given (LearningView, OrganizationalRealityView).

    Reality is skipped without a `state_watermark`: it is the version key every
    reality insight is identified by, so projecting without one would mint rows
    that cannot be told apart from the next snapshot's.
    """
    drafts: list[InsightDraft] = []
    for gen in _GENERATORS:
        drafts.extend(gen(view))
    if reality is not None and reality.state_watermark:
        for reality_gen in _REALITY_GENERATORS:
            drafts.extend(reality_gen(reality))
    return drafts
