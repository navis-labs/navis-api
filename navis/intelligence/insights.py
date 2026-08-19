"""The six intelligence derivations — pure functions over a LearningView.

Each emits FACTS (what the org has learned about itself), never recommendations.
Deterministic: sorted iteration, no clock/random/uuid4/LLM. Input is Learning only.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from navis.intelligence.models import InsightType

MIN_SAMPLES = 5
# Outcome types that represent customer engagement / positive results.
_CUSTOMER_POSITIVE = {"reply_received", "meeting_accepted", "meeting_completed"}
_NEGATIVE = {"action_failed", "no_response"}
_ATTEMPT = "action_executed"


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


def _outcome_total(view: LearningView, keys) -> int:
    return sum(a.total for a in view.outcome_aggs if a.key in keys)


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
    if not view.patterns:
        return []
    reliable = [p for p in view.patterns if p.kind == "reliable"]
    samples = sum(p.samples for p in view.patterns)
    return [
        InsightDraft(
            insight_type=InsightType.TEAM,
            key="signal_reliability",
            title=f"Signal judgment reliable in {len(reliable)}/{len(view.patterns)} patterns",
            summary=(
                f"The team's reasoning signals proved reliable in {len(reliable)} of "
                f"{len(view.patterns)} recurring patterns."
            ),
            evidence={"reliable": len(reliable), "total_patterns": len(view.patterns)},
            sample_size=samples,
            confidence=_confidence(len(view.patterns) * 10),
            supporting=[],
        )
    ]


# 4. Customer Intelligence (from outcome aggregates) ---------------------------

def customer_intelligence(view: LearningView) -> list[InsightDraft]:
    engaged = _outcome_total(view, _CUSTOMER_POSITIVE)
    attempts = _outcome_total(view, {_ATTEMPT})
    if attempts == 0:
        return []
    rate = round(min(1.0, engaged / attempts), 4)
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
    positive = _outcome_total(view, _CUSTOMER_POSITIVE)
    negative = _outcome_total(view, _NEGATIVE)
    decisive = positive + negative
    if decisive == 0:
        return []
    rate = round(positive / decisive, 4)
    return [
        InsightDraft(
            insight_type=InsightType.ORGANIZATIONAL,
            key="outcome_profile",
            title=f"{rate:.0%} of decisive outcomes were positive",
            summary=(
                f"Across {decisive} decisive outcomes, {rate:.0%} were positive; "
                f"{len(view.patterns)} recurring signal patterns have emerged."
            ),
            evidence={"positive": positive, "negative": negative, "patterns": len(view.patterns)},
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
    InsightType.STATUS,
    InsightType.TREND,
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


def _entities(signals: list) -> list[dict[str, Any]]:
    """The affected entities, named and bounded, sorted for determinism."""
    return [
        {
            "entity_type": s.entity_type,
            "entity_id": s.entity_id,
            "label": _label(s),
            "severity": round(s.severity, 4),
            "signal_id": s.signal_id,
        }
        for s in signals[:_MAX_NAMED_ENTITIES]
    ]


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
                    "entities": _entities(signals),
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
                    "entities": _entities(signals),
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
                "cooling_entities": _entities(cooling),
                "warm_entities": _entities(warm),
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
        "entities": _entities(signals),
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


_REALITY_GENERATORS = (
    risk_intelligence,
    opportunity_intelligence,
    relationship_intelligence,
    commitment_intelligence,
    status_intelligence,
    trend_intelligence,
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
