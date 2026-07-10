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


def project(view: LearningView) -> list[InsightDraft]:
    """Run all six generators. Pure + deterministic given the LearningView."""
    drafts: list[InsightDraft] = []
    for gen in _GENERATORS:
        drafts.extend(gen(view))
    return drafts
