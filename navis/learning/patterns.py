"""Pattern derivation — deterministic facts over signal aggregates (Intelligence
fuel). Stores facts ("this signal kind is reliable: 87% over 120 samples"), never
recommendations. Pure function of the aggregates, so it rebuilds identically.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from navis.learning.aggregates import success_rate
from navis.learning.models import LearningAggregate, pattern_id

MIN_SAMPLES = 5
RELIABLE_THRESHOLD = 0.7


@dataclass(frozen=True, slots=True)
class PatternFact:
    pattern_id: uuid.UUID
    org_id: uuid.UUID
    signal_id: str
    kind: str  # reliable | unreliable
    samples: int
    success_rate: float


def derive_signal_patterns(
    org_id: uuid.UUID, signal_aggregates: Sequence[LearningAggregate]
) -> list[PatternFact]:
    """Emit a pattern fact per SIGNAL aggregate with enough decisive samples
    (successes + failures >= MIN_SAMPLES). Pure + deterministic; key == signal_id."""
    facts: list[PatternFact] = []
    for agg in signal_aggregates:
        decisive = agg.successes + agg.failures
        if decisive < MIN_SAMPLES:
            continue
        rate = success_rate(agg.successes, agg.failures) or 0.0
        kind = "reliable" if rate >= RELIABLE_THRESHOLD else "unreliable"
        facts.append(
            PatternFact(
                pattern_id=pattern_id(org_id, agg.key),
                org_id=org_id,
                signal_id=agg.key,
                kind=kind,
                samples=decisive,
                success_rate=rate,
            )
        )
    return facts
