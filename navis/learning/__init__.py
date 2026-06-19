"""Learning — Layer 8. Historical outcome attribution.

Answers one question: "what actually works?" — by attributing observed outcomes
to the reasoning signals / decisions that produced them. Append-only, deterministic,
idempotent, replayable, multi-tenant.

Learning is NOT intelligence/reasoning/prediction/recommendation. It stores counts
and patterns (facts), never scores or actions.

    OUTCOME_OBSERVED -> learning_event -> count aggregates -> patterns
"""
from navis.learning.models import (
    AggregateDimension,
    LearningAggregate,
    LearningEvent,
    LearningResult,
)
from navis.learning.service import LearningService

__all__ = [
    "AggregateDimension",
    "LearningAggregate",
    "LearningEvent",
    "LearningResult",
    "LearningService",
]
