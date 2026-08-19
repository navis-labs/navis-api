"""Learning — Layer 8. Historical outcome attribution.

Answers one question: "what actually works?" — by attributing observed outcomes
to the reasoning signals / decisions that produced them. Append-only, deterministic,
idempotent, replayable, multi-tenant.

Learning is NOT intelligence/reasoning/prediction/recommendation. It stores counts
and patterns (facts), never scores or actions.

    OUTCOME_OBSERVED -> learning_event -> count aggregates -> patterns
                                                                 |
                                       navis.learning.metrics ---+--> consumers

OUTCOME_OBSERVED is the ONLY input. Learning never queries Execution, Observation,
a provider or a connector to reconstruct an outcome — everything it records comes
off that one event, which is what makes replaying it reproduce the same result.

`navis.learning.metrics` is the ONLY output. Consumers read the contract there,
not these tables.
"""
from navis.learning.metrics import (
    AggregateFact,
    LearningSnapshot,
    SignalPattern,
    learning_snapshot,
)
from navis.learning.models import (
    AggregateDimension,
    LearningAggregate,
    LearningEvent,
    LearningResult,
    OutcomeSurface,
)
from navis.learning.service import LearningService

__all__ = [
    "AggregateDimension",
    "AggregateFact",
    "LearningAggregate",
    "LearningEvent",
    "LearningResult",
    "LearningService",
    "LearningSnapshot",
    "OutcomeSurface",
    "SignalPattern",
    "learning_snapshot",
]
