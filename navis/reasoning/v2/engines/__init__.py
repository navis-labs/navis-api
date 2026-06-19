"""Engine registry — fixed order makes the reasoning pass deterministic."""
from navis.reasoning.v2.engines.risk import (
    CommitmentReasoningEngine,
    CustomerReasoningEngine,
    DecisionReasoningEngine,
    MeetingReasoningEngine,
    ProjectReasoningEngine,
    RelationshipReasoningEngine,
)
from navis.reasoning.v2.engines.meta import (
    ContextEngine,
    LeverageEngine,
    OpportunityEngine,
    PriorityEngine,
    TrendEngine,
)

ENGINES = [
    CommitmentReasoningEngine(),
    RelationshipReasoningEngine(),
    ProjectReasoningEngine(),
    CustomerReasoningEngine(),
    DecisionReasoningEngine(),
    MeetingReasoningEngine(),
    PriorityEngine(),
    OpportunityEngine(),
    LeverageEngine(),
    TrendEngine(),
    ContextEngine(),
]

__all__ = ["ENGINES"]
