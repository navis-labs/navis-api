"""Engine registry — fixed order makes the reasoning pass deterministic."""
from navis.reasoning.v2.engines.risk import (
    CommitmentReasoningEngine,
    CustomerReasoningEngine,
    DecisionReasoningEngine,
    MeetingReasoningEngine,
    ProjectReasoningEngine,
    RelationshipReasoningEngine,
)
from navis.reasoning.v2.engines.obligations import (
    CommitmentFollowUpEngine,
    DecisionOwedEngine,
)
from navis.reasoning.v2.engines.waiting import WaitingReplyReasoningEngine
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
    WaitingReplyReasoningEngine(),
    CommitmentFollowUpEngine(),
    DecisionOwedEngine(),
    PriorityEngine(),
    OpportunityEngine(),
    LeverageEngine(),
    TrendEngine(),
    ContextEngine(),
]

__all__ = ["ENGINES"]
