"""Engine registry. Adding an engine here makes it part of every reasoning pass."""
from navis.reasoning.engines.commitment_risk import CommitmentRiskEngine
from navis.reasoning.engines.relationship_risk import RelationshipRiskEngine
from navis.reasoning.engines.project_risk import ProjectRiskEngine
from navis.reasoning.engines.priority import PriorityEngine

# Order is irrelevant — the decision layer ranks by weight.
ENGINES = [
    CommitmentRiskEngine(),
    RelationshipRiskEngine(),
    ProjectRiskEngine(),
    PriorityEngine(),
]

__all__ = [
    "ENGINES",
    "CommitmentRiskEngine",
    "RelationshipRiskEngine",
    "ProjectRiskEngine",
    "PriorityEngine",
]
