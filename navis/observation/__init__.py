"""Observation — Layer 7. The append-only outcome system of record.

"What happened after execution?" — nothing else. Observation consumes Execution
events + provider webhooks, stores immutable, fully-provenanced outcome facts
(no success/confidence/prediction — those are Learning's), and emits
OUTCOME_OBSERVED with structured lineage so Learning attributes effects to the
reasoning signals that caused them.

It NEVER reads Memory/State/Reasoning/Decision/Learning/Intelligence — only
Execution events and webhooks.
"""
from navis.observation.models import Outcome, OutcomeType
from navis.observation.service import ObservationService

__all__ = ["Outcome", "OutcomeType", "ObservationService"]
