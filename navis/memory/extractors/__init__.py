"""Extractor registry.

Registration order is fixed here, which makes `run_extractors` output order
deterministic. CustomerMemory has no extractor yet: "customer" is a relationship
classification that must come from a CRM signal source, not be inferred from email
(inference would be intelligence, which Memory forbids). The CUSTOMER entity type
and payload schema exist; population awaits a CRM connector.
"""
from navis.memory.extractors.base import (
    Extractor,
    MemoryDraft,
    SignalView,
    extractors_for,
    register,
    run_extractors,
)
from navis.memory.extractors.email import (
    CommitmentExtractor,
    CompanyExtractor,
    DecisionExtractor,
    PersonExtractor,
    RelationshipExtractor,
    ThreadExtractor,
)
from navis.memory.extractors.calendar import CalendarAttendeeExtractor, MeetingExtractor
from navis.memory.extractors.drive import DriveProjectExtractor
from navis.memory.extractors.docs import DocumentExtractor
from navis.memory.extractors.meet import MeetMeetingExtractor
from navis.memory.extractors.slack import (
    SlackChannelExtractor,
    SlackCommitmentExtractor,
    SlackDecisionExtractor,
    SlackPersonExtractor,
    SlackRelationshipExtractor,
    SlackThreadExtractor,
)

# Order matters for deterministic output.
for _ext in (
    PersonExtractor(),
    CompanyExtractor(),
    ThreadExtractor(),
    RelationshipExtractor(),
    CommitmentExtractor(),
    DecisionExtractor(),
    MeetingExtractor(),
    CalendarAttendeeExtractor(),
    DriveProjectExtractor(),
    DocumentExtractor(),
    MeetMeetingExtractor(),
    SlackPersonExtractor(),
    SlackChannelExtractor(),
    SlackThreadExtractor(),
    SlackRelationshipExtractor(),
    SlackCommitmentExtractor(),
    SlackDecisionExtractor(),
):
    register(_ext)

__all__ = [
    "Extractor",
    "MemoryDraft",
    "SignalView",
    "run_extractors",
    "extractors_for",
    "register",
]
