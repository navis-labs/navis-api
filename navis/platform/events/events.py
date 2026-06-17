"""Domain event contract + the canonical event vocabulary.

The whole system is wired as a chain of events. A domain never calls another
domain directly across the pipeline boundary — it emits an event and the next
context reacts. This keeps each bounded context independently evolvable.

    GmailReceived -> CommitmentDetected -> StateUpdated -> ReasoningGenerated
        -> DecisionGenerated -> ActionExecuted -> OutcomeObserved -> LearningUpdated
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(str, Enum):
    # Layer 1 — Signals
    GMAIL_RECEIVED = "GmailReceived"
    CALENDAR_EVENT_CREATED = "CalendarEventCreated"
    MEETING_COMPLETED = "MeetingCompleted"
    SLACK_MESSAGE_RECEIVED = "SlackMessageReceived"
    SIGNAL_INGESTED = "SignalIngested"

    # Layer 2 — Memory
    PERSON_DETECTED = "PersonDetected"
    PROJECT_DETECTED = "ProjectDetected"
    COMMITMENT_DETECTED = "CommitmentDetected"
    DECISION_DETECTED = "DecisionDetected"
    RELATIONSHIP_UPDATED = "RelationshipUpdated"

    # Layer 3 — State
    STATE_UPDATED = "StateUpdated"

    # Layer 4 — Reasoning
    REASONING_GENERATED = "ReasoningGenerated"

    # Layer 5 — Decision
    DECISION_GENERATED = "DecisionGenerated"
    BRIEF_READY = "BriefReady"

    # Layer 6 — Execution
    ACTION_REQUESTED = "ActionRequested"
    ACTION_APPROVED = "ActionApproved"
    ACTION_EXECUTED = "ActionExecuted"

    # Layer 7 — Observation
    OUTCOME_OBSERVED = "OutcomeObserved"

    # Layer 8 — Learning
    LEARNING_UPDATED = "LearningUpdated"


@dataclass(slots=True)
class DomainEvent:
    """Immutable fact: something happened. Carries an org_id for tenancy."""

    type: EventType
    org_id: uuid.UUID | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "type": self.type.value,
            "org_id": str(self.org_id) if self.org_id else None,
            "payload": self.payload,
            "occurred_at": self.occurred_at.isoformat(),
            "source": self.source,
        }
