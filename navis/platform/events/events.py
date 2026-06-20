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
    ACTION_REJECTED = "ActionRejected"
    ACTION_EXECUTING = "ActionExecuting"
    ACTION_EXECUTED = "ActionExecuted"
    ACTION_FAILED = "ActionFailed"
    ACTION_CANCELLED = "ActionCancelled"

    # Layer 7 — Observation
    OUTCOME_OBSERVED = "OutcomeObserved"

    # Layer 8 — Learning
    LEARNING_UPDATED = "LearningUpdated"

    # Organization lifecycle / onboarding (owned by Organization V1)
    ORG_CREATED = "OrgCreated"
    ORG_PROFILE_COMPLETED = "OrgProfileCompleted"
    ORG_GMAIL_CONNECTED = "OrgGmailConnected"
    ORG_CALENDAR_CONNECTED = "OrgCalendarConnected"
    ORG_FIRST_SIGNAL = "OrgFirstSignal"
    ORG_MEMORY_READY = "OrgMemoryReady"
    ORG_STATE_READY = "OrgStateReady"
    ORG_FIRST_BRIEF = "OrgFirstBrief"
    ORG_ACTIVATED = "OrgActivated"

    # Identity administration (delegated-admin mutations; consumed by security
    # analytics / learning / intelligence — none of which Identity imports).
    MEMBER_INVITED = "MemberInvited"
    MEMBER_REMOVED = "MemberRemoved"
    ROLE_ASSIGNED = "RoleAssigned"
    API_KEY_CREATED = "ApiKeyCreated"
    API_KEY_REVOKED = "ApiKeyRevoked"
    SERVICE_ACCOUNT_CREATED = "ServiceAccountCreated"
    SERVICE_ACCOUNT_ROTATED = "ServiceAccountRotated"
    SERVICE_ACCOUNT_REVOKED = "ServiceAccountRevoked"
    SSO_UPDATED = "SsoUpdated"
    AUTHORIZATION_DENIED = "AuthorizationDenied"


EVENT_SCHEMA_VERSION = 1


@dataclass(slots=True)
class DomainEvent:
    """Immutable fact: something happened. Carries an org_id for tenancy.

    Lineage fields (added in Platform V2, all backward-compatible):
      * correlation_id — ties every event in one logical flow together (auto-filled
        from the active request/task id), so a whole pipeline run is traceable.
      * causation_id   — the id of the event that DIRECTLY caused this one, giving a
        precise parent→child chain for replay and debugging.
      * schema_version — the payload contract version, so consumers can evolve
        payloads without silent schema drift.
    """

    type: EventType
    org_id: uuid.UUID | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    schema_version: int = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Default the correlation id to the ambient request/task id so events emitted
        # while handling a request inherit its trace without callers doing anything.
        if self.correlation_id is None:
            from navis.platform.monitoring.logging import get_request_id

            rid = get_request_id()
            self.correlation_id = rid if rid != "-" else str(self.id)

    def caused(self, type: EventType, **kwargs: Any) -> "DomainEvent":
        """Build a child event that records THIS event as its cause (lineage chain)."""
        kwargs.setdefault("org_id", self.org_id)
        kwargs.setdefault("correlation_id", self.correlation_id)
        return DomainEvent(type=type, causation_id=str(self.id), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "type": self.type.value,
            "org_id": str(self.org_id) if self.org_id else None,
            "payload": self.payload,
            "occurred_at": self.occurred_at.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "schema_version": self.schema_version,
        }
