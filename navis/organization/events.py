"""Bridge onboarding transitions onto the platform event bus.

OnboardingService emits string-keyed events (contract.OnboardingEvent); this adapter
maps each to the platform EventType vocabulary and publishes a DomainEvent so any
domain can subscribe. Replay-safe: onboarding advance is idempotent (monotonic), so
re-emitting an event for an already-reached stage never happens.
"""
from __future__ import annotations

import uuid

from navis.organization.contract import OnboardingEvent
from navis.platform.events import DomainEvent, EventType, event_bus

_EVENT_TYPE: dict[str, EventType] = {
    OnboardingEvent.ORG_CREATED: EventType.ORG_CREATED,
    OnboardingEvent.PROFILE_COMPLETED: EventType.ORG_PROFILE_COMPLETED,
    OnboardingEvent.GMAIL_CONNECTED: EventType.ORG_GMAIL_CONNECTED,
    OnboardingEvent.CALENDAR_CONNECTED: EventType.ORG_CALENDAR_CONNECTED,
    OnboardingEvent.FIRST_SIGNAL: EventType.ORG_FIRST_SIGNAL,
    OnboardingEvent.MEMORY_READY: EventType.ORG_MEMORY_READY,
    OnboardingEvent.STATE_READY: EventType.ORG_STATE_READY,
    OnboardingEvent.FIRST_BRIEF: EventType.ORG_FIRST_BRIEF,
    OnboardingEvent.ORG_ACTIVATED: EventType.ORG_ACTIVATED,
}


async def bus_publisher(event_name: str, payload: dict) -> None:
    """OnboardingService publisher → platform bus. Unknown names are no-ops."""
    event_type = _EVENT_TYPE.get(event_name)
    if event_type is None:
        return
    org = payload.get("organization_id")
    await event_bus.publish(
        DomainEvent(
            type=event_type,
            org_id=uuid.UUID(org) if org else None,
            payload=payload,
            source="organization",
        )
    )
