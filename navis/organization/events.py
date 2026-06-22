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
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("organization.events")

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


async def publish_plan_changed(
    *, org_id: uuid.UUID, old_plan: str, new_plan: str, actor_id, changed_at: str
) -> None:
    """Announce a COMMITTED plan change on the bus (ORG-003). The payload is
    self-contained (old + new plan), so consumers react without a DB read. Fully
    decoupled: any domain subscribes via `@subscribe(EventType.ORG_PLAN_CHANGED)` without
    touching BillingService."""
    await event_bus.publish(
        DomainEvent(
            type=EventType.ORG_PLAN_CHANGED,
            org_id=org_id,
            payload={
                "organization_id": str(org_id),
                "old_plan": old_plan,
                "new_plan": new_plan,
                "actor_id": str(actor_id) if actor_id else None,
                "changed_at": changed_at,
            },
            source="organization",
        )
    )


async def bus_publisher(event_name: str, payload: dict) -> None:
    """OnboardingService publisher → platform bus.

    An unmapped event name would otherwise vanish silently and break the funnel, so it
    is logged + metered. A parity test guards completeness, so this is a misconfiguration
    alarm (must never fire in production), not a normal code path.
    """
    event_type = _EVENT_TYPE.get(event_name)
    if event_type is None:
        logger.error("unmapped onboarding event %r dropped — add it to _EVENT_TYPE", event_name)
        metrics.incr("organization.events.unmapped", labels={"event": str(event_name)})
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
