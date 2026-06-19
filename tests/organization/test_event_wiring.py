"""Organization↔platform-bus wiring: event mapping completeness + subscriber registration."""
from __future__ import annotations

from navis.organization.contract import STAGE_EVENT, OnboardingEvent, OnboardingStage
from navis.organization.events import _EVENT_TYPE
from navis.platform.events.events import EventType


def test_every_onboarding_event_maps_to_a_platform_event_type():
    onboarding_events = {
        v for k, v in vars(OnboardingEvent).items()
        if not k.startswith("_") and isinstance(v, str)
    }
    assert onboarding_events == set(_EVENT_TYPE)  # no event left un-bridged
    assert all(isinstance(et, EventType) for et in _EVENT_TYPE.values())


def test_every_stage_has_an_event():
    assert set(STAGE_EVENT) == set(OnboardingStage)


def test_provisioning_and_funnel_subscribers_registered():
    import navis.registry  # noqa: F401  registers subscribers
    from navis.platform.events import event_bus

    def handlers(et):
        return [h.__name__ for h in event_bus._handlers.get(et, [])]

    assert "provision_organization" in handlers(EventType.ORG_CREATED)
    assert "on_state_ready" in handlers(EventType.STATE_UPDATED)
    assert "on_first_brief" in handlers(EventType.BRIEF_READY)
    assert "on_first_signal" in handlers(EventType.SIGNAL_INGESTED)
    assert "on_memory_ready" in handlers(EventType.COMMITMENT_DETECTED)
