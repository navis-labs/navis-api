"""Organization public contract: the stable surface business domains depend on.

Business domains (Signals…Adaptive) import THIS to ask "may this org use X?" — they
never import Organization internals. Organization sits below them (above Identity)
and imports only platform + identity, so the dependency arrow always points down.
"""
from __future__ import annotations

import enum


class Plan(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    TEAM = "team"
    ENTERPRISE = "enterprise"


class LifecycleStatus(str, enum.Enum):
    TRIAL = "trial"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CHURNED = "churned"


class Capability(str, enum.Enum):
    # Signal sources
    GMAIL = "gmail"
    CALENDAR = "calendar"
    DRIVE = "drive"
    SLACK = "slack"
    # Microsoft 365 sources. Separate capabilities from their Google
    # counterparts rather than a shared `mail` / `calendar` cap, because
    # entitlement is a commercial decision made per connector: an org can be
    # licensed for Gmail and not for Outlook. Feature gates that mean "any mail
    # source" resolve the lane on the frontend and then check that connector's
    # cap, so nothing here has to know about lanes.
    #
    # These are what the Connections page filters cards on. A capability that
    # is true is a promise that the founder can connect the thing NOW, which is
    # why TEAMS is granted by no plan below — the member exists so the wire
    # format is complete, not because there is anything behind it yet.
    OUTLOOK = "outlook"
    OUTLOOK_CALENDAR = "outlook_calendar"
    ONEDRIVE = "onedrive"
    # No TEAMS member. Reading Teams channel messages needs Microsoft's
    # Protected APIs approval, so there is no connector behind it — and a
    # granted source capability renders a Connect card, which would leave a
    # founder with a card that can never reach a connected state.
    #
    # It belongs here in the same change that lands the connector. Adding it
    # early would also force an exception into PLAN_CAPABILITIES' invariant
    # that ENTERPRISE grants every capability, which is worth preserving.
    # Pipeline
    MEMORY = "memory"
    STATE = "state"
    REASONING = "reasoning"
    DECISION = "decision"
    EXECUTION = "execution"
    OBSERVATION = "observation"
    LEARNING = "learning"
    INTELLIGENCE = "intelligence"
    ADAPTIVE = "adaptive"
    ML = "ml"


class OnboardingStage(str, enum.Enum):
    CREATED = "created"
    PROFILE_COMPLETE = "profile_complete"
    GMAIL_CONNECTED = "gmail_connected"
    CALENDAR_CONNECTED = "calendar_connected"
    FIRST_SIGNAL = "first_signal"
    MEMORY_READY = "memory_ready"
    STATE_READY = "state_ready"
    FIRST_BRIEF = "first_brief"
    ACTIVATED = "activated"


# Canonical forward order — onboarding only ever advances along this path.
STAGE_ORDER: tuple[OnboardingStage, ...] = (
    OnboardingStage.CREATED,
    OnboardingStage.PROFILE_COMPLETE,
    OnboardingStage.GMAIL_CONNECTED,
    OnboardingStage.CALENDAR_CONNECTED,
    OnboardingStage.FIRST_SIGNAL,
    OnboardingStage.MEMORY_READY,
    OnboardingStage.STATE_READY,
    OnboardingStage.FIRST_BRIEF,
    OnboardingStage.ACTIVATED,
)


def stage_index(stage: OnboardingStage) -> int:
    return STAGE_ORDER.index(stage)


class OnboardingEvent(str, enum.Enum):
    ORG_CREATED = "org.created"
    PROFILE_COMPLETED = "org.profile_completed"
    GMAIL_CONNECTED = "org.gmail_connected"
    CALENDAR_CONNECTED = "org.calendar_connected"
    FIRST_SIGNAL = "org.first_signal"
    MEMORY_READY = "org.memory_ready"
    STATE_READY = "org.state_ready"
    FIRST_BRIEF = "org.first_brief"
    ORG_ACTIVATED = "org.activated"


# Stage -> the onboarding event emitted when it is reached.
STAGE_EVENT: dict[OnboardingStage, str] = {
    OnboardingStage.CREATED: OnboardingEvent.ORG_CREATED,
    OnboardingStage.PROFILE_COMPLETE: OnboardingEvent.PROFILE_COMPLETED,
    OnboardingStage.GMAIL_CONNECTED: OnboardingEvent.GMAIL_CONNECTED,
    OnboardingStage.CALENDAR_CONNECTED: OnboardingEvent.CALENDAR_CONNECTED,
    OnboardingStage.FIRST_SIGNAL: OnboardingEvent.FIRST_SIGNAL,
    OnboardingStage.MEMORY_READY: OnboardingEvent.MEMORY_READY,
    OnboardingStage.STATE_READY: OnboardingEvent.STATE_READY,
    OnboardingStage.FIRST_BRIEF: OnboardingEvent.FIRST_BRIEF,
    OnboardingStage.ACTIVATED: OnboardingEvent.ORG_ACTIVATED,
}

# Plan -> the capabilities it grants by default (entitlements seed from this).
#
# The Microsoft sources mirror their Google counterparts exactly — outlook
# wherever gmail appears, outlook_calendar wherever calendar does, onedrive
# wherever drive does. Same product, different vendor; pricing the two
# differently is a decision nobody has made, and making it implicitly here
# would strand every Microsoft org on a plan that reads their mail but not
# their calendar.
PLAN_CAPABILITIES: dict[Plan, frozenset[Capability]] = {
    Plan.FREE: frozenset({
        Capability.GMAIL, Capability.CALENDAR, Capability.MEMORY, Capability.STATE,
        Capability.OUTLOOK, Capability.OUTLOOK_CALENDAR,
    }),
    Plan.PRO: frozenset({
        Capability.GMAIL, Capability.CALENDAR, Capability.DRIVE, Capability.MEMORY,
        Capability.STATE, Capability.REASONING, Capability.DECISION,
        Capability.OUTLOOK, Capability.OUTLOOK_CALENDAR, Capability.ONEDRIVE,
    }),
    Plan.TEAM: frozenset({
        Capability.GMAIL, Capability.CALENDAR, Capability.DRIVE, Capability.SLACK,
        Capability.MEMORY, Capability.STATE, Capability.REASONING, Capability.DECISION,
        Capability.EXECUTION, Capability.INTELLIGENCE,
        Capability.OUTLOOK, Capability.OUTLOOK_CALENDAR, Capability.ONEDRIVE,
    }),
    Plan.ENTERPRISE: frozenset(Capability),
}


def capabilities_for_plan(plan: Plan) -> frozenset[Capability]:
    return PLAN_CAPABILITIES.get(plan, frozenset())
