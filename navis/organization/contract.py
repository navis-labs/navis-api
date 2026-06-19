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
    # Pipeline
    MEMORY = "memory"
    STATE = "state"
    REASONING = "reasoning"
    DECISION = "decision"
    EXECUTION = "execution"
    INTELLIGENCE = "intelligence"
    ADAPTIVE = "adaptive"


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


class OnboardingEvent:
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
PLAN_CAPABILITIES: dict[Plan, frozenset[Capability]] = {
    Plan.FREE: frozenset({
        Capability.GMAIL, Capability.CALENDAR, Capability.MEMORY, Capability.STATE,
    }),
    Plan.PRO: frozenset({
        Capability.GMAIL, Capability.CALENDAR, Capability.DRIVE, Capability.MEMORY,
        Capability.STATE, Capability.REASONING, Capability.DECISION,
    }),
    Plan.TEAM: frozenset({
        Capability.GMAIL, Capability.CALENDAR, Capability.DRIVE, Capability.SLACK,
        Capability.MEMORY, Capability.STATE, Capability.REASONING, Capability.DECISION,
        Capability.EXECUTION, Capability.INTELLIGENCE,
    }),
    Plan.ENTERPRISE: frozenset(Capability),
}


def capabilities_for_plan(plan: Plan) -> frozenset[Capability]:
    return PLAN_CAPABILITIES.get(plan, frozenset())
