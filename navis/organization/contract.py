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
# ── WHY REASONING AND OBSERVATION ARE IN EVERY TIER (P0-4) ──────────────────
# This matrix was shaped for the APPLICATION, where a tier buys connectors and
# richer inference. The DEVELOPER surface reads the same capabilities, and on
# FREE — which is what every self-serve signup gets — that left 7 of its 14
# route groups answering 402, including the ENTIRE agent execution ledger,
# because `observation` was ENTERPRISE-only.
#
# That is not a pricing position, it is an unusable product: an agent platform
# whose free tier cannot record what an agent did, and cannot return the
# organization's own conclusions, has nothing for a developer to evaluate. The
# two capabilities added here are the ones the Developer surface cannot do
# without:
#
#   observation  the agent run ledger — tasks, runs, steps, outcomes. An agent
#                WRITES this about itself, so it is the cheapest thing in the
#                product to serve and the first thing a developer needs. Still
#                gated a second time on the credential's `record_agent_runs`
#                permission, which a founder grants deliberately; the plan
#                decides whether the tenant may, the key decides whether this
#                caller may.
#   reasoning    the organization's own conclusions. Without it
#                `POST /developer/context` withholds the reasoning layer and an
#                agent gets evidence with no judgement attached.
#
# DELIBERATELY STILL PAID: decision, intelligence, learning, execution — the
# layers that act, or that are expensive to compute. A FREE developer org reads
# what the organization knows and concluded, and records what its agents did.
#
# APPLICATION IMPACT IS NIL, verified rather than assumed. `Capability.REASONING`
# is referenced only by Developer routes. `Capability.OBSERVATION` adds one
# Application route (`observation/router.py`) to a FREE tenant's reach, which
# WIDENS access and cannot break a caller. `/chat` gates on `MEMORY` alone and
# is untouched.
#
# TIERS STAY STRICTLY NESTED (`test_plan_capability_monotonicity`), so
# `observation` is added to PRO and TEAM as well: a paid tier may never hold
# less than a free one.
PLAN_CAPABILITIES: dict[Plan, frozenset[Capability]] = {
    Plan.FREE: frozenset({
        Capability.GMAIL, Capability.CALENDAR, Capability.MEMORY, Capability.STATE,
        Capability.REASONING, Capability.OBSERVATION,
    }),
    Plan.PRO: frozenset({
        Capability.GMAIL, Capability.CALENDAR, Capability.DRIVE, Capability.MEMORY,
        Capability.STATE, Capability.REASONING, Capability.DECISION,
        Capability.OBSERVATION,
    }),
    Plan.TEAM: frozenset({
        Capability.GMAIL, Capability.CALENDAR, Capability.DRIVE, Capability.SLACK,
        Capability.MEMORY, Capability.STATE, Capability.REASONING, Capability.DECISION,
        Capability.EXECUTION, Capability.INTELLIGENCE, Capability.OBSERVATION,
    }),
    Plan.ENTERPRISE: frozenset(Capability),
}


def capabilities_for_plan(plan: Plan) -> frozenset[Capability]:
    return PLAN_CAPABILITIES.get(plan, frozenset())
