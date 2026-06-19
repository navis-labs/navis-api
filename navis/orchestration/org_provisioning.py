"""Identity ↔ Organization integration (event-driven; no import inversion).

Identity publishes ORG_CREATED at signup. Organization is provisioned HERE, in the
orchestration layer — Identity never imports Organization. The same module advances
the OrganizationOnboarding funnel from real pipeline events, so onboarding state is
driven by reality, not a parallel procedural script.

All handlers are idempotent: bootstrap uses ensure_*; onboarding.advance is monotonic.
"""
from __future__ import annotations

from navis.organization.contract import OnboardingStage
from navis.organization.service import OrganizationService
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, subscribe
from navis.platform.monitoring import get_logger

logger = get_logger("orchestration.org_provisioning")


@subscribe(EventType.ORG_CREATED)
async def provision_organization(event: DomainEvent) -> None:
    """Bootstrap lifecycle(TRIAL) + onboarding(CREATED) + default entitlements/flags."""
    if event.org_id is None or event.source == "organization":
        return  # ignore organization's own re-emissions (loop guard)
    async with session_scope() as db:
        await OrganizationService(db).bootstrap(event.org_id)
    logger.info("organization bootstrapped org=%s", event.org_id)


async def _advance(org_id, stage: OnboardingStage) -> None:
    if org_id is None:
        return
    async with session_scope() as db:
        await OrganizationService(db).onboarding.advance(org_id, stage)


@subscribe(EventType.GMAIL_RECEIVED)
@subscribe(EventType.CALENDAR_EVENT_CREATED)
@subscribe(EventType.SIGNAL_INGESTED)
async def on_first_signal(event: DomainEvent) -> None:
    await _advance(event.org_id, OnboardingStage.FIRST_SIGNAL)


@subscribe(EventType.COMMITMENT_DETECTED)
@subscribe(EventType.PERSON_DETECTED)
async def on_memory_ready(event: DomainEvent) -> None:
    await _advance(event.org_id, OnboardingStage.MEMORY_READY)


@subscribe(EventType.STATE_UPDATED)
async def on_state_ready(event: DomainEvent) -> None:
    await _advance(event.org_id, OnboardingStage.STATE_READY)


@subscribe(EventType.BRIEF_READY)
async def on_first_brief(event: DomainEvent) -> None:
    # Reaching first brief advances to FIRST_BRIEF, which flips lifecycle to ACTIVE.
    await _advance(event.org_id, OnboardingStage.FIRST_BRIEF)
