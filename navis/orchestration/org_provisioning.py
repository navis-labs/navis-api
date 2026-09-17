"""Identity ↔ Organization integration (event-driven; no import inversion).

Identity publishes ORG_CREATED at signup. Organization is provisioned HERE, in the
orchestration layer — Identity never imports Organization. The same module advances
the OrganizationOnboarding funnel from real pipeline events, so onboarding state is
driven by reality, not a parallel procedural script.

All handlers are idempotent: bootstrap uses ensure_*; onboarding.advance is monotonic.

This module also carries the OPERATOR entry point for lifecycle transitions
(`set_organization_lifecycle`) — see its docstring for why it is a task and not a
route.
"""
from __future__ import annotations

import uuid

from navis.organization.contract import LifecycleStatus, OnboardingStage
from navis.organization.lifecycle import LifecycleError
from navis.organization.service import OrganizationService
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, subscribe
from navis.platform.monitoring import get_logger
from navis.platform.queue import celery_app, run_async

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
@subscribe(EventType.CALENDAR_EVENT_UPDATED)
@subscribe(EventType.CALENDAR_EVENT_CANCELLED)
@subscribe(EventType.SIGNAL_INGESTED)
async def on_first_signal(event: DomainEvent) -> None:
    # Any first signal advances the funnel — now that calendar events route to distinct
    # types (event-type routing fix), subscribe to all of them so the milestone still fires.
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


# ── Operator lifecycle control (T-3) ────────────────────────────────────────
#
# WHY THIS EXISTS. `LifecycleService.suspend()` and `.churn()` were written,
# tested, and reachable from nothing: the only production caller of the whole
# service was `onboarding.advance` -> `activate`. So no organization could ever
# leave TRIAL/ACTIVE, which made the FIRST and most authoritative gate in
# `EntitlementService._decide` — the lifecycle check — dead code, and with it the
# execution worker's "the org is suspended for non-payment" refusal.
#
# WHY A TASK AND NOT A ROUTE. Suspension is an operator action against a tenant,
# and every HTTP principal in this system is scoped to exactly one organization by
# `get_principal`. There is no cross-tenant principal, and inventing one would be
# a second authorization system for a single lever — precisely what this task
# forbids. Operator-invoked, deliberately-unscheduled Celery tasks are the
# established pattern here for exactly this shape of action (see
# `signals.token_backfill.reencrypt_legacy_tokens`).
#
# WHY NO SELF-SERVICE. A founder must not be able to reactivate their own
# suspended organization — that would make suspension advisory. Self-service
# ACCOUNT CLOSURE is a defensible product feature and deliberately NOT built
# here: it is a product decision, not the security gap this closes.
#
# The transition table in `LifecycleService` is the authority on what is legal;
# this task adds no rules of its own. CHURNED remains terminal, so a suspended
# organization cannot churn its way back to service.


_METHOD = {
    LifecycleStatus.ACTIVE: "activate",
    LifecycleStatus.SUSPENDED: "suspend",
    LifecycleStatus.CHURNED: "churn",
}
_SETTABLE = sorted(s.value for s in _METHOD)


async def _set_lifecycle(org_id: uuid.UUID, target: LifecycleStatus) -> dict:
    async with session_scope() as db:
        svc = OrganizationService(db)
        row = await getattr(svc.lifecycle, _METHOD[target])(org_id)
        return {"org_id": str(org_id), "status": row.status}


@celery_app.task(name="navis.orchestration.org_provisioning.set_organization_lifecycle")
def set_organization_lifecycle(org_id: str, status: str) -> dict:
    """Move ONE organization to ONE lifecycle status. Operator-invoked.

        celery -A navis.platform.queue call \\
            navis.orchestration.org_provisioning.set_organization_lifecycle \\
            --args='["<org-uuid>", "suspended"]'

    Validates before it acts, and reports rather than raises on a refusal so an
    operator sees WHY instead of a traceback:

      * an unknown status is refused (the closed `LifecycleStatus` vocabulary);
      * an illegal transition is refused by `LifecycleService` (CHURNED is
        terminal, so this is also what stops a churned tenant being revived);
      * TRIAL is deliberately absent — it is the provisioning state, and moving an
        organization *back* to it would hand a suspended tenant a fresh trial.

    Idempotent: re-asserting the current status writes nothing and audits nothing
    (`LifecycleService._transition` returns early), so a retried task is a no-op.
    """
    try:
        target = LifecycleStatus(status)
    except ValueError:
        return {
            "org_id": org_id,
            "error": f"unknown lifecycle status {status!r}",
            "allowed": _SETTABLE,
        }
    if target not in _METHOD:
        return {
            "org_id": org_id,
            "error": f"{target.value} is not an operator-settable status",
            "allowed": _SETTABLE,
        }
    try:
        return run_async(_set_lifecycle(uuid.UUID(org_id), target))
    except LifecycleError as exc:
        logger.warning("lifecycle transition refused org=%s -> %s: %s", org_id, status, exc)
        return {"org_id": org_id, "error": str(exc)}
