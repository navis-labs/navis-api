"""Billing/plan management. V1 is plan assignment + entitlement reseed — no payment
processor (Stripe integration is a later concern; the seam is here)."""
from __future__ import annotations

import uuid
from typing import Awaitable, Callable

from navis.identity.audit import AuditService
from navis.organization.contract import Plan
from navis.organization.entitlements import EntitlementService
from navis.organization.repository import OrganizationRepository
from navis.platform.database.base import utcnow

# Publisher seam (ORG-003): emits OrganizationPlanChanged. Injected so BillingService stays
# decoupled from the bus and unit-testable; OrganizationService wires the real publisher.
PlanChangePublisher = Callable[..., Awaitable[None]]


class BillingService:
    def __init__(
        self,
        repo: OrganizationRepository,
        entitlements: EntitlementService | None = None,
        audit: AuditService | None = None,
        publisher: PlanChangePublisher | None = None,
    ) -> None:
        self.repo = repo
        self.entitlements = entitlements or EntitlementService(repo)
        self.audit = audit
        self.publisher = publisher

    async def get_plan(self, org_id: uuid.UUID) -> Plan:
        row = await self.repo.get_lifecycle(org_id)
        return Plan(row.plan) if row else Plan.FREE

    async def set_plan(self, org_id: uuid.UUID, plan: Plan, *, actor_id=None) -> None:
        lifecycle = await self.repo.ensure_lifecycle(org_id)
        old_plan = lifecycle.plan
        if old_plan == plan.value:
            # Idempotent: already on this plan. Re-seeding would rewrite every capability
            # to the plan default and CLOBBER explicit per-org entitlement overrides
            # (e.g. an admin-granted capability beyond the plan); it would also append a
            # redundant audit row AND a redundant domain event. A no-op re-assert must
            # change nothing. can_use() already resolves plan defaults via fallback.
            return
        lifecycle.plan = plan.value
        await self.repo.flush()
        # A genuine plan CHANGE reseeds capability entitlements to the new plan's defaults.
        await self.entitlements.seed_for_plan(org_id, plan)
        if self.audit is not None:
            await self.audit.record(
                f"org.plan.{plan.value}", org_id=org_id, actor_id=actor_id, target_id=str(org_id)
            )
        # Commit-before-publish (mirrors onboarding.advance): the change is durable before
        # any consumer sees the event, so a handler reading the DB never sees stale state.
        # Emitted exactly once per real change (the no-op guard above prevents duplicates).
        await self.repo.commit()
        if self.publisher is not None:
            await self.publisher(
                org_id=org_id,
                old_plan=old_plan,
                new_plan=plan.value,
                actor_id=actor_id,
                changed_at=utcnow().isoformat(),
            )
