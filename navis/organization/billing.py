"""Billing/plan management. V1 is plan assignment + entitlement reseed — no payment
processor (Stripe integration is a later concern; the seam is here)."""
from __future__ import annotations

import uuid

from navis.identity.audit import AuditService
from navis.organization.contract import Plan
from navis.organization.entitlements import EntitlementService
from navis.organization.repository import OrganizationRepository


class BillingService:
    def __init__(self, repo: OrganizationRepository, entitlements: EntitlementService | None = None, audit: AuditService | None = None) -> None:
        self.repo = repo
        self.entitlements = entitlements or EntitlementService(repo)
        self.audit = audit

    async def get_plan(self, org_id: uuid.UUID) -> Plan:
        row = await self.repo.get_lifecycle(org_id)
        return Plan(row.plan) if row else Plan.FREE

    async def set_plan(self, org_id: uuid.UUID, plan: Plan, *, actor_id=None) -> None:
        lifecycle = await self.repo.ensure_lifecycle(org_id)
        lifecycle.plan = plan.value
        await self.repo.flush()
        # A plan change reseeds capability entitlements to the plan's defaults.
        await self.entitlements.seed_for_plan(org_id, plan)
        if self.audit is not None:
            await self.audit.record(
                f"org.plan.{plan.value}", org_id=org_id, actor_id=actor_id, target_id=str(org_id)
            )
