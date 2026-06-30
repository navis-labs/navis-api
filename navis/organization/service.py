"""OrganizationService — the single front door + the capability source of truth.

`can_use(org_id, capability)` is the one authority every business domain calls.
`bootstrap` provisions a new org's lifecycle/onboarding/entitlements at signup.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.audit import AuditService
from navis.organization.activation import ActivationService
from navis.organization.billing import BillingService
from navis.organization.contract import Capability, OnboardingStage, Plan
from navis.organization.entitlements import EntitlementService
from navis.organization.events import bus_publisher, publish_plan_changed
from navis.organization.feature_flags import FeatureFlagService
from navis.organization.lifecycle import LifecycleService
from navis.organization.metrics import MetricsService
from navis.organization.onboarding import OnboardingService
from navis.organization.repository import OrganizationRepository
from navis.platform.database.base import utcnow

# Minimum fields required to advance to PROFILE_COMPLETE.
# These are the onboarding form fields the user must fill in.
_PROFILE_REQUIRED = ("company_name", "industry", "team_size", "org_stage")


class OrganizationService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = OrganizationRepository(db)
        self.audit = AuditService(db)
        self.flags = FeatureFlagService(self.repo)
        self.entitlements = EntitlementService(self.repo, self.flags)
        self.billing = BillingService(self.repo, self.entitlements, self.audit, publisher=publish_plan_changed)
        self.lifecycle = LifecycleService(self.repo, self.audit)
        self.onboarding = OnboardingService(self.repo, audit=self.audit, publisher=bus_publisher)
        self.activation = ActivationService(self.repo)
        self.metrics = MetricsService(self.repo)

    # ---- provisioning ----
    async def bootstrap(self, org_id: uuid.UUID, *, plan: Plan = Plan.FREE, actor_id=None) -> None:
        """Idempotent. Triggered by the ORG_CREATED event (never imported by Identity).

        Stamps the CREATED clock for TTFI but does NOT re-emit ORG_CREATED — Identity
        already published it at signup, so re-emitting would loop the provisioner.
        """
        await self.repo.ensure_lifecycle(org_id)
        onboarding = await self.repo.ensure_onboarding(org_id)
        if OnboardingStage.CREATED.value not in (onboarding.stage_timestamps or {}):
            onboarding.stage_timestamps = {
                **(onboarding.stage_timestamps or {}),
                OnboardingStage.CREATED.value: utcnow().isoformat(),
            }
            await self.repo.flush()
        await self.billing.set_plan(org_id, plan, actor_id=actor_id)  # seeds entitlements

    # ---- the capability authority ----
    async def can_use(self, org_id: uuid.UUID, capability: Capability | str) -> bool:
        return await self.entitlements.can_use(org_id, capability)

    # ---- profile ----
    async def update_profile(self, org_id: uuid.UUID, fields: dict[str, Any], *, actor_id=None):
        row = await self.repo.upsert_profile(org_id, {k: v for k, v in fields.items() if v is not None})
        if all(getattr(row, f, None) for f in _PROFILE_REQUIRED):
            await self.onboarding.advance(org_id, OnboardingStage.PROFILE_COMPLETE, actor_id=actor_id)
        return row

    async def get_profile(self, org_id: uuid.UUID):
        return await self.repo.get_profile(org_id)
