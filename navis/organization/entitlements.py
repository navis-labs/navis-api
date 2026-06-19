"""Entitlements — the single source of truth for "may this org use capability X?".

Resolution order (most authoritative first):
  1. Lifecycle gate — SUSPENDED/CHURNED orgs can use nothing.
  2. Kill switch — an enabled `kill:<capability>` flag forces the capability off.
  3. Explicit entitlement row — an org-specific grant/deny.
  4. Plan default — capabilities_for_plan(plan).
"""
from __future__ import annotations

import uuid

from navis.organization.contract import (
    Capability,
    LifecycleStatus,
    Plan,
    capabilities_for_plan,
)
from navis.organization.feature_flags import FeatureFlagService
from navis.organization.repository import OrganizationRepository


class EntitlementService:
    def __init__(self, repo: OrganizationRepository, flags: FeatureFlagService | None = None) -> None:
        self.repo = repo
        self.flags = flags or FeatureFlagService(repo)

    async def can_use(self, org_id: uuid.UUID, capability: Capability | str) -> bool:
        cap = capability.value if isinstance(capability, Capability) else capability

        lifecycle = await self.repo.get_lifecycle(org_id)
        status = lifecycle.status if lifecycle else LifecycleStatus.TRIAL.value
        if status in (LifecycleStatus.SUSPENDED.value, LifecycleStatus.CHURNED.value):
            return False

        if await self.flags.is_killed(org_id, cap):
            return False

        explicit = await self.repo.get_entitlement(org_id, cap)
        if explicit is not None:
            return bool(explicit.enabled)

        plan = Plan(lifecycle.plan) if lifecycle else Plan.FREE
        return Capability(cap) in capabilities_for_plan(plan)

    async def seed_for_plan(self, org_id: uuid.UUID, plan: Plan) -> None:
        """Write an explicit entitlement row per capability for the plan (idempotent)."""
        granted = capabilities_for_plan(plan)
        for cap in Capability:
            await self.repo.set_entitlement(org_id, cap.value, cap in granted)
        await self.repo.flush()

    async def set(self, org_id: uuid.UUID, capability: Capability | str, enabled: bool) -> None:
        cap = capability.value if isinstance(capability, Capability) else capability
        await self.repo.set_entitlement(org_id, cap, enabled)
        await self.repo.flush()

    async def snapshot(self, org_id: uuid.UUID) -> dict[str, bool]:
        return {cap.value: await self.can_use(org_id, cap) for cap in Capability}
