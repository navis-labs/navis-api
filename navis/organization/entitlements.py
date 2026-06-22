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
from navis.organization.feature_flags import FeatureFlagService, kill_switch_flag
from navis.organization.repository import OrganizationRepository


class EntitlementService:
    def __init__(self, repo: OrganizationRepository, flags: FeatureFlagService | None = None) -> None:
        self.repo = repo
        self.flags = flags or FeatureFlagService(repo)

    async def can_use(self, org_id: uuid.UUID, capability: Capability | str) -> bool:
        cap = capability.value if isinstance(capability, Capability) else capability

        # ONE round-trip for all decision inputs (was 3 sequential reads — the hot path).
        # Resolution order is unchanged: lifecycle gate -> kill switch -> explicit -> plan.
        r = await self.repo.resolve_capability(org_id, cap, kill_flag=kill_switch_flag(cap))

        status = r["status"] or LifecycleStatus.TRIAL.value
        if status in (LifecycleStatus.SUSPENDED.value, LifecycleStatus.CHURNED.value):
            return False

        if r["killed"]:
            return False

        if r["explicit"] is not None:
            return bool(r["explicit"])

        plan = Plan(r["plan"]) if r["plan"] else Plan.FREE
        # Compare by VALUE (not Capability(cap)) so an unknown/garbage capability
        # string denies (False) instead of raising ValueError -> 500. Identical result
        # for valid capabilities; deny-by-default for anything outside the enum.
        return cap in {c.value for c in capabilities_for_plan(plan)}

    async def seed_for_plan(self, org_id: uuid.UUID, plan: Plan) -> None:
        """Write an explicit entitlement row per capability for the plan (idempotent).

        One batched upsert (not N sequential writes) so provisioning stays within the
        event-handler timeout regardless of how many capabilities exist.
        """
        granted = capabilities_for_plan(plan)
        await self.repo.set_entitlements(org_id, {cap.value: cap in granted for cap in Capability})
        await self.repo.flush()

    async def set(self, org_id: uuid.UUID, capability: Capability | str, enabled: bool) -> None:
        cap = capability.value if isinstance(capability, Capability) else capability
        await self.repo.set_entitlement(org_id, cap, enabled)
        await self.repo.flush()

    async def snapshot(self, org_id: uuid.UUID) -> dict[str, bool]:
        return {cap.value: await self.can_use(org_id, cap) for cap in Capability}
