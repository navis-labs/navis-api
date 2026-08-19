"""Entitlements — the single source of truth for "may this org use capability X?".

Resolution order (most authoritative first):
  1. Lifecycle gate — only TRIAL/ACTIVE orgs can use anything.
  2. Kill switch — an enabled `kill:<capability>` flag forces the capability off.
  3. Explicit entitlement row — an org-specific grant/deny.
  4. Plan default — capabilities_for_plan(plan).

The decision depends ONLY on the org's own persisted state (lifecycle, flags,
entitlement rows, plan). It is deterministic and environment-independent: there
is no deployment, debug or test branch anywhere in this module. Environments that
need broader access get it the honest way — real entitlement rows, seeded from a
real plan (see OrganizationService.bootstrap) — so what dev exercises is the same
code path production runs.
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

    @staticmethod
    def _decide(
        *,
        status: str | None,
        plan: str | None,
        explicit: bool | None,
        killed: bool,
        capability: str,
    ) -> bool:
        """The single decision function — both can_use() (one row) and snapshot() (N rows)
        feed through this, so they can NEVER disagree about a capability. Resolution
        order matches the module docstring: lifecycle -> kill switch -> explicit -> plan.

        EVERY exit from this function is a deliberate grant. Malformed or unrecognised
        input denies; nothing falls through to True.
        """
        # Lifecycle. Tested against the PERMITTED statuses, not the denied ones: an
        # unrecognised status (a legacy row, a future value, hand-edited data) must
        # deny rather than sail past a `not in (SUSPENDED, CHURNED)` check as though
        # the org were healthy. Identical for every value LifecycleService can write.
        #
        # A missing lifecycle row (status None) is NOT malformed — it is the gap
        # between Identity creating the org and the async ORG_CREATED handler
        # provisioning it. It resolves to TRIAL/FREE below, the product's minimum
        # tier, so a half-provisioned org can read its own mail but can never reach
        # EXECUTION, DECISION or INTELLIGENCE.
        s = status or LifecycleStatus.TRIAL.value
        if s not in (LifecycleStatus.TRIAL.value, LifecycleStatus.ACTIVE.value):
            return False
        if killed:
            return False
        if explicit is not None:
            return bool(explicit)
        # An unrecognised plan string denies instead of raising ValueError -> 500.
        # Same class of bug as the unknown capability below: the plan column is free
        # text at the database level, so a legacy/renamed/mistyped value reaches here,
        # and a 500 on the capability gate is an outage, not a denial.
        try:
            p = Plan(plan) if plan else Plan.FREE
        except ValueError:
            return False
        # Compare by VALUE (not Capability(capability)) so an unknown/garbage capability
        # string denies (False) instead of raising ValueError -> 500. Identical result
        # for valid capabilities; deny-by-default for anything outside the enum.
        return capability in {c.value for c in capabilities_for_plan(p)}

    async def can_use(self, org_id: uuid.UUID, capability: Capability | str) -> bool:
        cap = capability.value if isinstance(capability, Capability) else capability
        # ONE round-trip for all decision inputs (was 3 sequential reads — the hot path).
        r = await self.repo.resolve_capability(org_id, cap, kill_flag=kill_switch_flag(cap))
        return self._decide(
            status=r["status"], plan=r["plan"],
            explicit=r["explicit"], killed=r["killed"], capability=cap,
        )

    async def can_use_all(
        self, org_id: uuid.UUID, capabilities: list[Capability | str]
    ) -> tuple[bool, list[str], str | None]:
        """Are ALL of these capabilities currently available? One round-trip.

        Returns `(allowed, denied, lifecycle_status)`:

          * `denied` names WHICH capabilities failed, so an audit record can say
            that instead of a bare "not authorized";
          * `lifecycle_status` is the org's current status, passed straight
            through from the row thi
            s query already fetched. It carries NO extra
            decision — `_decide` remains the only place lifecycle is judged — but
            it is what lets an operator tell "suspended for non-payment" from
            "kill switch thrown" when a wave of executions is refused. Without it
            both look identical: every capability denied, no stated cause.

        The execution worker re-checks authorization immediately before every
        provider call and needs two answers (the blanket EXECUTION capability and
        the action's connector capability). Asking twice would put 2×Neon RTT in
        front of every side effect; `snapshot_capabilities` already fetches N
        capabilities in one query, so this reuses it — and runs each row through
        the SAME `_decide` as `can_use`, so the pre-flight gate and the worker
        re-check can never disagree about the same org.

        Fail closed on an empty result: a capability the query did not answer for
        counts as denied, never as allowed by omission.
        """
        caps = [c.value if isinstance(c, Capability) else c for c in capabilities]
        if not caps:
            return True, [], None
        rows = await self.repo.snapshot_capabilities(org_id, caps)
        decided = {
            row["capability"]: self._decide(
                status=row["status"], plan=row["plan"],
                explicit=row["explicit"], killed=row["killed"],
                capability=row["capability"],
            )
            for row in rows
        }
        # Every row carries the same org's status; take it from the first.
        status = next((row["status"] for row in rows), None)
        # `.get(cap, False)` — a capability missing from the result set denies.
        denied = [cap for cap in caps if not decided.get(cap, False)]
        return not denied, denied, status

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
        """All capabilities for an org in ONE round-trip — the dashboard reads this on
        every page load, so N×can_use would scale linearly with the Capability enum.
        Both paths share _decide(), so single-cap and bulk can never drift."""
        caps = [c.value for c in Capability]
        rows = await self.repo.snapshot_capabilities(org_id, caps)
        return {
            row["capability"]: self._decide(
                status=row["status"], plan=row["plan"],
                explicit=row["explicit"], killed=row["killed"],
                capability=row["capability"],
            )
            for row in rows
        }
