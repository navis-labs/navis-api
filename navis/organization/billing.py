"""Billing/plan management. V1 is plan assignment + entitlement reseed — no payment
processor (Stripe integration is a later concern; the seam is here)."""
from __future__ import annotations

import uuid
from typing import Awaitable, Callable

from navis.identity.audit import AuditService
from navis.organization.contract import Plan, capabilities_for_plan
from navis.organization.entitlements import EntitlementService
from navis.organization.repository import OrganizationRepository
from navis.platform.database.base import utcnow
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("organization.billing")

# Publisher seam (ORG-003): emits OrganizationPlanChanged. Injected so BillingService stays
# decoupled from the bus and unit-testable; OrganizationService wires the real publisher.
PlanChangePublisher = Callable[..., Awaitable[None]]


class PlanEscalationDenied(Exception):
    """A plan change would GRANT capabilities and no billing authority backed it.

    THE HOLE THIS CLOSES. `PUT /organization/plan` gates on
    `Permission.MANAGE_BILLING`, which every FOUNDER holds by default — and the
    first user of every organization is a founder. `set_plan` then called
    `seed_for_plan`, which writes EXPLICIT `organization_entitlements` rows.
    Explicit rows outrank plan defaults in `EntitlementService._decide`, so
    `{"plan": "enterprise"}` permanently granted the caller's own organization all
    fourteen capabilities — EXECUTION, ADAPTIVE, INTELLIGENCE, ML — with no
    payment, no approval and no way to tell it apart from a legitimate upgrade.
    That made the entire plan layer advisory.

    DELIBERATELY NOT SOLVED WITH AN ENVIRONMENT BRANCH. `guards.require_capability`
    used to return unchecked whenever `is_production` was false, and because
    ENVIRONMENT defaults to development an unset variable removed the capability
    boundary entirely. The rule here is therefore environment-blind: nobody
    escalates, anywhere. Development still exercises the full product because
    `OrganizationService.bootstrap` PROVISIONS non-production orgs on ENTERPRISE —
    a provisioning decision, which is allowed to know about deployments, rather
    than an authorization one, which is not.
    """


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

    @staticmethod
    def _refuse_escalation(org_id, old_plan: str | None, plan: Plan, *, actor_id=None) -> None:
        """Raise unless `plan` grants nothing beyond the organization's current plan.

        An unrecognised stored plan resolves to FREE — the SAME fallback
        `EntitlementService._decide` applies to an unparseable plan column — so the
        two can never disagree about what an organization currently holds. Falling
        back to FREE is also the strict direction: it treats the org as holding the
        least, so every real change looks like an escalation and is refused rather
        than waved through.
        """
        try:
            current = Plan(old_plan) if old_plan else Plan.FREE
        except ValueError:
            current = Plan.FREE
        gained = capabilities_for_plan(plan) - capabilities_for_plan(current)
        if not gained:
            return
        logger.warning(
            "plan escalation DENIED org=%s actor=%s %s -> %s gains=%s",
            org_id, actor_id, current.value, plan.value,
            ",".join(sorted(c.value for c in gained)),
        )
        metrics.incr(
            "organization.plan.escalation_denied",
            labels={"from": current.value, "to": plan.value},
        )
        raise PlanEscalationDenied(
            f"changing plan from '{current.value}' to '{plan.value}' would grant "
            f"{', '.join(sorted(c.value for c in gained))}; a capability-granting "
            "plan change requires billing authorization and cannot be self-assigned"
        )

    async def set_plan(
        self,
        org_id: uuid.UUID,
        plan: Plan,
        *,
        actor_id=None,
        billing_authorized: bool = False,
    ) -> None:
        """Assign a plan and reseed entitlements to its defaults.

        `billing_authorized` is the ONE capability-granting authority. It is a
        keyword-only server-side flag with no route, body field or header behind
        it: no HTTP caller can set it, so no tenant can grant itself capabilities.
        It is passed by the two callers that legitimately represent a billing
        decision —

          * `OrganizationService.bootstrap`, which provisions a NEW organization
            onto its starting plan (FREE in production, ENTERPRISE elsewhere);
          * a future payment-processor integration, which is exactly the seam this
            module's docstring reserves.

        Without it, a plan change may only KEEP OR REDUCE the organization's
        capability set. Downgrades stay fully self-service — giving up
        capabilities harms nobody and needs no approval — so a founder can still
        move themselves off a paid tier.

        Comparison is on the CAPABILITY SETS, not on an ordering of plan names.
        There is no total order over plans (a future SKU could add one capability
        and drop another), and `capabilities_for_plan` is already the single
        source of truth `EntitlementService` decides from. Subset containment is
        the honest question: does this change hand the organization anything it
        does not already have?
        """
        lifecycle = await self.repo.ensure_lifecycle(org_id)
        old_plan = lifecycle.plan
        if not billing_authorized:
            self._refuse_escalation(org_id, old_plan, plan, actor_id=actor_id)
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
