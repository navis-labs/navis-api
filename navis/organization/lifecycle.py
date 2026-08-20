"""Organization lifecycle transitions: TRIAL → ACTIVE → SUSPENDED/CHURNED.

ACTIVE is a BUSINESS status (the org reached its first brief), NOT an intelligence-
health signal: an ACTIVE org does not imply MEMORY_READY/STATE_READY or a healthy
pipeline. What an org may *run* is governed by capability gating (EntitlementService),
never by this status. Onboarding drives lifecycle (FIRST_BRIEF → ACTIVE); the reverse
is not enforced.

A suspended/churned org's capabilities are all gated off by EntitlementService.
Every transition is audited via Identity's append-only audit log.
"""
from __future__ import annotations

import uuid

from navis.identity.audit import AuditService
from navis.organization.contract import LifecycleStatus
from navis.organization.models import OrganizationLifecycle
from navis.organization.repository import OrganizationRepository
from navis.platform.database.base import utcnow
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("organization.lifecycle")

# Allowed forward/again transitions.
_ALLOWED = {
    LifecycleStatus.TRIAL: {LifecycleStatus.ACTIVE, LifecycleStatus.CHURNED},
    LifecycleStatus.ACTIVE: {LifecycleStatus.SUSPENDED, LifecycleStatus.CHURNED},
    LifecycleStatus.SUSPENDED: {LifecycleStatus.ACTIVE, LifecycleStatus.CHURNED},
    LifecycleStatus.CHURNED: set(),
}


class LifecycleError(Exception):
    pass


class LifecycleService:
    def __init__(self, repo: OrganizationRepository, audit: AuditService | None = None) -> None:
        self.repo = repo
        self.audit = audit

    async def _transition(self, org_id: uuid.UUID, target: LifecycleStatus, *, actor_id=None) -> OrganizationLifecycle:
        row = await self.repo.ensure_lifecycle(org_id)
        current = LifecycleStatus(row.status)
        if target == current:
            return row  # idempotent: re-asserting the current status writes nothing and audits nothing
        if target not in _ALLOWED[current]:
            raise LifecycleError(f"illegal transition {current.value} -> {target.value}")
        row.status = target.value
        now = utcnow()
        if target is LifecycleStatus.ACTIVE and row.activated_at is None:
            row.activated_at = now
        if target is LifecycleStatus.SUSPENDED:
            row.suspended_at = now
        if target is LifecycleStatus.CHURNED:
            row.churned_at = now
        await self.repo.flush()
        if self.audit is not None:
            await self.audit.record(
                f"org.lifecycle.{target.value}", org_id=org_id, actor_id=actor_id, target_id=str(org_id)
            )

        # AUTHORIZATION INVALIDATION. `EntitlementService` reads lifecycle from the
        # database on every capability check, so entitlements need no invalidation —
        # but the JWT principal cache holds already-authenticated Principals for up
        # to `_PRINCIPAL_CACHE_TTL_S`, and an operator suspending a tenant expects
        # that to bite immediately, not on a TTL boundary. Deferred import: Identity
        # is BELOW Organization in the dependency order and importing it at module
        # scope would invert that arrow.
        from navis.identity.authz import principal_cache_evict_org

        evicted = principal_cache_evict_org(org_id)

        # Observable: a lifecycle change is a rare, high-consequence event and the
        # single most useful line in the log when a tenant reports "everything
        # stopped". Identifier-only — no plan, no profile, no actor email.
        logger.warning(
            "org lifecycle %s -> %s org=%s actor=%s principals_evicted=%d",
            current.value, target.value, org_id, actor_id, evicted,
        )
        metrics.incr(
            "organization.lifecycle.transition",
            labels={"from": current.value, "to": target.value},
        )
        return row

    async def activate(self, org_id, *, actor_id=None):
        return await self._transition(org_id, LifecycleStatus.ACTIVE, actor_id=actor_id)

    async def suspend(self, org_id, *, actor_id=None):
        return await self._transition(org_id, LifecycleStatus.SUSPENDED, actor_id=actor_id)

    async def churn(self, org_id, *, actor_id=None):
        return await self._transition(org_id, LifecycleStatus.CHURNED, actor_id=actor_id)
