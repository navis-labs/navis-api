"""Organization lifecycle transitions: TRIAL → ACTIVE → SUSPENDED/CHURNED.

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
        if target != current and target not in _ALLOWED[current]:
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
        return row

    async def activate(self, org_id, *, actor_id=None):
        return await self._transition(org_id, LifecycleStatus.ACTIVE, actor_id=actor_id)

    async def suspend(self, org_id, *, actor_id=None):
        return await self._transition(org_id, LifecycleStatus.SUSPENDED, actor_id=actor_id)

    async def churn(self, org_id, *, actor_id=None):
        return await self._transition(org_id, LifecycleStatus.CHURNED, actor_id=actor_id)
