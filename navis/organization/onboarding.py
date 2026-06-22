"""Onboarding state machine: CREATED → … → ACTIVATED (monotonic forward only).

Each transition stamps the stage clock (for TTFI), emits an onboarding event via an
injectable publisher, audits it, and recomputes activation. Reaching FIRST_BRIEF/
ACTIVATED also flips the org lifecycle to ACTIVE.
"""
from __future__ import annotations

import uuid
from typing import Awaitable, Callable

from navis.identity.audit import AuditService
from navis.organization.activation import ActivationService
from navis.organization.contract import STAGE_EVENT, OnboardingStage, stage_index
from navis.organization.lifecycle import LifecycleService
from navis.organization.models import OrganizationOnboarding
from navis.organization.repository import OrganizationRepository
from navis.platform.database.base import utcnow

Publisher = Callable[[str, dict], Awaitable[None]]


async def _noop_publisher(event: str, payload: dict) -> None:
    return None


class OnboardingService:
    def __init__(
        self,
        repo: OrganizationRepository,
        *,
        audit: AuditService | None = None,
        publisher: Publisher | None = None,
    ) -> None:
        self.repo = repo
        self.audit = audit
        self.publisher = publisher or _noop_publisher
        self.activation = ActivationService(repo)
        self.lifecycle = LifecycleService(repo, audit=audit)

    async def advance(
        self,
        org_id: uuid.UUID,
        stage: OnboardingStage,
        *,
        signals_count: int | None = None,
        memory_count: int | None = None,
        actor_id=None,
    ) -> OrganizationOnboarding:
        # Lock the row: advance read-modify-writes stage_timestamps, so concurrent
        # advances (independent domain events, possibly on separate workers) must
        # serialize or one stamp is lost. The lock is held until this txn commits below.
        row = await self.repo.ensure_onboarding(org_id, for_update=True)
        current = OnboardingStage(row.stage)

        if signals_count is not None:
            row.signals_count = signals_count
        if memory_count is not None:
            row.memory_count = memory_count

        # Monotonic: never move backwards, and re-asserting the CURRENT stage is a no-op
        # (idempotent) — counts above are still persisted, but we do NOT re-publish /
        # re-audit / re-activate. Only a strictly-forward stage runs the full transition.
        if stage_index(stage) <= stage_index(current):
            await self.repo.flush()
            return row

        stamps = dict(row.stage_timestamps or {})
        stamps.setdefault(stage.value, utcnow().isoformat())
        row.stage_timestamps = stamps
        row.stage = stage.value
        if stage in (OnboardingStage.FIRST_BRIEF, OnboardingStage.ACTIVATED):
            row.completed_at = utcnow()
        await self.repo.flush()

        # Do ALL DB work (audit, lifecycle activation, activation metrics) and COMMIT
        # before announcing — the event must never reference a stage the DB hasn't
        # durably committed (commit-before-publish; mirrors the signals ingest fix).
        # All writes share one transaction, so the commit is atomic.
        if self.audit is not None:
            await self.audit.record(STAGE_EVENT[stage], org_id=org_id, actor_id=actor_id, target_id=str(org_id))
        # The brief is first intelligence — the org becomes ACTIVE then.
        if stage in (OnboardingStage.FIRST_BRIEF, OnboardingStage.ACTIVATED):
            await self.lifecycle.activate(org_id, actor_id=actor_id)
        await self.activation.recompute(org_id)
        await self.repo.commit()
        await self.publisher(STAGE_EVENT[stage], {"organization_id": str(org_id), "stage": stage.value})
        return row
