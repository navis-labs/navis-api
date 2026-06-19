"""Read-side org metrics: a single rollup of activation + onboarding state."""
from __future__ import annotations

import uuid
from typing import Any

from navis.organization.contract import LifecycleStatus, Plan
from navis.organization.repository import OrganizationRepository


class MetricsService:
    def __init__(self, repo: OrganizationRepository) -> None:
        self.repo = repo

    async def rollup(self, org_id: uuid.UUID) -> dict[str, Any]:
        onboarding = await self.repo.get_onboarding(org_id)
        lifecycle = await self.repo.get_lifecycle(org_id)
        metrics = await self.repo.get_metrics(org_id)
        return {
            "status": lifecycle.status if lifecycle else LifecycleStatus.TRIAL.value,
            "plan": lifecycle.plan if lifecycle else Plan.FREE.value,
            "stage": onboarding.stage if onboarding else None,
            "signals_count": onboarding.signals_count if onboarding else 0,
            "memory_count": onboarding.memory_count if onboarding else 0,
            "ttfi_seconds": metrics.ttfi_seconds if metrics else None,
            "activation_score": metrics.activation_score if metrics else 0.0,
        }
