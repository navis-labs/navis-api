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
        # Single anchored LEFT JOIN (1 round-trip) instead of 3 sequential PK reads.
        # An absent table yields NULL columns — same defaulting as the independent reads.
        r = await self.repo.rollup_row(org_id)
        return {
            "status": r["status"] or LifecycleStatus.TRIAL.value,
            "plan": r["plan"] or Plan.FREE.value,
            "stage": r["stage"],
            "signals_count": r["signals_count"] if r["signals_count"] is not None else 0,
            "memory_count": r["memory_count"] if r["memory_count"] is not None else 0,
            "ttfi_seconds": r["ttfi_seconds"],
            "activation_score": r["activation_score"] if r["activation_score"] is not None else 0.0,
        }
