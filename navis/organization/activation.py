"""Activation metrics — TTFI (Time To First Intelligence) is the north star.

TTFI = first-brief timestamp − created timestamp, read from the onboarding stage
clock. activation_score = fraction of the onboarding funnel completed (0–100).
Pure given an onboarding row, so it is trivially recomputable.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from navis.organization.contract import STAGE_ORDER, OnboardingStage, stage_index
from navis.organization.models import OrganizationOnboarding
from navis.organization.repository import OrganizationRepository

# First Intelligence is operationally the first brief delivered to the founder.
_FIRST_INTELLIGENCE = OnboardingStage.FIRST_BRIEF


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def compute_ttfi_seconds(onboarding: OrganizationOnboarding) -> int | None:
    stamps = onboarding.stage_timestamps or {}
    start = _parse(stamps.get(OnboardingStage.CREATED.value))
    first = _parse(stamps.get(_FIRST_INTELLIGENCE.value))
    if start is None or first is None:
        return None
    return max(0, int((first - start).total_seconds()))


def compute_activation_score(onboarding: OrganizationOnboarding) -> float:
    reached = stage_index(OnboardingStage(onboarding.stage))
    total = len(STAGE_ORDER) - 1  # CREATED is the baseline
    return round(100.0 * reached / total, 2)


class ActivationService:
    def __init__(self, repo: OrganizationRepository) -> None:
        self.repo = repo

    async def recompute(self, org_id: uuid.UUID):
        onboarding = await self.repo.get_onboarding(org_id)
        if onboarding is None:
            return None
        return await self.repo.upsert_metrics(
            org_id,
            ttfi_seconds=compute_ttfi_seconds(onboarding),
            activation_score=compute_activation_score(onboarding),
        )
