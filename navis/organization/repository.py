"""OrganizationRepository — persistence for the organization tables only.

Imports platform + its own models. Never a business domain.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.organization.models import (
    OrganizationEntitlement,
    OrganizationFeatureFlag,
    OrganizationLifecycle,
    OrganizationMetrics,
    OrganizationOnboarding,
    OrganizationProfile,
)


class OrganizationRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ---- profile ----
    async def get_profile(self, org_id: uuid.UUID) -> OrganizationProfile | None:
        return await self.db.get(OrganizationProfile, org_id)

    async def upsert_profile(self, org_id: uuid.UUID, fields: dict[str, Any]) -> OrganizationProfile:
        row = await self.get_profile(org_id)
        if row is None:
            row = OrganizationProfile(organization_id=org_id, **fields)
            self.db.add(row)
        else:
            for k, v in fields.items():
                setattr(row, k, v)
        await self.db.flush()
        return row

    # ---- lifecycle ----
    async def get_lifecycle(self, org_id: uuid.UUID) -> OrganizationLifecycle | None:
        return await self.db.get(OrganizationLifecycle, org_id)

    async def ensure_lifecycle(self, org_id: uuid.UUID) -> OrganizationLifecycle:
        row = await self.get_lifecycle(org_id)
        if row is None:
            row = OrganizationLifecycle(organization_id=org_id)
            self.db.add(row)
            await self.db.flush()
        return row

    # ---- onboarding ----
    async def get_onboarding(self, org_id: uuid.UUID) -> OrganizationOnboarding | None:
        return await self.db.get(OrganizationOnboarding, org_id)

    async def ensure_onboarding(self, org_id: uuid.UUID) -> OrganizationOnboarding:
        row = await self.get_onboarding(org_id)
        if row is None:
            row = OrganizationOnboarding(organization_id=org_id)
            self.db.add(row)
            await self.db.flush()
        return row

    # ---- entitlements ----
    async def get_entitlement(self, org_id: uuid.UUID, capability: str) -> OrganizationEntitlement | None:
        return await self.db.scalar(
            select(OrganizationEntitlement).where(
                OrganizationEntitlement.organization_id == org_id,
                OrganizationEntitlement.capability == capability,
            )
        )

    async def list_entitlements(self, org_id: uuid.UUID) -> list[OrganizationEntitlement]:
        return list(
            (
                await self.db.scalars(
                    select(OrganizationEntitlement).where(OrganizationEntitlement.organization_id == org_id)
                )
            ).all()
        )

    async def set_entitlement(self, org_id: uuid.UUID, capability: str, enabled: bool) -> None:
        stmt = (
            pg_insert(OrganizationEntitlement)
            .values(organization_id=org_id, capability=capability, enabled=enabled)
            .on_conflict_do_update(
                index_elements=["organization_id", "capability"], set_={"enabled": enabled}
            )
        )
        await self.db.execute(stmt)

    # ---- feature flags ----
    async def get_flag(self, org_id: uuid.UUID | None, flag: str) -> OrganizationFeatureFlag | None:
        return await self.db.scalar(
            select(OrganizationFeatureFlag).where(
                OrganizationFeatureFlag.organization_id == org_id,
                OrganizationFeatureFlag.flag == flag,
            )
        )

    async def flags_for(self, org_id: uuid.UUID, flag: str) -> list[OrganizationFeatureFlag]:
        """Both the org-specific row and the global default (org_id NULL)."""
        return list(
            (
                await self.db.scalars(
                    select(OrganizationFeatureFlag).where(
                        OrganizationFeatureFlag.flag == flag,
                        OrganizationFeatureFlag.organization_id.in_([org_id, None]),
                    )
                )
            ).all()
        )

    async def set_flag(self, org_id: uuid.UUID | None, flag: str, *, kind: str, enabled: bool) -> None:
        existing = await self.get_flag(org_id, flag)
        if existing is None:
            self.db.add(OrganizationFeatureFlag(organization_id=org_id, flag=flag, kind=kind, enabled=enabled))
        else:
            existing.kind = kind
            existing.enabled = enabled
        await self.db.flush()

    # ---- metrics ----
    async def get_metrics(self, org_id: uuid.UUID) -> OrganizationMetrics | None:
        return await self.db.get(OrganizationMetrics, org_id)

    async def upsert_metrics(self, org_id: uuid.UUID, *, ttfi_seconds: int | None, activation_score: float) -> OrganizationMetrics:
        row = await self.get_metrics(org_id)
        if row is None:
            row = OrganizationMetrics(
                organization_id=org_id, ttfi_seconds=ttfi_seconds, activation_score=activation_score
            )
            self.db.add(row)
        else:
            row.ttfi_seconds = ttfi_seconds
            row.activation_score = activation_score
        await self.db.flush()
        return row

    async def flush(self) -> None:
        await self.db.flush()
