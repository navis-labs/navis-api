"""Organization HTTP surface: profile, onboarding, entitlements, flags, plan, metrics."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.authz import Principal, get_principal, require_permission
from navis.identity.permissions import Permission
from navis.organization.contract import Capability, Plan
from navis.organization.feature_flags import FeatureFlagService
from navis.organization.repository import OrganizationRepository
from navis.organization.service import OrganizationService
from navis.platform.database import get_session

router = APIRouter(prefix="/organization", tags=["organization"])


class ProfileBody(BaseModel):
    company_name: str | None = Field(None, max_length=255)
    website: str | None = Field(None, max_length=255)
    industry: str | None = Field(None, max_length=128)
    # Non-negative; an org cannot have a negative headcount.
    employee_count: int | None = Field(None, ge=0)
    # ISO 3166-1 alpha-2. The column is varchar(2); without this a 3+ char value
    # overflows the column and surfaces as an unhandled 500 instead of a clean 422.
    hq_country: str | None = Field(None, min_length=2, max_length=2)


class PlanBody(BaseModel):
    plan: Plan


class FlagBody(BaseModel):
    enabled: bool
    # Closed vocabulary (matches OrganizationFeatureFlag.kind); rejects typos with 422.
    kind: Literal["release", "permission", "kill_switch"] = "release"


@router.get("/profile")
async def get_profile(
    p: Principal = Depends(get_principal),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    row = await OrganizationService(db).get_profile(p.org_id)
    if row is None:
        return {"profile": None}
    return {"profile": {
        "company_name": row.company_name, "website": row.website, "industry": row.industry,
        "employee_count": row.employee_count, "hq_country": row.hq_country,
    }}


@router.put("/profile")
async def update_profile(
    body: ProfileBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await OrganizationService(db).update_profile(p.org_id, body.model_dump(), actor_id=p.subject_id)
    return {"status": "updated"}


@router.get("/entitlements")
async def entitlements(
    p: Principal = Depends(get_principal),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    return {"entitlements": await OrganizationService(db).entitlements.snapshot(p.org_id)}


@router.get("/onboarding")
async def onboarding(
    p: Principal = Depends(get_principal),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    row = await OrganizationRepository(db).get_onboarding(p.org_id)
    return {"stage": row.stage if row else None,
            "signals_count": row.signals_count if row else 0,
            "memory_count": row.memory_count if row else 0}


@router.get("/metrics")
async def metrics(
    p: Principal = Depends(get_principal),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    return await OrganizationService(db).metrics.rollup(p.org_id)


@router.put("/plan")
async def set_plan(
    body: PlanBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_BILLING)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await OrganizationService(db).billing.set_plan(p.org_id, body.plan, actor_id=p.subject_id)
    return {"plan": body.plan.value}


@router.put("/flags/{flag}")
async def set_flag(
    flag: str,
    body: FlagBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_BILLING)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await FeatureFlagService(OrganizationRepository(db)).set(
        p.org_id, flag, kind=body.kind, enabled=body.enabled
    )
    return {"flag": flag, "enabled": body.enabled}
