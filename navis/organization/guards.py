"""require_capability — the dependency business domains use to gate features.

A business router declares `Depends(require_capability(Capability.EXECUTION))`; the
guard resolves the Identity principal (for org_id) and asks OrganizationService.
402 PAYMENT_REQUIRED is returned when the org is not entitled (vs 403 for authz),
so clients can distinguish "not allowed" from "not on your plan".
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.authz import Principal, get_principal
from navis.organization.contract import Capability
from navis.organization.service import OrganizationService
from navis.platform.database import get_session


def require_capability(capability: Capability):
    async def _dep(
        principal: Principal = Depends(get_principal),
        db: AsyncSession = Depends(get_session),
    ) -> Principal:
        # Fail closed at the capability boundary. principal.org_id is non-null today
        # (get_principal rejects org-less JWTs with 403; api_keys.organization_id is NOT
        # NULL), but can_use() defaults a None/unprovisioned org to TRIAL/Plan.FREE — i.e.
        # it would fail OPEN. Asserting presence here keeps the gate fail-closed even if a
        # future principal kind introduces a null org, instead of silently granting FREE.
        if getattr(principal, "org_id", None) is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "principal has no organization")
        if not await OrganizationService(db).can_use(principal.org_id, capability):
            raise HTTPException(
                status.HTTP_402_PAYMENT_REQUIRED,
                f"capability '{capability.value}' not enabled for this organization",
            )
        return principal

    return _dep
