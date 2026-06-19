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
        if not await OrganizationService(db).can_use(principal.org_id, capability):
            raise HTTPException(
                status.HTTP_402_PAYMENT_REQUIRED,
                f"capability '{capability.value}' not enabled for this organization",
            )
        return principal

    return _dep
