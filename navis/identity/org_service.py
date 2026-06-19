"""Organization-lifecycle seam (P2 extraction prep).

Per the Identity Law, organization profile/lifecycle belongs to a future
Organization V1 domain, NOT to Identity. To keep that extraction a *move* rather
than a *rewrite*, all organization mutations funnel through this one service.
Today it wraps the existing `organizations` table; tomorrow its body is replaced
by a call into the Organization domain and Identity keeps only the seam.
"""
from __future__ import annotations

import re
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.models import Organization


def _slugify(name: str, salt: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "org"
    return f"{base}-{salt[:6]}"


class OrganizationService:
    """The single chokepoint for org creation/lookup that Identity depends on."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, *, name: str, slug_salt: str | None = None) -> Organization:
        org = Organization(name=name, slug=_slugify(name, slug_salt or uuid.uuid4().hex))
        self.db.add(org)
        await self.db.flush()
        return org

    async def get(self, org_id: uuid.UUID) -> Organization | None:
        return await self.db.get(Organization, org_id)
