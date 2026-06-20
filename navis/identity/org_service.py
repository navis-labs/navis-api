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

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.models import Organization

# 8 hex chars of salt (was 6 -> 16.7M space; 8 -> ~4.3e9) plus the retry loop below.
# Correctness comes from the retry, not from the (low) collision probability.
_SALT_LEN = 8
_MAX_SLUG_ATTEMPTS = 4


class OrganizationError(Exception):
    """Org creation failed (empty name, or no unique slug after retries)."""


def _slugify(name: str, salt: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "org"
    return f"{base}-{salt[:_SALT_LEN]}"


class OrganizationService:
    """The single chokepoint for org creation/lookup that Identity depends on."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, *, name: str, slug_salt: str | None = None) -> Organization:
        if not name or not name.strip():
            raise OrganizationError("organization name must not be empty")
        name = name.strip()

        # Production correctness must not depend on slug-collision probability. Insert
        # inside a SAVEPOINT so a UNIQUE(slug) violation rolls back ONLY this insert
        # (not the surrounding signup transaction), then retry with a fresh salt.
        for _ in range(_MAX_SLUG_ATTEMPTS):
            slug = _slugify(name, slug_salt or uuid.uuid4().hex)
            try:
                async with self.db.begin_nested():  # SAVEPOINT
                    org = Organization(name=name, slug=slug)
                    self.db.add(org)
                    await self.db.flush()
                return org
            except IntegrityError:
                # A caller-pinned salt is deterministic — retrying yields the same slug,
                # so fail fast instead of looping on a guaranteed collision.
                if slug_salt is not None:
                    raise OrganizationError(f"slug '{slug}' already exists")
                continue
        raise OrganizationError("could not allocate a unique organization slug")

    async def get(self, org_id: uuid.UUID) -> Organization | None:
        return await self.db.get(Organization, org_id)
