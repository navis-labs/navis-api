"""API keys: hashed-at-rest, scoped, revocable credentials.

Format: navis_<prefix>_<secret>. Only the SHA-256 of the full string is stored,
plus the public prefix for O(1) lookup. The plaintext is returned exactly once.
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.enterprise_models import ApiKey

_PREFIX_BYTES = 8   # -> 16 hex chars (fits String(16)); ~1.8e19 space, collision-free at scale
_SECRET_BYTES = 32
# Only refresh last_used_at when it's this stale — avoids a write on EVERY auth
# request (write-amplification) while keeping "last used" accurate to the minute.
_LAST_USED_REFRESH = timedelta(minutes=15)



def _hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


def _mint() -> tuple[str, str]:
    prefix = secrets.token_hex(_PREFIX_BYTES)
    secret = secrets.token_urlsafe(_SECRET_BYTES)
    return prefix, f"navis_{prefix}_{secret}"


class ApiKeyService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(
        self,
        *,
        org_id: uuid.UUID,
        name: str,
        scopes: list[str],
        created_by: uuid.UUID | None = None,
        service_account_id: uuid.UUID | None = None,
    ) -> tuple[ApiKey, str]:
        prefix, plaintext = _mint()
        row = ApiKey(
            organization_id=org_id,
            name=name,
            prefix=prefix,
            hashed_secret=_hash(plaintext),
            scopes=list(scopes),
            created_by=created_by,
            service_account_id=service_account_id,
        )
        self.db.add(row)
        await self.db.flush()
        return row, plaintext  # plaintext shown once, never stored

    async def verify(self, plaintext: str) -> ApiKey | None:
        """Resolve a presented key to its (active) row, updating last_used_at."""
        # The secret segment may itself contain "_" (base64url), so split at most twice.
        parts = plaintext.split("_", 2)
        if len(parts) != 3 or parts[0] != "navis":
            return None
        prefix = parts[1]
        # prefix is NOT unique (8 hex) — iterate all active matches and constant-time
        # compare, so a prefix collision can never reject a genuinely valid key.
        digest = _hash(plaintext)
        rows = (
            await self.db.scalars(
                select(ApiKey).where(ApiKey.prefix == prefix, ApiKey.revoked_at.is_(None))
            )
        ).all()
        now = datetime.now(timezone.utc)
        for row in rows:
            if secrets.compare_digest(row.hashed_secret, digest):
                # Throttle the write: only bump last_used_at when it's stale, so a key
                # doing 100 req/s doesn't generate 100 writes/s for no business value.
                if row.last_used_at is None or (now - row.last_used_at) > _LAST_USED_REFRESH:
                    row.last_used_at = now
                return row
        return None

    async def revoke(self, key_id: uuid.UUID, *, org_id: uuid.UUID) -> ApiKey | None:
        """Org-scoped: a key may only be revoked by its OWNING org (tenant isolation)."""
        row = await self.db.get(ApiKey, key_id)
        if row is None or row.organization_id != org_id:
            return None  # not found OR belongs to another tenant -> 404 at the edge
        if row.revoked_at is None:
            row.revoked_at = datetime.now(timezone.utc)
            await self.db.flush()
        return row

    async def revoke_for_service_account(self, service_account_id: uuid.UUID) -> None:
        # Single bulk UPDATE — a service account may have many keys; no row-load loop.
        await self.db.execute(
            update(ApiKey)
            .where(
                ApiKey.service_account_id == service_account_id, ApiKey.revoked_at.is_(None)
            )
            .values(revoked_at=datetime.now(timezone.utc))
        )

    async def list_for_org(
        self, org_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> list[ApiKey]:
        return list(
            (
                await self.db.scalars(
                    select(ApiKey)
                    .where(ApiKey.organization_id == org_id)
                    .order_by(ApiKey.created_at.desc(), ApiKey.id)
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        )
