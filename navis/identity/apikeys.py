"""API keys: hashed-at-rest, scoped, revocable credentials.

Format: navis_<prefix>_<secret>. Only the SHA-256 of the full string is stored,
plus the public prefix for O(1) lookup. The plaintext is returned exactly once.
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.enterprise_models import ApiKey

_PREFIX_BYTES = 4   # -> 8 hex chars
_SECRET_BYTES = 32


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
        row = await self.db.scalar(
            select(ApiKey).where(ApiKey.prefix == prefix, ApiKey.revoked_at.is_(None))
        )
        if row is None or not secrets.compare_digest(row.hashed_secret, _hash(plaintext)):
            return None
        row.last_used_at = datetime.now(timezone.utc)
        return row

    async def revoke(self, key_id: uuid.UUID) -> ApiKey | None:
        row = await self.db.get(ApiKey, key_id)
        if row is not None and row.revoked_at is None:
            row.revoked_at = datetime.now(timezone.utc)
            await self.db.flush()
        return row

    async def revoke_for_service_account(self, service_account_id: uuid.UUID) -> None:
        rows = (
            await self.db.scalars(
                select(ApiKey).where(
                    ApiKey.service_account_id == service_account_id, ApiKey.revoked_at.is_(None)
                )
            )
        ).all()
        now = datetime.now(timezone.utc)
        for r in rows:
            r.revoked_at = now
        await self.db.flush()

    async def list_for_org(self, org_id: uuid.UUID) -> list[ApiKey]:
        return list(
            (await self.db.scalars(select(ApiKey).where(ApiKey.organization_id == org_id))).all()
        )
