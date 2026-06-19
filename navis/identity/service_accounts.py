"""Service accounts: machine identities (workers, integrations).

Org-scoped, permission-scoped, with rotatable credentials. Credentials are API
keys bound to the service account, so rotation = mint new key + revoke old.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.apikeys import ApiKeyService
from navis.identity.enterprise_models import ApiKey, ServiceAccount


class ServiceAccountService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.keys = ApiKeyService(db)

    async def create(
        self, *, org_id: uuid.UUID, name: str, scopes: list[str], created_by: uuid.UUID | None = None
    ) -> tuple[ServiceAccount, str]:
        sa = ServiceAccount(
            organization_id=org_id, name=name, scopes=list(scopes), created_by=created_by
        )
        self.db.add(sa)
        await self.db.flush()
        _, plaintext = await self.keys.create(
            org_id=org_id, name=f"{name} key", scopes=scopes,
            created_by=created_by, service_account_id=sa.id,
        )
        return sa, plaintext

    async def rotate(self, sa_id: uuid.UUID) -> str:
        sa = await self._require(sa_id)
        await self.keys.revoke_for_service_account(sa.id)
        _, plaintext = await self.keys.create(
            org_id=sa.organization_id, name=f"{sa.name} key (rotated)",
            scopes=sa.scopes, created_by=sa.created_by, service_account_id=sa.id,
        )
        return plaintext

    async def revoke(self, sa_id: uuid.UUID) -> ServiceAccount:
        sa = await self._require(sa_id)
        sa.revoked_at = datetime.now(timezone.utc)
        await self.keys.revoke_for_service_account(sa.id)
        await self.db.flush()
        return sa

    async def list_for_org(self, org_id: uuid.UUID) -> list[ServiceAccount]:
        return list(
            (await self.db.scalars(select(ServiceAccount).where(ServiceAccount.organization_id == org_id))).all()
        )

    async def active_keys(self, sa_id: uuid.UUID) -> list[ApiKey]:
        return list(
            (
                await self.db.scalars(
                    select(ApiKey).where(ApiKey.service_account_id == sa_id, ApiKey.revoked_at.is_(None))
                )
            ).all()
        )

    async def _require(self, sa_id: uuid.UUID) -> ServiceAccount:
        sa = await self.db.get(ServiceAccount, sa_id)
        if sa is None:
            raise LookupError("service account not found")
        return sa
