"""Service accounts: machine identities (workers, integrations) — and AGENTS.

Org-scoped, permission-scoped, with rotatable credentials. Credentials are API
keys bound to the service account, so rotation = mint new key + revoke old.

A SERVICE ACCOUNT IS WHAT `/api/v1/developer/agents` CALLS AN AGENT. The product
word and the table name differ; the row does not. See `navis/identity/agents.py`
for why there is no second table, and `navis/developer/agents_router.py` for the
surface built on this service. Everything below is shared by both edges, so an
agent and a worker are revoked, rotated and scoped by exactly one implementation.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.agents import AgentEnvironment, AgentRuntime
from navis.identity.apikeys import ApiKeyService
from navis.identity.enterprise_models import ApiKey, ServiceAccount


class ServiceAccountService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.keys = ApiKeyService(db)

    async def create(
        self, *, org_id: uuid.UUID, name: str, scopes: list[str], created_by: uuid.UUID | None = None
    ) -> tuple[ServiceAccount, str]:
        """The original two-value contract, kept for `/identity/service-accounts`.

        Delegates to `create_agent` so there is ONE creation path: an account made
        here and an account made through the Agents API are the same row written
        by the same code, and neither can acquire a property the other lacks.
        """
        sa, _key, plaintext = await self.create_agent(
            org_id=org_id, name=name, scopes=scopes, created_by=created_by
        )
        return sa, plaintext

    async def create_agent(
        self,
        *,
        org_id: uuid.UUID,
        name: str,
        scopes: list[str],
        created_by: uuid.UUID | None = None,
        description: str = "",
        environment: str = AgentEnvironment.DEVELOPMENT.value,
        runtime: str = AgentRuntime.CUSTOM.value,
        credential_name: str | None = None,
    ) -> tuple[ServiceAccount, ApiKey, str]:
        """Create the identity AND its first credential, in one transaction.

        Returns the key ROW as well as the plaintext. The row carries the public
        prefix and the credential id — the handles an operator needs to recognise
        and later revoke this credential — and they are unrecoverable afterwards
        from the plaintext alone, because nothing stores it.

        `org_id` is the CALLER'S resolved organization in every caller. This
        function takes it as an argument rather than resolving it because it is
        also reachable from a worker with no request behind it; the HTTP edges
        pass `Principal.org_id` and accept no organization from the wire.
        """
        sa = ServiceAccount(
            organization_id=org_id,
            name=name,
            scopes=list(scopes),
            created_by=created_by,
            description=description,
            environment=environment,
            runtime=runtime,
        )
        self.db.add(sa)
        await self.db.flush()
        key, plaintext = await self.keys.create(
            org_id=org_id,
            name=credential_name or f"{name} key",
            scopes=scopes,
            created_by=created_by,
            service_account_id=sa.id,
        )
        return sa, key, plaintext

    async def get(self, sa_id: uuid.UUID, *, org_id: uuid.UUID) -> ServiceAccount | None:
        """Org-scoped read. A row in another tenant answers None, never the row."""
        sa = await self.db.get(ServiceAccount, sa_id)
        if sa is None or sa.organization_id != org_id:
            return None
        return sa

    async def update(
        self,
        sa_id: uuid.UUID,
        *,
        org_id: uuid.UUID,
        name: str | None = None,
        description: str | None = None,
        environment: str | None = None,
        runtime: str | None = None,
        scopes: list[str] | None = None,
    ) -> ServiceAccount:
        """Edit the descriptive fields, and — separately — the grants.

        SCOPES CASCADE TO THE LIVE CREDENTIALS, AND THEY MUST.
        `authz._principal_from_api_key` authorizes from `api_keys.scopes`, not
        from the account's list: the account's copy is the template a key is
        minted from. Editing only the account would therefore change what the
        console displays and nothing about what the credential in an agent's
        environment can actually reach — a permission screen that lies. The
        UPDATE below runs in the same transaction as the account edit, so the
        two copies are never observably different.

        Per-credential narrowing is deliberately NOT a feature. Every active key
        of an account carries the account's scopes, and a future narrowing
        contract would have to decide what a cascade means for a key that was
        intentionally given less — a decision no caller needs today.
        """
        sa = await self._require(sa_id, org_id)
        if name is not None:
            sa.name = name
        if description is not None:
            sa.description = description
        if environment is not None:
            sa.environment = environment
        if runtime is not None:
            sa.runtime = runtime
        if scopes is not None:
            sa.scopes = list(scopes)
            await self.db.execute(
                update(ApiKey)
                .where(
                    ApiKey.service_account_id == sa.id,
                    ApiKey.revoked_at.is_(None),
                )
                .values(scopes=list(scopes))
            )
        await self.db.flush()
        return sa

    async def rotate(self, sa_id: uuid.UUID, *, org_id: uuid.UUID) -> str:
        sa = await self._require(sa_id, org_id)
        await self.keys.revoke_for_service_account(sa.id)
        _, plaintext = await self.keys.create(
            org_id=sa.organization_id, name=f"{sa.name} key (rotated)",
            scopes=sa.scopes, created_by=sa.created_by, service_account_id=sa.id,
        )
        return plaintext

    async def issue_credential(
        self, sa_id: uuid.UUID, *, org_id: uuid.UUID, name: str, created_by: uuid.UUID | None = None
    ) -> tuple[ApiKey, str]:
        """Mint an ADDITIONAL credential without retiring the existing ones.

        Distinct from `rotate`, which revokes every key first. Both exist because
        they answer different operational questions: rotate is "this secret
        leaked, kill it", issue is "the same agent now also runs in CI". Scopes
        come from the ACCOUNT, never from the request, so a second credential can
        never be broader than the agent it belongs to.

        Refuses on a revoked account: minting a working credential for an
        identity that authentication already rejects would produce a key that is
        valid at the `api_keys` row and dead at the gate.
        """
        sa = await self._require(sa_id, org_id)
        if sa.revoked_at is not None:
            raise LookupError("service account not found")
        return await self.keys.create(
            org_id=sa.organization_id,
            name=name,
            scopes=list(sa.scopes or []),
            created_by=created_by,
            service_account_id=sa.id,
        )

    async def revoke(self, sa_id: uuid.UUID, *, org_id: uuid.UUID) -> ServiceAccount:
        sa = await self._require(sa_id, org_id)
        sa.revoked_at = datetime.now(timezone.utc)
        await self.keys.revoke_for_service_account(sa.id)
        await self.db.flush()
        return sa

    async def list_for_org(
        self, org_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> list[ServiceAccount]:
        return list(
            (
                await self.db.scalars(
                    select(ServiceAccount)
                    .where(ServiceAccount.organization_id == org_id)
                    .order_by(ServiceAccount.created_at.desc(), ServiceAccount.id)
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        )

    async def count_for_org(self, org_id: uuid.UUID, *, active_only: bool = False) -> int:
        """How many accounts this tenant holds. Backs pagination and the ceiling.

        `active_only` counts what may still authenticate. The creation ceiling
        uses it so revoking an agent frees its slot — a limit that counted
        tombstones would make an organization that rotates its fleet
        permanently unable to create another.
        """
        stmt = select(func.count()).select_from(ServiceAccount).where(
            ServiceAccount.organization_id == org_id
        )
        if active_only:
            stmt = stmt.where(ServiceAccount.revoked_at.is_(None))
        return int(await self.db.scalar(stmt) or 0)

    async def active_keys(self, sa_id: uuid.UUID) -> list[ApiKey]:
        return list(
            (
                await self.db.scalars(
                    select(ApiKey).where(ApiKey.service_account_id == sa_id, ApiKey.revoked_at.is_(None))
                )
            ).all()
        )

    async def credentials(
        self, sa_id: uuid.UUID, *, org_id: uuid.UUID, limit: int = 50, offset: int = 0
    ) -> list[ApiKey]:
        """Every credential of one account, revoked ones included.

        BOTH predicates, deliberately. Filtering on `service_account_id` alone
        would be correct today — a key and its account are pinned to one tenant
        by `authz`'s mismatch refusal — but this is a tenant-scoped read and the
        organization it is scoped to belongs in the query, not in an argument
        about why the query does not need it.
        """
        return list(
            (
                await self.db.scalars(
                    select(ApiKey)
                    .where(
                        ApiKey.service_account_id == sa_id,
                        ApiKey.organization_id == org_id,
                    )
                    .order_by(ApiKey.created_at.desc(), ApiKey.id)
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        )

    async def _require(self, sa_id: uuid.UUID, org_id: uuid.UUID) -> ServiceAccount:
        # Org-scoped: a SA from another tenant is indistinguishable from "not found".
        sa = await self.db.get(ServiceAccount, sa_id)
        if sa is None or sa.organization_id != org_id:
            raise LookupError("service account not found")
        return sa
