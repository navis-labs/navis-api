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
from navis.platform.monitoring import get_logger

logger = get_logger("identity.service_accounts")


class ServiceAccountService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.keys = ApiKeyService(db)

    async def create(
        self,
        *,
        org_id: uuid.UUID,
        name: str,
        scopes: list[str],
        created_by: uuid.UUID | None = None,
        expires_at: datetime | None = None,
    ) -> tuple[ServiceAccount, str]:
        """The original two-value contract, kept for `/identity/service-accounts`.

        Delegates to `create_agent` so there is ONE creation path: an account made
        here and an account made through the Agents API are the same row written
        by the same code, and neither can acquire a property the other lacks.

        `expires_at` bounds the FIRST CREDENTIAL, not the account — see
        `create_agent`. Defaulting to None keeps this signature's existing
        callers producing exactly the key they produced before.
        """
        sa, _key, plaintext = await self.create_agent(
            org_id=org_id,
            name=name,
            scopes=scopes,
            created_by=created_by,
            expires_at=expires_at,
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
        #: Optional lifetime for the FIRST CREDENTIAL. None means it does not
        #: expire, which is what every credential minted before expiry existed
        #: is — so omitting it leaves this function's output unchanged.
        expires_at: datetime | None = None,
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

        `expires_at` BOUNDS THE CREDENTIAL, NEVER THE AGENT. `service_accounts`
        has no expiry column and must not grow one: an agent is the identity that
        outlives its secrets — it is what keeps historical runs attributed after
        a credential is rotated out — so a lifetime belongs on the thing you
        replace, not on the thing you replace it for. Validation (an instant
        already in the past is refused) lives in `ApiKeyService.create`, the one
        floor under every minting path.
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
            expires_at=expires_at,
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

    async def rotate(
        self,
        sa_id: uuid.UUID,
        *,
        org_id: uuid.UUID,
        #: Lifetime for the REPLACEMENT credential. None means it does not
        #: expire, which is what every rotation produced before this parameter
        #: existed — see the note below on why that is not inherited.
        expires_at: datetime | None = None,
    ) -> str:
        """Revoke every credential of this account and mint one replacement.

        ROTATION DOES NOT INHERIT THE OLD EXPIRY, AND THE CALLER MUST SAY SO.
        An account can hold several credentials with different lifetimes — one
        never-expiring, one dated — so there is no single expiry to carry over:
        picking the longest would quietly extend a bound somebody chose, and
        picking the shortest would shorten one. Both are the service guessing at
        an operator's intent about a credential that is about to be the only one
        that works.

        The consequence is worth stating plainly, because it is a downgrade the
        caller can miss: rotating a dated credential without passing
        `expires_at` replaces it with one that never expires. That was the only
        behaviour available before, so it is not a regression — but it is no
        longer the only option, and the log line below records it when it
        happens so an audit can see where a bound was dropped.
        """
        sa = await self._require(sa_id, org_id)
        if expires_at is None:
            # Read BEFORE the revocation below, which does not clear `expires_at`
            # but does make "which keys were live" unanswerable from the rows.
            dated = [k for k in await self.active_keys(sa.id) if k.expires_at is not None]
            if dated:
                logger.info(
                    "rotation dropped credential expiry: service_account=%s org=%s "
                    "dated_credentials=%d replacement=never-expires",
                    sa.id, sa.organization_id, len(dated),
                )
        await self.keys.revoke_for_service_account(sa.id)
        _, plaintext = await self.keys.create(
            org_id=sa.organization_id, name=f"{sa.name} key (rotated)",
            scopes=sa.scopes, created_by=sa.created_by, service_account_id=sa.id,
            expires_at=expires_at,
        )
        return plaintext

    async def issue_credential(
        self,
        sa_id: uuid.UUID,
        *,
        org_id: uuid.UUID,
        name: str,
        created_by: uuid.UUID | None = None,
        #: Optional lifetime for THIS credential. None means it does not
        #: expire, matching every credential minted before expiry existed.
        expires_at: datetime | None = None,
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
            expires_at=expires_at,
        )

    async def revoke(self, sa_id: uuid.UUID, *, org_id: uuid.UUID) -> ServiceAccount:
        sa = await self._require(sa_id, org_id)
        sa.revoked_at = datetime.now(timezone.utc)
        await self.keys.revoke_for_service_account(sa.id)
        await self.db.flush()
        return sa

    async def list_for_org(
        self,
        org_id: uuid.UUID,
        *,
        limit: int = 50,
        offset: int = 0,
        created_by: uuid.UUID | None = None,
    ) -> list[ServiceAccount]:
        """This tenant's agents, optionally only the ones one person created.

        `created_by=None` means NO CREATOR FILTER — the whole organization's
        fleet, which is what every caller before P1-7 asked for and still gets.
        It deliberately does NOT mean "rows whose creator is NULL": the filter
        is applied only when a creator is named, so an organization-wide caller
        keeps seeing agents registered before the column was populated.

        Passing a creator is how the self-serve Developer scope is expressed
        (`agents_router._sees_every_agent`). The tenant predicate is
        unconditional either way — the creator filter narrows, never widens.
        """
        stmt = select(ServiceAccount).where(ServiceAccount.organization_id == org_id)
        if created_by is not None:
            stmt = stmt.where(ServiceAccount.created_by == created_by)
        return list(
            (
                await self.db.scalars(
                    stmt.order_by(ServiceAccount.created_at.desc(), ServiceAccount.id)
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
        )

    async def count_for_org(
        self,
        org_id: uuid.UUID,
        *,
        active_only: bool = False,
        created_by: uuid.UUID | None = None,
    ) -> int:
        """How many accounts this tenant holds. Backs pagination and the ceiling.

        `active_only` counts what may still authenticate. The creation ceiling
        uses it so revoking an agent frees its slot — a limit that counted
        tombstones would make an organization that rotates its fleet
        permanently unable to create another.

        `created_by` narrows the count to one person's agents, so a paged list
        reports a `total` that matches what the caller can actually see. IT IS
        NOT USED FOR THE CEILING, deliberately: the agent limit is a property of
        the ORGANIZATION, and counting per-creator would let a team of ten
        quietly hold ten times the agents the tenant is provisioned for.
        """
        stmt = select(func.count()).select_from(ServiceAccount).where(
            ServiceAccount.organization_id == org_id
        )
        if active_only:
            stmt = stmt.where(ServiceAccount.revoked_at.is_(None))
        if created_by is not None:
            stmt = stmt.where(ServiceAccount.created_by == created_by)
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
