"""The Agent lifecycle — `/api/v1/developer/agents`.

WHAT AN AGENT IS, IN THIS SYSTEM. A row in `service_accounts`, with a credential
in `api_keys` bound to it. Nothing here creates a new kind of identity, a new
tenant resolver or a new authorization system: every handler resolves the caller
through `identity.authz.get_principal`, takes the tenant from `Principal.org_id`,
gates on the EXISTING `Permission` vocabulary, and writes through the EXISTING
`ServiceAccountService`. An agent registered here is authenticated by exactly the
code path that already authenticates a service account — see
`authz._principal_from_api_key`, including its credential/agent tenant-mismatch
refusal — which is the whole reason this is a view over that table and not a new
one. `navis/identity/agents.py` records why.

AN AGENT IS NOT "A USER PLUS AN API KEY". It has its own row, its own name, its
own grants and its own revocation clock, and it survives the departure of the
person who created it (`created_by` is attribution, never authority). Equally, it
is not an administrator: `agent.role` does not exist, no code path treats a
service-account principal as privileged, and the ONLY thing a credential can
reach is what its scope list names.

THE CHAIN, FOR EVERY ROUTE HERE:

    credential -> get_principal -> Principal(org from membership/api_keys row)
               -> require_permission(manage_service_accounts)   (403)
               -> require_capability(STATE)                     (402)
               -> rate_limit                                    (429)
               -> [writes only] human principal                 (403)
               -> ServiceAccountService, scoped to ctx.organization_id

WRITES ARE HUMAN-ONLY, AND THAT IS A DELIBERATE NARROWING. `POST
/identity/service-accounts` lets any principal holding
`manage_service_accounts` create one, machine included. That is defensible for
the admin surface and wrong for this one: an agent that can mint agents can
create identities that OUTLIVE its own revocation, so containing a leaked agent
credential would no longer be a single `DELETE`. The narrowing applies only to
this new surface; the existing route is unchanged, because changing it would
revoke a capability someone may already depend on.

NO ROUTE TAKES AN ORGANIZATION. There is no `organization_id` in any path, query,
header or body on this module — `CreateAgentRequest` and `UpdateAgentRequest`
have no such field, and `agent_id` is looked up UNDER `ctx.organization_id`, so
naming another tenant's agent id answers 404 rather than reaching its row.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.developer.agent_schemas import (
    AgentCredentialOut,
    AgentCredentialPageOut,
    AgentOut,
    AgentPageOut,
    AgentRevokedOut,
    CreateAgentCredentialRequest,
    CreateAgentRequest,
    CreatedAgentCredentialOut,
    CreatedAgentOut,
    CredentialRevokedOut,
    UpdateAgentRequest,
)
from navis.developer.gateway import DeveloperContext, require_developer
from navis.developer.schemas import DEVELOPER_ERRORS_GATED, DeveloperErrorOut
from navis.identity.agents import status_of
from navis.identity.audit import AuditAction, AuditService
from navis.identity.authz import PrincipalKind, ensure_grantable_scopes
from navis.identity.enterprise_models import ApiKey, ServiceAccount
from navis.identity.permissions import Permission
from navis.identity.service_accounts import ServiceAccountService
from navis.identity.throttle import rate_limit
from navis.organization.contract import Capability
from navis.platform.config import settings
from navis.platform.database import get_session
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger

logger = get_logger("developer.agents")

# Tagged `developer`, the same tag the read routes carry, because this IS the
# Developer boundary — one tag, one section of the OpenAPI document, one contract
# for a generated client to consume. A `developer-agents` tag would split the
# boundary in two everywhere it is rendered, and `test_architecture` pins the
# single tag for exactly that reason.
router = APIRouter(prefix="/developer/agents", tags=["developer"])

# Reads share the "read" bucket with every other Developer route, for the reason
# documented there: one per-organization budget, so a tenant's agent fleet cannot
# spend capacity its humans are not counted against.
_READ_LIMIT = Depends(rate_limit("read", "rate_limit_read_per_window"))
# Writes get their own, much smaller bucket. Creating machine identities is rare
# and durable; a burst is a runaway script or an attempt to fill the table.
_WRITE_LIMIT = Depends(rate_limit("agent_write", "rate_limit_agent_write_per_window"))

# WHY `Capability.STATE` GATES AGENT MANAGEMENT.
#
# There is no `Capability.AGENTS` and inventing one would be inventing a
# commercial policy at an API edge. STATE is the honest existing answer for two
# reasons. It is what an agent EXISTS to reach — an organization not entitled to
# its own state projection has nothing for an agent to read — and it is included
# from FREE upward, so this gate does not tier agent management by plan. What it
# does still enforce, through `EntitlementService._decide`, is LIFECYCLE: a
# SUSPENDED or CHURNED organization cannot mint new machine credentials, and a
# thrown kill switch stops it too. That is the property worth having here.
_AGENT_CAPABILITY = Capability.STATE

_AGENT_ERRORS: dict[int | str, dict] = {
    **DEVELOPER_ERRORS_GATED,
    404: {
        "model": DeveloperErrorOut,
        "description": (
            "No agent with that id in THIS organization. An agent id that exists "
            "in another tenant answers exactly this, and is indistinguishable "
            "from an id that exists nowhere."
        ),
    },
}

_AGENT_WRITE_ERRORS: dict[int | str, dict] = {
    **_AGENT_ERRORS,
    400: {"model": DeveloperErrorOut, "description": "An unknown permission was named."},
    409: {
        "model": DeveloperErrorOut,
        "description": "This organization is at its active-agent ceiling.",
    },
}

_MAX_PAGE = 200


async def _operator(
    ctx: DeveloperContext = Depends(
        require_developer(Permission.MANAGE_SERVICE_ACCOUNTS, _AGENT_CAPABILITY)
    ),
) -> DeveloperContext:
    """Authenticated, permitted, entitled — AND a person.

    See the module note: a machine principal that can register machines can
    create identities its own revocation does not reach. `PrincipalKind.USER` is
    only ever produced by `_principal_from_jwt`, which resolves the organization
    from a `memberships` row, so "is a user" and "is a member of this tenant" are
    the same assertion here rather than two that could drift.
    """
    if ctx.principal.kind != PrincipalKind.USER:
        logger.warning(
            "agent write refused for machine principal: kind=%s subject=%s org=%s",
            ctx.kind, ctx.subject_id, ctx.organization_id,
        )
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "agent lifecycle changes require a signed-in member, not a machine credential",
        )
    return ctx


#: Reads: the same gates minus the human requirement, so an agent may list the
#: organization's agents when its scopes say it may. Nothing it reads here is a
#: secret, and refusing would make an operator tool unable to use its own key.
_reader = require_developer(Permission.MANAGE_SERVICE_ACCOUNTS, _AGENT_CAPABILITY)


# ── Projection helpers ──────────────────────────────────────────────────────


async def _credential_facts(
    db: AsyncSession, agent_ids: list[uuid.UUID], org_id: uuid.UUID
) -> dict[uuid.UUID, tuple[int, datetime | None]]:
    """`{agent_id: (active_credential_count, last_used_at)}` in ONE query.

    Per-agent queries would make listing a fleet an N+1 against the credential
    table on every console page load. `last_used_at` is the MAX across the
    agent's credentials, including revoked ones: "when did this agent last act"
    is a question about the agent, and dropping the history of a rotated key
    would silently reset it.
    """
    if not agent_ids:
        return {}
    rows = (
        await db.execute(
            select(
                ApiKey.service_account_id,
                func.count().filter(ApiKey.revoked_at.is_(None)),
                func.max(ApiKey.last_used_at),
            )
            .where(
                ApiKey.service_account_id.in_(agent_ids),
                # Tenant predicate on the credential table too. Redundant given
                # the agents were already selected under this org, and cheap:
                # a filter that has to be argued about is a filter that will be
                # missing from the next query written from this one.
                ApiKey.organization_id == org_id,
            )
            .group_by(ApiKey.service_account_id)
        )
    ).all()
    return {row[0]: (int(row[1] or 0), row[2]) for row in rows}


def _agent_out(sa: ServiceAccount, active: int, last_used: datetime | None) -> AgentOut:
    return AgentOut(
        id=sa.id,
        organization_id=sa.organization_id,
        name=sa.name,
        description=sa.description or "",
        environment=sa.environment,
        runtime=sa.runtime,
        status=status_of(sa.revoked_at),
        permissions=sorted(sa.scopes or []),
        credential_count=active,
        last_used_at=last_used,
        created_at=sa.created_at,
        updated_at=sa.updated_at,
    )


def _credential_out(key: ApiKey) -> AgentCredentialOut:
    """Metadata only. `hashed_secret` is not read here and has no field to land in."""
    return AgentCredentialOut(
        id=key.id,
        # Non-null for every row this module returns: each is selected by
        # `service_account_id`, so the column cannot be NULL in the result.
        agent_id=key.service_account_id,  # type: ignore[arg-type]
        organization_id=key.organization_id,
        name=key.name,
        prefix=key.prefix,
        scopes=sorted(key.scopes or []),
        revoked=key.revoked_at is not None,
        last_used_at=key.last_used_at,
        created_at=key.created_at,
    )


async def _require_agent(
    db: AsyncSession, agent_id: uuid.UUID, org_id: uuid.UUID
) -> ServiceAccount:
    """The tenant-scoped lookup every single-agent route starts with.

    404 rather than 403 for a foreign id, deliberately: a 403 would confirm that
    the id names a real agent somewhere, which is a cross-tenant existence oracle.
    """
    sa = await ServiceAccountService(db).get(agent_id, org_id=org_id)
    if sa is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "agent not found")
    return sa


async def _emit(event_type: EventType, ctx: DeveloperContext, **payload) -> None:
    await event_bus.publish(
        DomainEvent(
            type=event_type,
            org_id=ctx.organization_id,
            source="developer.agents",
            payload={
                "actor_id": str(ctx.subject_id),
                **{k: str(v) for k, v in payload.items()},
            },
        )
    )


# ── Lifecycle ───────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=CreatedAgentOut,
    status_code=status.HTTP_201_CREATED,
    responses=_AGENT_WRITE_ERRORS,
    dependencies=[_WRITE_LIMIT],
    summary="Register an agent and mint its first credential",
)
async def create_agent(
    body: CreateAgentRequest,
    ctx: DeveloperContext = Depends(_operator),
    db: AsyncSession = Depends(get_session),
) -> CreatedAgentOut:
    """Create the identity and its first credential, in one transaction.

    THE ORGANIZATION IS `ctx.organization_id` AND NOTHING ELSE. `CreateAgentRequest`
    declares no organization field, so there is no value to ignore and no
    parameter a client could come to believe in.

    PERMISSIONS ARE BOUNDED BY THE CALLER'S OWN. `ensure_grantable_scopes` — the
    same function `/identity/api-keys` and `/identity/service-accounts` use —
    rejects an unknown permission with 400 and a permission the caller does not
    hold with 403. An operator therefore cannot mint an agent more capable than
    themselves, which is what stops agent creation being a privilege-escalation
    primitive.

    THE SECRET IS RETURNED ONCE, HERE. It is generated server-side by
    `ApiKeyService`, only its SHA-256 is stored, and no other route in this
    module returns a model that has a `secret` field. It is not logged: the lines
    below carry ids and the credential's public prefix.
    """
    ensure_grantable_scopes(body.permissions, ctx.principal)

    svc = ServiceAccountService(db)
    # The ceiling counts ACTIVE agents, so revoking one frees its slot. Checked
    # before the insert and inside the request's transaction; two simultaneous
    # creates at the boundary can both pass, which overshoots by one and is the
    # correct trade for a capacity guard — serializing every create behind a
    # tenant lock would cost more than the row it prevents.
    if await svc.count_for_org(ctx.organization_id, active_only=True) >= settings.max_agents_per_organization:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"this organization is at its limit of {settings.max_agents_per_organization} "
            "active agents; revoke one before creating another",
        )

    sa, key, plaintext = await svc.create_agent(
        org_id=ctx.organization_id,
        name=body.name,
        scopes=list(body.permissions),
        created_by=ctx.subject_id,
        description=body.description,
        environment=body.environment.value,
        runtime=body.runtime.value,
    )
    await AuditService(db).record(
        AuditAction.SERVICE_ACCOUNT_CREATE,
        org_id=ctx.organization_id,
        actor_id=ctx.subject_id,
        target_id=str(sa.id),
        # Facts, no secret. `AuditService._sanitize` would redact a key named
        # "secret" anyway; not putting one there is the better half of that.
        metadata={
            "surface": "developer.agents",
            "environment": sa.environment,
            "runtime": sa.runtime,
            "permissions": sorted(sa.scopes or []),
            "credential_id": str(key.id),
        },
    )
    await _emit(EventType.SERVICE_ACCOUNT_CREATED, ctx, service_account_id=sa.id)
    logger.info(
        "agent created id=%s org=%s credential=%s prefix=%s",
        sa.id, ctx.organization_id, key.id, key.prefix,
    )
    return CreatedAgentOut(
        agent=_agent_out(sa, active=1, last_used=None),
        credential=CreatedAgentCredentialOut(
            **_credential_out(key).model_dump(), secret=plaintext
        ),
    )


@router.get(
    "",
    response_model=AgentPageOut,
    responses=DEVELOPER_ERRORS_GATED,
    dependencies=[_READ_LIMIT],
    summary="This organization's agents",
)
async def list_agents(
    limit: int = Query(50, ge=1, le=_MAX_PAGE),
    offset: int = Query(0, ge=0),
    ctx: DeveloperContext = Depends(_reader),
    db: AsyncSession = Depends(get_session),
) -> AgentPageOut:
    """Paged from the first response, and served by a tenant-keyed index.

    PAGED BY DEFAULT, not once a customer notices. `/developer/state` is unpaged
    because it is one projection with a `record_count`; a fleet is a list that
    grows with usage, and an unpaged list read is a read that gets slower forever.
    `total` is reported alongside the page so a client can tell "50 agents" from
    "the first 50 of 900".

    `ix_service_accounts_org_created` matches this ORDER BY exactly, so one
    organization's fleet size never affects another's scan.
    """
    svc = ServiceAccountService(db)
    rows = await svc.list_for_org(ctx.organization_id, limit=limit, offset=offset)
    facts = await _credential_facts(db, [r.id for r in rows], ctx.organization_id)
    return AgentPageOut(
        agents=[_agent_out(r, *facts.get(r.id, (0, None))) for r in rows],
        count=len(rows),
        total=await svc.count_for_org(ctx.organization_id),
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{agent_id}",
    response_model=AgentOut,
    responses=_AGENT_ERRORS,
    dependencies=[_READ_LIMIT],
    summary="One agent",
)
async def get_agent(
    agent_id: uuid.UUID,
    ctx: DeveloperContext = Depends(_reader),
    db: AsyncSession = Depends(get_session),
) -> AgentOut:
    """Resolved under the caller's tenant. A foreign agent id is a 404."""
    sa = await _require_agent(db, agent_id, ctx.organization_id)
    facts = await _credential_facts(db, [sa.id], ctx.organization_id)
    return _agent_out(sa, *facts.get(sa.id, (0, None)))


@router.patch(
    "/{agent_id}",
    response_model=AgentOut,
    responses=_AGENT_WRITE_ERRORS,
    dependencies=[_WRITE_LIMIT],
    summary="Edit an agent, or change what it may reach",
)
async def update_agent(
    agent_id: uuid.UUID,
    body: UpdateAgentRequest,
    ctx: DeveloperContext = Depends(_operator),
    db: AsyncSession = Depends(get_session),
) -> AgentOut:
    """A partial edit. Omitted fields are untouched.

    CHANGING `permissions` CHANGES THE LIVE CREDENTIALS. Authorization reads
    `api_keys.scopes`, so `ServiceAccountService.update` cascades the new list to
    every unrevoked key of this agent in the same transaction. Without that, this
    endpoint would change what the console displays and nothing about what the
    credential in the agent's environment can actually reach.

    A REVOKED AGENT IS NOT EDITABLE. Revocation is terminal (see
    `identity.agents.AgentStatus`); editing a tombstone would suggest it can be
    brought back, and it cannot — its credentials are gone.
    """
    sa = await _require_agent(db, agent_id, ctx.organization_id)
    if sa.revoked_at is not None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "agent not found")
    if body.permissions is not None:
        ensure_grantable_scopes(body.permissions, ctx.principal)

    before = sorted(sa.scopes or [])
    sa = await ServiceAccountService(db).update(
        sa.id,
        org_id=ctx.organization_id,
        name=body.name,
        description=body.description,
        environment=body.environment.value if body.environment else None,
        runtime=body.runtime.value if body.runtime else None,
        scopes=list(body.permissions) if body.permissions is not None else None,
    )
    await AuditService(db).record(
        AuditAction.SERVICE_ACCOUNT_UPDATE,
        org_id=ctx.organization_id,
        actor_id=ctx.subject_id,
        target_id=str(sa.id),
        metadata={
            "surface": "developer.agents",
            # Both sides of a grant change, because "what did this become" is
            # only half of what an auditor needs to review the decision.
            "permissions_before": before,
            "permissions_after": sorted(sa.scopes or []),
            "fields": sorted(body.model_dump(exclude_none=True).keys()),
        },
    )
    facts = await _credential_facts(db, [sa.id], ctx.organization_id)
    return _agent_out(sa, *facts.get(sa.id, (0, None)))


@router.delete(
    "/{agent_id}",
    response_model=AgentRevokedOut,
    responses=_AGENT_ERRORS,
    dependencies=[_WRITE_LIMIT],
    summary="Revoke an agent and every credential it holds",
)
async def revoke_agent(
    agent_id: uuid.UUID,
    ctx: DeveloperContext = Depends(_operator),
    db: AsyncSession = Depends(get_session),
) -> AgentRevokedOut:
    """DELETE means REVOKE, not erase, and the distinction matters.

    The row stays, stamped with `revoked_at`, and every credential bound to it is
    stamped too — `ServiceAccountService.revoke` does both in one transaction.
    Authentication then refuses the credential twice over: the key is revoked, and
    `_principal_from_api_key` re-checks `service_accounts.revoked_at` and answers
    401 even if a key were somehow missed.

    Deleting the row instead would take the audit trail's target with it: the
    outcome rows, audit entries and log lines that name this agent id would point
    at nothing, exactly when someone is investigating why it was revoked.

    IDEMPOTENT. Revoking an already-revoked agent reports the same terminal state
    rather than 404 or 409 — a retry after a dropped response must not look like
    a different outcome.
    """
    sa = await _require_agent(db, agent_id, ctx.organization_id)
    svc = ServiceAccountService(db)
    active = len(await svc.active_keys(sa.id))
    if sa.revoked_at is None:
        await svc.revoke(sa.id, org_id=ctx.organization_id)
        await AuditService(db).record(
            AuditAction.SERVICE_ACCOUNT_REVOKE,
            org_id=ctx.organization_id,
            actor_id=ctx.subject_id,
            target_id=str(sa.id),
            metadata={"surface": "developer.agents", "credentials_revoked": active},
        )
        await _emit(EventType.SERVICE_ACCOUNT_REVOKED, ctx, service_account_id=sa.id)
        logger.info("agent revoked id=%s org=%s credentials=%d", sa.id, ctx.organization_id, active)
    else:
        active = 0
    return AgentRevokedOut(id=sa.id, status=status_of(sa.revoked_at), credentials_revoked=active)


# ── Credentials ─────────────────────────────────────────────────────────────


@router.get(
    "/{agent_id}/credentials",
    response_model=AgentCredentialPageOut,
    responses=_AGENT_ERRORS,
    dependencies=[_READ_LIMIT],
    summary="An agent's credentials — metadata only, never a secret",
)
async def list_agent_credentials(
    agent_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=_MAX_PAGE),
    offset: int = Query(0, ge=0),
    ctx: DeveloperContext = Depends(_reader),
    db: AsyncSession = Depends(get_session),
) -> AgentCredentialPageOut:
    """Revoked credentials are included, and should be.

    A credential list that hides revoked rows cannot answer "was the leaked key
    the one we retired?" — the question the list is opened for. `revoked` names
    each row's state instead.

    There is no secret in the response model, so this route has nothing to leak:
    `AgentCredentialOut` declares no such field and `hashed_secret` is never read.
    """
    sa = await _require_agent(db, agent_id, ctx.organization_id)
    rows = await ServiceAccountService(db).credentials(
        sa.id, org_id=ctx.organization_id, limit=limit, offset=offset
    )
    return AgentCredentialPageOut(
        credentials=[_credential_out(k) for k in rows],
        count=len(rows),
        limit=limit,
        offset=offset,
    )


@router.post(
    "/{agent_id}/credentials",
    response_model=CreatedAgentCredentialOut,
    status_code=status.HTTP_201_CREATED,
    responses=_AGENT_WRITE_ERRORS,
    dependencies=[_WRITE_LIMIT],
    summary="Mint an additional credential for an agent",
)
async def create_agent_credential(
    agent_id: uuid.UUID,
    body: CreateAgentCredentialRequest,
    ctx: DeveloperContext = Depends(_operator),
    db: AsyncSession = Depends(get_session),
) -> CreatedAgentCredentialOut:
    """A second key for the same agent — CI as well as a laptop.

    SCOPES COME FROM THE AGENT, NEVER THE REQUEST. `CreateAgentCredentialRequest`
    has no scopes field, so a credential is exactly as capable as the agent it
    belongs to and this route cannot be used to widen anything. The caller's own
    permissions were already checked when the agent was granted them.

    Distinct from rotation on purpose: this does not revoke the existing
    credentials. `POST /identity/service-accounts/{id}/rotate` is the "this
    secret leaked" operation and keeps its meaning.
    """
    sa = await _require_agent(db, agent_id, ctx.organization_id)
    try:
        key, plaintext = await ServiceAccountService(db).issue_credential(
            sa.id, org_id=ctx.organization_id, name=body.name, created_by=ctx.subject_id
        )
    except LookupError:
        # Raised for a revoked agent. Reported as 404 for the same reason the
        # cross-tenant lookup is: a revoked agent is not a resource that can
        # accept credentials, and saying so more precisely tells a caller
        # nothing they can act on.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "agent not found")

    await AuditService(db).record(
        AuditAction.API_KEY_CREATE,
        org_id=ctx.organization_id,
        actor_id=ctx.subject_id,
        target_id=str(key.id),
        metadata={"surface": "developer.agents", "agent_id": str(sa.id), "prefix": key.prefix},
    )
    await _emit(EventType.API_KEY_CREATED, ctx, key_id=key.id, service_account_id=sa.id)
    logger.info(
        "agent credential minted agent=%s org=%s credential=%s prefix=%s",
        sa.id, ctx.organization_id, key.id, key.prefix,
    )
    return CreatedAgentCredentialOut(**_credential_out(key).model_dump(), secret=plaintext)


@router.delete(
    "/{agent_id}/credentials/{credential_id}",
    response_model=CredentialRevokedOut,
    responses=_AGENT_ERRORS,
    dependencies=[_WRITE_LIMIT],
    summary="Revoke one credential, leaving the agent",
)
async def revoke_agent_credential(
    agent_id: uuid.UUID,
    credential_id: uuid.UUID,
    ctx: DeveloperContext = Depends(_operator),
    db: AsyncSession = Depends(get_session),
) -> CredentialRevokedOut:
    """Retire one key without retiring the identity.

    BOTH IDS ARE VERIFIED, in this order: the agent under the caller's tenant,
    then the credential under that agent. A credential id belonging to another
    agent — or another tenant — is a 404 either way, so neither parameter can be
    used to reach a row the caller could not already name.
    """
    sa = await _require_agent(db, agent_id, ctx.organization_id)
    key = await db.get(ApiKey, credential_id)
    if (
        key is None
        or key.organization_id != ctx.organization_id
        or key.service_account_id != sa.id
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "credential not found")

    if key.revoked_at is None:
        # Revoked through the org-scoped service rather than by assigning here,
        # so the one revocation implementation stays the one revocation
        # implementation.
        await ServiceAccountService(db).keys.revoke(key.id, org_id=ctx.organization_id)
        await AuditService(db).record(
            AuditAction.API_KEY_REVOKE,
            org_id=ctx.organization_id,
            actor_id=ctx.subject_id,
            target_id=str(key.id),
            metadata={"surface": "developer.agents", "agent_id": str(sa.id)},
        )
        await _emit(EventType.API_KEY_REVOKED, ctx, key_id=key.id, service_account_id=sa.id)
        logger.info(
            "agent credential revoked agent=%s org=%s credential=%s", sa.id, ctx.organization_id, key.id
        )
    return CredentialRevokedOut(id=key.id, agent_id=sa.id, revoked=True)
