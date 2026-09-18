"""The Developer request context, and the one gate every Developer route passes.

WHY A CONTEXT OBJECT EXISTS AT ALL. `Principal` already answers four of the five
questions a Developer request must answer — organization, principal, permissions,
and (through `kind`/`subject_id`) whether the caller is a machine. It does not
answer the fifth in a usable shape: which SERVICE ACCOUNT and which CREDENTIAL
are acting, because `subject_id` deliberately collapses the two. `DeveloperContext`
is a *projection* of `Principal` that separates them. It resolves nothing, queries
nothing, and holds no authority `Principal` did not already have — if it ever
needs a rule of its own, that rule belongs in Identity instead.

ORGANIZATION IS NEVER AN INPUT. `DeveloperContext.of` reads `principal.org_id`
and nothing else, and there is no constructor that accepts an organization from
anywhere. For a JWT that value is `membership.organization_id` — the row the
database agreed to, not the `org` claim the token asked for. For an API key it is
`api_keys.organization_id`. Neither can be influenced by a header, a query
parameter or a body, and no Developer route declares a parameter that could.

THE GATE ORDER IS THE ERROR CONTRACT. Authentication (401) precedes authorization
(403) precedes entitlement (402), so a caller can always tell "I am not who you
think" from "I may not do this" from "this organization is not on a plan that
includes it". Reversing the last two would leak an organization's plan to a
credential that was never allowed to ask.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.authz import Principal, PrincipalKind, get_principal, require_permission
from navis.identity.permissions import Permission
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("developer.gateway")


@dataclass(frozen=True, slots=True)
class DeveloperContext:
    """Who is calling, on behalf of which organization, with what powers.

    Frozen: a handler may read the caller's identity but may never edit it, so a
    route cannot re-point itself at another tenant halfway through by assigning
    to `organization_id`. `slots=True` closes the same hole from the other side —
    an attribute this class does not declare cannot be attached at runtime.
    """

    principal: Principal
    organization_id: uuid.UUID
    kind: str
    #: The acting subject. A user id, a service-account id, or — for a bare API
    #: key with no service account behind it — the key's own id. Identical to
    #: `Principal.subject_id`; named here so the wire contract does not depend on
    #: an Identity field name.
    subject_id: uuid.UUID
    #: The machine identity acting, when there is one. `None` for a human session
    #: and for a bare API key: absence means "no service account is behind this
    #: call", which is a different fact from "the service account is unknown".
    service_account_id: uuid.UUID | None
    #: The `api_keys` row that authenticated this request. The attribution,
    #: rotation and revocation handle — see `Principal.credential_id`.
    credential_id: uuid.UUID | None
    permissions: frozenset[str]

    @classmethod
    def of(cls, principal: Principal) -> "DeveloperContext":
        # A SERVICE_ACCOUNT principal's `subject_id` IS the service account id
        # (`authz._principal_from_api_key` sets it from `key.service_account_id`).
        # A bare API_KEY principal has no account behind it, so the field is None
        # rather than a copy of the key id — reporting the key as an "agent"
        # would invent a machine identity the database never created.
        is_agent = principal.kind == PrincipalKind.SERVICE_ACCOUNT
        return cls(
            principal=principal,
            organization_id=principal.org_id,
            kind=str(principal.kind),
            subject_id=principal.subject_id,
            service_account_id=principal.subject_id if is_agent else None,
            credential_id=principal.credential_id,
            permissions=principal.permissions,
        )


def require_developer(permission: str, capability: Capability):
    """Authenticate, authorize, check entitlement, and hand back the context.

    Composes the EXISTING gates rather than reimplementing them: `_perm` is
    `identity.authz.require_permission` and `_cap` is
    `organization.guards.require_capability`, called as plain coroutines with the
    dependencies FastAPI already resolved. A second copy of either check is a
    second authorization decision that is free to drift from the one the web edge
    enforces, which is the whole failure this boundary exists to avoid.

    BOTH gates, always. The web-edge read routes this boundary delegates to
    (`/context`, `/knowledge/*`, `/outcomes`, `/learning/summary`) are gated on a
    CAPABILITY alone — an org-level entitlement, which answers "is this tenant on
    a plan that includes it", never "may this credential do it". An API key minted
    with `scopes=[]` therefore holds no permission at all and still passes them.
    Developer routes are machine-facing by definition, so every one of them names
    a permission as well; a scopeless key gets 403 here.
    """
    _perm = require_permission(permission)
    _cap = require_capability(capability)

    async def _dep(
        principal: Principal = Depends(get_principal),
        db: AsyncSession = Depends(get_session),
    ) -> DeveloperContext:
        await _perm(principal)      # 403 — this credential lacks the permission
        await _cap(principal, db)   # 402 — this organization lacks the capability
        return DeveloperContext.of(principal)

    return _dep


#: THE TWO GRANTS THAT REACH THE AGENT SURFACE (P1-7).
#:
#: `MANAGE_SERVICE_ACCOUNTS` is organization-wide agent administration and is
#: UNCHANGED — it means exactly what it meant before, so no existing credential
#: moved. `MANAGE_OWN_AGENTS` is the self-serve grant a Developer holds by
#: default: the same operations, bounded to agents they created.
#:
#: DEFINED HERE RATHER THAN IN `agents_router`, because TWO modules gate on it.
#: `/developer/permissions` serves the agent surface — it is the vocabulary the
#: console offers when registering one — so it has to admit exactly the callers
#: the agent routes admit. When it was gated on `manage_service_accounts` alone,
#: a self-serve developer passed `POST /developer/agents` and was handed an
#: empty permission catalog, so the form they were allowed to submit had nothing
#: to tick. One constant, so the two cannot drift apart again.
AGENT_MANAGEMENT_PERMISSIONS: tuple[str, ...] = (
    Permission.MANAGE_SERVICE_ACCOUNTS,
    Permission.MANAGE_OWN_AGENTS,
)


def require_developer_any(permissions: tuple[str, ...], capability: Capability):
    """As `require_developer`, but any ONE of `permissions` is enough (P1-7).

    WHY A DISJUNCTION EXISTS AT ALL, AND WHY IT IS NOT A SECOND AUTHORIZATION
    SYSTEM. The agent routes are reachable two ways that are genuinely different
    grants: `manage_service_accounts` (every agent in the organization) and
    `manage_own_agents` (only the ones you created). Both are real answers to
    "may you call this route", and the narrower one is the whole of P1-7 — an
    engineer must not need organization-wide machine-identity administration to
    register their own development agent.

    The alternative was to gate on the narrower permission alone and have the
    broader one imply it. That is smaller code and it would have been a
    CREDENTIAL REGRESSION: `manage_service_accounts` is agent-assignable and a
    live agent key holds it today, so re-gating the read routes on a permission
    that key does not carry would have 401'd a running integration. A gate that
    accepts either is the version with no blast radius.

    NOTHING NEW DECIDES ANYTHING HERE. The input is still
    `Principal.permissions`, resolved by the one `effective_permissions`; the
    entitlement half is still `organization.guards.require_capability`; the
    membership and organization were still settled by `get_principal`. This
    function asks the existing evaluator a question with an `or` in it, which is
    the only thing it adds.

    ORDER IS PRESERVED: 403 before 402, so a caller who may not do this at all
    is never told what the organization's plan includes.
    """
    _cap = require_capability(capability)

    async def _dep(
        principal: Principal = Depends(get_principal),
        db: AsyncSession = Depends(get_session),
    ) -> DeveloperContext:
        if not any(principal.has(permission) for permission in permissions):
            # Named in the SAME shape `require_permission` uses, and listing
            # every acceptable permission rather than one of them: a caller
            # told "missing permission: manage_service_accounts" would go and
            # ask for organization-wide agent administration when
            # `manage_own_agents` was all they needed.
            logger.warning(
                "permission denied: principal=%s kind=%s org=%s needs one of %s",
                principal.subject_id, principal.kind, principal.org_id,
                sorted(permissions),
            )
            metrics.incr(
                "auth.permission_denied",
                labels={"permission": "|".join(sorted(permissions))},
            )
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "missing permission: one of " + ", ".join(sorted(permissions)),
            )
        await _cap(principal, db)   # 402 — this organization lacks the capability
        return DeveloperContext.of(principal)

    return _dep


async def get_developer_context(
    principal: Principal = Depends(get_principal),
) -> DeveloperContext:
    """Authentication only — for the one route that must answer BEFORE authorizing.

    `/developer/me` reports what a credential is and what it may do. Gating it on
    a permission would make a correctly-issued but under-scoped credential unable
    to discover that fact, so the only requirement is that the caller
    authenticated. It publishes nothing a caller could not already infer from the
    403s it collects by trying every other route.
    """
    return DeveloperContext.of(principal)
