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

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.authz import Principal, PrincipalKind, get_principal, require_permission
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session


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
