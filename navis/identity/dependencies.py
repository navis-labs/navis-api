"""Legacy auth dependency: `CurrentContext` is a thin USER-ONLY adapter over the
single auth engine (`authz.get_principal`).

There is exactly ONE place that decodes/validates a token and resolves a principal:
`authz.get_principal`. This module used to duplicate that logic — token decode,
jti-denylist + bulk-revoke checks, user load, membership lookup — which is how auth
bugs are born (fix one path, forget the other; the same token then works on one
route and fails on another). That duplication is gone.

New routes should depend on `Principal` directly (it carries permissions/kind/
subject_id). `CurrentContext` remains only for routes that predate `Principal`.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status

from navis.identity.authz import Principal, get_principal
from navis.identity.models import User


@dataclass(slots=True)
class CurrentContext:
    user: User
    org_id: uuid.UUID
    role: str


async def get_current_context(
    principal: Principal = Depends(get_principal),
) -> CurrentContext:
    """Resolve the active human (JWT) context via the single auth engine.

    Machine principals (API key / service account) are rejected here — legacy
    `CurrentContext` routes are human-only; machine callers must use `Principal`.
    Revocation, membership, and tenant checks all happen inside `get_principal`.
    """
    if principal.kind != "user" or principal.user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user credentials required")
    return CurrentContext(
        user=principal.user,
        org_id=principal.org_id,
        role=principal.role or "",  # never assume role is non-null
    )
