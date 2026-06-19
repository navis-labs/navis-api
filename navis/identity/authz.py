"""Authorization: resolve a Principal (human or machine) and gate on permissions.

A Principal is the unified actor — a user (JWT) or a machine (API key / service
account). Every authenticated request resolves one, and `require_permission(perm)`
asserts the principal holds `perm`. No route should gate on membership/role alone.

Token revocation (jti denylist + bulk-revoke watermark) is enforced here, so a
revoked access token cannot resolve a Principal.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.enterprise_models import RolePermission
from navis.identity.models import Membership, User
from navis.identity.permissions import effective_permissions
from navis.identity.tokens import TokenRevocationService, token_predates_bulk_revoke
from navis.platform.database import get_session
from navis.platform.security import TokenError, decode_token

_bearer = HTTPBearer(auto_error=False)


@dataclass(slots=True)
class Principal:
    kind: str                      # "user" | "api_key" | "service_account"
    org_id: uuid.UUID
    subject_id: uuid.UUID          # user id, or api-key/service-account id
    permissions: frozenset[str]
    role: str | None = None
    user: User | None = None
    scopes: list[str] = field(default_factory=list)

    def has(self, permission: str) -> bool:
        return permission in self.permissions


async def _grants_for(db: AsyncSession, org_id: uuid.UUID, role: str) -> list[str]:
    rows = (
        await db.scalars(
            select(RolePermission).where(
                RolePermission.role == role,
                RolePermission.organization_id.in_([org_id, None]),
            )
        )
    ).all()
    return [r.permission for r in rows]


async def _principal_from_jwt(db: AsyncSession, token: str) -> Principal:
    try:
        payload = decode_token(token, expected_type="access")
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))

    user_id, org_id = payload.get("sub"), payload.get("org")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "malformed token")

    revocation = TokenRevocationService(db)
    if await revocation.is_jti_revoked(payload.get("jti")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token revoked")

    user = await db.get(User, uuid.UUID(user_id))
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found or inactive")
    if token_predates_bulk_revoke(payload, user):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token revoked")

    membership = None
    if org_id:
        membership = await db.scalar(
            select(Membership).where(
                Membership.user_id == user.id,
                Membership.organization_id == uuid.UUID(org_id),
            )
        )
    if membership is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "no active org membership")

    role = membership.role.value if membership.role else None
    grants = await _grants_for(db, membership.organization_id, role or "")
    return Principal(
        kind="user",
        org_id=membership.organization_id,
        subject_id=user.id,
        permissions=effective_permissions(role, grants),
        role=role,
        user=user,
    )


async def _principal_from_api_key(db: AsyncSession, plaintext: str) -> Principal:
    from navis.identity.apikeys import ApiKeyService
    from navis.identity.enterprise_models import ServiceAccount

    key = await ApiKeyService(db).verify(plaintext)
    if key is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid api key")
    kind = "api_key"
    if key.service_account_id is not None:
        sa = await db.get(ServiceAccount, key.service_account_id)
        if sa is None or sa.revoked_at is not None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "service account revoked")
        kind = "service_account"
    scopes = list(key.scopes or [])
    return Principal(
        kind=kind,
        org_id=key.organization_id,
        subject_id=key.service_account_id or key.id,
        permissions=frozenset(scopes),
        scopes=scopes,
    )


async def get_principal(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_session),
) -> Principal:
    if x_api_key:
        return await _principal_from_api_key(db, x_api_key)
    if creds and creds.scheme.lower() == "bearer":
        # A navis_ prefixed bearer is an API key; otherwise a JWT.
        if creds.credentials.startswith("navis_"):
            return await _principal_from_api_key(db, creds.credentials)
        return await _principal_from_jwt(db, creds.credentials)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing credentials")


def require_permission(permission: str):
    """FastAPI dependency factory: 403 unless the principal holds `permission`."""

    async def _dep(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.has(permission):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, f"missing permission: {permission}"
            )
        return principal

    return _dep
