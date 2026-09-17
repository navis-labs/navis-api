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
from enum import StrEnum

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.enterprise_models import RolePermission
from navis.identity.models import Membership, User
from navis.identity.permissions import effective_permissions
from navis.identity.tokens import TokenRevocationService, token_predates_bulk_revoke
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.logging import set_org_id
from navis.platform.monitoring.metrics import metrics
from navis.platform.security import TokenError, decode_token

logger = get_logger("identity.authz")
_bearer = HTTPBearer(auto_error=False)


class PrincipalKind(StrEnum):
    """Closed set of actor kinds. StrEnum members ARE strings, so existing
    comparisons/serialization (`principal.kind == "user"`) keep working while
    callers get typo safety and a single source of truth."""

    USER = "user"
    API_KEY = "api_key"
    SERVICE_ACCOUNT = "service_account"


@dataclass(slots=True)
class Principal:
    kind: PrincipalKind            # user | api_key | service_account
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
        # Enterprise forensics: who tried to reach which org with a valid token but
        # no membership (a probe / stale access). Cheap log + metric — never a DB row
        # (an audit row per forged request would be a DoS-amplification vector).
        logger.warning("auth denied: user=%s no membership in org=%s", user_id, org_id)
        metrics.incr("auth.rejected", labels={"reason": "no_membership"})
        raise HTTPException(status.HTTP_403_FORBIDDEN, "no active org membership")

    role = membership.role.value if membership.role else None
    grants = await _grants_for(db, membership.organization_id, role or "")
    return Principal(
        kind=PrincipalKind.USER,
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
    kind = PrincipalKind.API_KEY
    if key.service_account_id is not None:
        sa = await db.get(ServiceAccount, key.service_account_id)
        if sa is None or sa.revoked_at is not None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "service account revoked")
        kind = PrincipalKind.SERVICE_ACCOUNT
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
    try:
        # Ambiguous intent: an X-API-Key AND a bearer token in the same request. We
        # refuse to silently pick one credential to authenticate as — reject so the
        # caller fixes the request rather than getting a surprising identity.
        if x_api_key and creds:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "provide either an API key or a bearer token, not both",
            )
        if x_api_key:
            principal = await _principal_from_api_key(db, x_api_key)
        elif creds and creds.scheme.lower() == "bearer":
            # A navis_ prefixed bearer is an API key; otherwise a JWT.
            if creds.credentials.startswith("navis_"):
                principal = await _principal_from_api_key(db, creds.credentials)
            else:
                principal = await _principal_from_jwt(db, creds.credentials)
        else:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing credentials")
        # Bind the authenticated org_id to the logging context so every downstream log
        # line (and every DomainEvent) automatically carries the tenant without callers
        # doing anything. Cleared automatically when the request context is torn down.
        set_org_id(str(principal.org_id))
        return principal
    except HTTPException as exc:
        # One place that counts every auth rejection (401 invalid/revoked vs 403).
        metrics.incr("auth.rejected", labels={"status": str(exc.status_code)})
        raise


def require_permission(permission: str):
    """FastAPI dependency factory: 403 unless the principal holds `permission`."""

    async def _dep(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.has(permission):
            # Cheap SOC signal: an authenticated principal hitting a wall. A metric
            # (not a DB audit row) — one row per forged request would be a DoS-
            # amplification vector, same reasoning as the no-membership path above.
            logger.warning(
                "permission denied: principal=%s kind=%s org=%s perm=%s",
                principal.subject_id, principal.kind, principal.org_id, permission,
            )
            metrics.incr("auth.permission_denied", labels={"permission": permission})
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, f"missing permission: {permission}"
            )
        return principal

    return _dep
