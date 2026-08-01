"""Authorization: resolve a Principal (human or machine) and gate on permissions.

A Principal is the unified actor — a user (JWT) or a machine (API key / service
account). Every authenticated request resolves one, and `require_permission(perm)`
asserts the principal holds `perm`. No route should gate on membership/role alone.

Token revocation (jti denylist + bulk-revoke watermark) is enforced here, so a
revoked access token cannot resolve a Principal.
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
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
from navis.platform.monitoring import get_logger, set_org_id
from navis.platform.monitoring.metrics import metrics
from navis.platform.security import TokenError, decode_token

logger = get_logger("identity.authz")
_bearer = HTTPBearer(auto_error=False)

# ── Principal cache (JWT auth hot path) ─────────────────────────────────────
# Every authenticated request currently pays 4 sequential Neon round-trips
# under `_principal_from_jwt`: jti-denylist, user, membership, grants. On a
# serverless Postgres those alone dominate the /context tail latency
# (measured ~1.5-2s per request before this cache).
#
# The cache stores the resolved Principal (already-validated: not revoked,
# user active, membership present, grants materialized) keyed by the JWT
# `jti` — one entry per issued access token. Bounded LRU + TTL, both
# short-window so revocations propagate within the TTL. Explicit
# invalidation hooks in TokenRevocationService.revoke_jti /
# revoke_all_for_user shorten that window to zero on logout / bulk-revoke.
#
# Security posture (why 30s is defensible):
#   * JWT expiry is still checked every request (decode_token runs before
#     the cache lookup) — an expired token can NEVER hit the cache.
#   * revoke_jti evicts the entry synchronously → logout is immediate.
#   * revoke_all_for_user evicts every entry for that user → bulk revoke
#     (deactivation, "log out everywhere") is immediate.
#   * The remaining window (30s) covers passive role/grant changes and
#     the tokens_valid_from watermark being bumped by out-of-band code
#     paths. This is the same trade-off AWS STS credential caching uses.
#
# Cached User objects are session-detached (db.expunge) before caching so
# downstream column reads work but no lazy load can trigger against a
# closed session.
_PRINCIPAL_CACHE_MAX = 1024
_PRINCIPAL_CACHE_TTL_S = 30.0
_PRINCIPAL_CACHE: "OrderedDict[str, tuple[float, uuid.UUID, Principal]]" = OrderedDict()


def _principal_cache_get(jti: str | None) -> "Principal | None":
    if not jti:
        return None
    entry = _PRINCIPAL_CACHE.get(jti)
    if entry is None:
        return None
    expires_at, _user_id, principal = entry
    if time.monotonic() >= expires_at:
        # Expired — evict lazily; next request pays the DB cost.
        _PRINCIPAL_CACHE.pop(jti, None)
        return None
    _PRINCIPAL_CACHE.move_to_end(jti)  # LRU touch
    return principal


def _principal_cache_put(jti: str | None, principal: "Principal") -> None:
    if not jti:
        return
    expires_at = time.monotonic() + _PRINCIPAL_CACHE_TTL_S
    # Track the user_id alongside the entry so evict_user can drop all
    # tokens for a user without walking their JWT payloads.
    user_id = principal.subject_id if principal.kind == PrincipalKind.USER else uuid.UUID(int=0)
    _PRINCIPAL_CACHE[jti] = (expires_at, user_id, principal)
    _PRINCIPAL_CACHE.move_to_end(jti)
    while len(_PRINCIPAL_CACHE) > _PRINCIPAL_CACHE_MAX:
        _PRINCIPAL_CACHE.popitem(last=False)


def principal_cache_evict_jti(jti: str | None) -> None:
    """Called by TokenRevocationService.revoke_jti — logout must propagate
    immediately, not on the 30s TTL boundary."""
    if not jti:
        return
    if _PRINCIPAL_CACHE.pop(jti, None) is not None:
        metrics.incr("auth.cache.evict", labels={"reason": "revoke_jti"})


def principal_cache_evict_user(user_id: uuid.UUID) -> None:
    """Called by TokenRevocationService.revoke_all_for_user — deactivation,
    membership removal, and 'log out everywhere' MUST invalidate every
    outstanding session for that user, not just the one presenting a jti."""
    victims = [k for k, (_exp, uid, _p) in _PRINCIPAL_CACHE.items() if uid == user_id]
    for k in victims:
        _PRINCIPAL_CACHE.pop(k, None)
    if victims:
        metrics.incr(
            "auth.cache.evict", value=len(victims), labels={"reason": "revoke_user"}
        )


def principal_cache_clear() -> None:
    """Public-in-tests helper. Only auth-cache tests should call this."""
    _PRINCIPAL_CACHE.clear()


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

    # Fast path — decoded JWT is valid, its jti has been seen recently, and
    # the resolved Principal is still within its short TTL. Skip the 4 DB
    # round-trips (revocation, user, membership, grants). Correctness: token
    # expiry was just re-verified by decode_token; revoke_jti /
    # revoke_all_for_user evict this cache synchronously on logout /
    # deactivation, so the residual staleness window is bounded to
    # _PRINCIPAL_CACHE_TTL_S for slow-path invariants (passive role edits,
    # bulk-revoke watermark bumps by out-of-band code paths).
    jti = payload.get("jti")
    cached = _principal_cache_get(jti)
    if cached is not None:
        metrics.incr("auth.cache.hit")
        return cached
    metrics.incr("auth.cache.miss")

    revocation = TokenRevocationService(db)
    if await revocation.is_jti_revoked(jti):
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

    # Detach the User row from this request's session before caching so a
    # future cache hit — served under a DIFFERENT session — can safely read
    # its column attributes without accidentally lazy-loading against a
    # closed session. Detach is best-effort: it only makes sense when the
    # instance is in this session; if it isn't, expunge is a no-op we ignore.
    try:
        db.sync_session.expunge(user)
    except Exception:
        pass

    principal = Principal(
        kind=PrincipalKind.USER,
        org_id=membership.organization_id,
        subject_id=user.id,
        permissions=effective_permissions(role, grants),
        role=role,
        user=user,
    )
    _principal_cache_put(jti, principal)
    return principal


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
        # Single choke point: bind the authenticated org to the logging context so EVERY
        # log line emitted for the rest of this request (and any sync work on this task)
        # carries org_id. Runs only after auth succeeds; rejections never set it.
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
