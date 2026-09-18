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
from navis.identity.permissions import developer_grant_key, effective_permissions
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


def principal_cache_evict_org(org_id: uuid.UUID) -> int:
    """Drop every cached Principal belonging to an organization.

    Called on a lifecycle transition (see `organization.lifecycle`). Suspending or
    churning an organization does not change anyone's PERMISSIONS — the capability
    gate reads lifecycle from the database on every request, so entitlements need
    no invalidation — but it does change whether those permissions may be used at
    all, and an operator suspending a tenant reasonably expects that to take
    effect now rather than within a TTL.

    Returns the number of entries evicted, so the caller can report it. Cheap: the
    cache is bounded at `_PRINCIPAL_CACHE_MAX`, so this is a walk of at most a
    thousand entries and runs only on a lifecycle change.
    """
    victims = [k for k, (_exp, _uid, p) in _PRINCIPAL_CACHE.items() if p.org_id == org_id]
    for k in victims:
        _PRINCIPAL_CACHE.pop(k, None)
    if victims:
        metrics.incr(
            "auth.cache.evict", value=len(victims), labels={"reason": "org_lifecycle"}
        )
    return len(victims)


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
    #: The member's role on the DEVELOPER surface, or None for no Developer
    #: access (P1-6). Reported so `/auth/me` can tell a console what the member
    #: is, and so an audit record can name it.
    #:
    #: ALREADY FOLDED INTO `permissions` ABOVE — reading this field is never how
    #: a gate decides anything. `effective_permissions` unions the Application
    #: and Developer halves into the one set `require_permission` checks, so
    #: there is exactly one authorization input and this is a label on it.
    #: None for every machine principal: a credential's reach is its
    #: `api_keys.scopes` and it holds no membership row to have a role in.
    developer_role: str | None = None
    user: User | None = None
    scopes: list[str] = field(default_factory=list)
    # WHICH CREDENTIAL authenticated this request, as distinct from WHO it
    # authenticated. `subject_id` collapses the two for machine callers: when a
    # key is bound to a service account it becomes the SERVICE ACCOUNT's id, so
    # the `api_keys` row that actually presented itself was unrecoverable from
    # the resolved principal. A service account may hold several keys over its
    # life (`ServiceAccountService.rotate` mints one and revokes the rest), and
    # every question about a machine call — which credential to revoke after a
    # leak, which one a rotation retired, which one is spending a quota, which
    # one performed an audited read — is a question about the KEY, not the
    # account. Non-null for `api_key`/`service_account` principals and always
    # the `api_keys.id`.
    #
    # None for `user` principals: a JWT's credential identity is its `jti`,
    # which lives in the token rather than in a table, is already enforced by
    # the revocation denylist, and has no attribution consumer. Carrying it
    # here would put a session identifier on a field whose whole contract is
    # "a revocable credential row", so the honest value is absence.
    credential_id: uuid.UUID | None = None

    def has(self, permission: str) -> bool:
        return permission in self.permissions


async def grants_for_roles(
    db: AsyncSession, org_id: uuid.UUID, roles: list[str]
) -> dict[str, list[str]]:
    """Per-org permission grants for SEVERAL role keys, in ONE round-trip.

    EXISTS FOR THE ROUND-TRIP COUNT, NOT FOR TIDINESS. Since P1-6 a human
    principal resolves two role keys — the Application role and the namespaced
    Developer role (`developer_grant_key`) — and `_principal_from_jwt` is the
    hot path that already pays four sequential Neon round-trips. Asking twice
    would add a fifth to every authenticated request for a table that is empty
    in every deployment today.

    Returns a dict keyed by the role strings that were ASKED FOR, so a caller
    can index it without checking for absence; a role with no rows maps to the
    empty list. Blank keys are dropped before querying — `developer_grant_key`
    returns "" for a member with no Developer role, and that must match nothing
    rather than becoming a wildcard.
    """
    wanted = [r for r in roles if r]
    if not wanted:
        return {r: [] for r in roles}
    rows = (
        await db.scalars(
            select(RolePermission).where(
                RolePermission.role.in_(wanted),
                RolePermission.organization_id.in_([org_id, None]),
            )
        )
    ).all()
    out: dict[str, list[str]] = {r: [] for r in roles}
    for row in rows:
        out.setdefault(row.role, []).append(row.permission)
    return out


async def grants_for(db: AsyncSession, org_id: uuid.UUID, role: str) -> list[str]:
    """Per-org permission grants layered on a role's code defaults.

    PUBLIC because membership administration must resolve a TARGET's effective
    permissions exactly as this resolves the ACTOR's. Two queries with the same
    intent would be free to drift, and a drift here means an authorization
    comparison between two differently-computed permission sets.

    Delegates to `grants_for_roles` for exactly that reason: the batched form is
    now the one that the authentication path uses, so this single-role form must
    be the same query rather than a second one that resembles it.
    """
    return (await grants_for_roles(db, org_id, [role]))[role]


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
    # THE TWO SURFACES, RESOLVED TOGETHER (P1-6). `memberships.role` answers for
    # the Application surface and `memberships.developer_role` for the Developer
    # surface; `effective_permissions` unions them into the one set every gate
    # checks. A NULL `developer_role` contributes nothing, which is precisely
    # what "this member has no Developer access" has to mean.
    #
    # Both grant lookups are ONE query (`grants_for_roles`), so the surface split
    # costs this hot path no additional round-trip.
    developer_role = membership.developer_role
    developer_key = developer_grant_key(developer_role)
    grants_by_role = await grants_for_roles(
        db, membership.organization_id, [role or "", developer_key]
    )

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
        permissions=effective_permissions(
            role,
            grants_by_role[role or ""],
            developer_role=developer_role,
            developer_grants=grants_by_role[developer_key],
        ),
        role=role,
        developer_role=developer_role,
        user=user,
    )
    _principal_cache_put(jti, principal)
    return principal


async def _principal_from_api_key(db: AsyncSession, plaintext: str) -> Principal:
    from navis.identity.apikeys import ApiKeyService, CredentialRefusal
    from navis.identity.enterprise_models import ServiceAccount

    checked = await ApiKeyService(db).check(plaintext)
    key = checked.key
    if key is None:
        # WHICH REFUSAL IS NAMED, AND WHY ONLY THIS ONE.
        #
        # `expired` is reported because only the holder of a genuine credential
        # can provoke it — a guess produces `malformed` or `unknown` — so it
        # reveals nothing the caller did not already possess, and it is the
        # difference between "rotate this key" and an afternoon spent checking
        # the header for typos. It is also what OpenAI and Anthropic answer.
        #
        # `revoked` is deliberately NOT distinguished on the wire. It is also
        # only reachable with a real key, but a revoked credential is one
        # somebody chose to cut off, and confirming to whoever still holds it
        # that it was specifically revoked tells them the organization noticed.
        # "Invalid" is the truthful and sufficient answer there.
        #
        # The metric keeps all four apart for operators, who are the reader that
        # actually needs to tell a forgotten rotation from a deliberate cutoff.
        reason = checked.refusal.value if checked.refusal else "unknown"
        metrics.incr("auth.api_key_refused", labels={"reason": reason})
        if checked.refusal is CredentialRefusal.EXPIRED:
            # The PREFIX only. It is the public, non-secret lookup segment the
            # console already displays, and it is what lets an operator find the
            # row; the secret segment is never touched. A malformed key cannot
            # reach here, so the split is safe.
            logger.info(
                "auth denied: api key expired prefix=%s", plaintext.split("_", 2)[1]
            )
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "api key expired; mint a replacement in the Navis console",
            )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid api key")
    kind = PrincipalKind.API_KEY
    if key.service_account_id is not None:
        sa = await db.get(ServiceAccount, key.service_account_id)
        if sa is None or sa.revoked_at is not None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "service account revoked")
        # THE CREDENTIAL -> AGENT -> ORGANIZATION CHAIN, CLOSED.
        #
        # The organization below is taken from the KEY and the subject from the
        # SERVICE ACCOUNT, and until now nothing checked that those two rows name
        # the same tenant. No route can currently create that mismatch — the only
        # writer of `api_keys.service_account_id` is `ServiceAccountService`,
        # which sets both from one org — but "unreachable today" is a property of
        # the callers, not of this function, and this function is the one place
        # every machine request is authorized from. The database cannot help: the
        # FK is to `service_accounts.id` alone, so no constraint spans the pair.
        #
        # A mismatch would produce a principal whose org_id is tenant A and whose
        # acting agent belongs to tenant B — every downstream tenant filter would
        # then work perfectly and still be filtering the wrong tenant. Fail closed
        # rather than trust an invariant nothing enforces.
        if sa.organization_id != key.organization_id:
            logger.error(
                "credential/agent tenant mismatch: key=%s org=%s service_account=%s org=%s",
                key.id, key.organization_id, sa.id, sa.organization_id,
            )
            metrics.incr("auth.rejected", labels={"reason": "credential_org_mismatch"})
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid api key")
        kind = PrincipalKind.SERVICE_ACCOUNT
    scopes = list(key.scopes or [])
    return Principal(
        kind=kind,
        org_id=key.organization_id,
        subject_id=key.service_account_id or key.id,
        permissions=frozenset(scopes),
        scopes=scopes,
        # ALWAYS the api_keys row, never the service account. See the field note
        # on `Principal.credential_id`: `subject_id` above deliberately reports
        # the ACCOUNT when there is one, which is what authorization asks about;
        # attribution, rotation and revocation ask about the KEY, and this is
        # the only place that still knows it.
        credential_id=key.id,
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


def ensure_grantable_scopes(scopes: list[str], principal: Principal) -> None:
    """A machine credential may never hold a permission its creator lacks.

    Closes the "mint an api key with `execute_actions` you don't have"
    escalation, and rejects typo/unknown scopes (400) so garbage never persists
    into a credential.

    LIVES HERE, NOT ON A ROUTER. It began as a private helper in
    `identity.admin_router` and is now needed by a second edge
    (`developer.agents_router`). Two copies of an escalation guard is one copy
    that gets a fix and one that does not, and the divergence is invisible until
    a credential is minted through the wrong door. `authz` is the module that
    already owns "what may this principal do", so it owns this too.

    IT ALSO ENFORCES THE HUMAN-ONLY BOUND (P1-8), and that is a SERVER rule
    rather than a console default. The permission list the console offers when
    minting a credential is documented as a management-surface default and not
    a server rule, so omitting `manage_developer_access` from it was not
    sufficient: a Developer owner holds the permission, the creator-bound
    check below would therefore have passed it, and the resulting credential
    would have satisfied the member-administration gates. An agent able to
    grant humans Developer access is an escalation out of the agent sandbox,
    so the refusal belongs at the one chokepoint every credential goes
    through — this function, which both `identity.admin_router` and
    `developer.agents_router` already call.

    `HUMAN_ONLY_PERMISSIONS` is a BOUND, not a description: unlike the
    catalog, it is consulted here deliberately, and the test that forbids this
    function from reading descriptive tables is about exactly that
    distinction.
    """
    from navis.identity.permissions import HUMAN_ONLY_PERMISSIONS, validate_permission

    for scope in scopes:
        try:
            validate_permission(scope)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    # Checked BEFORE the creator-bound comparison, so the refusal names the
    # real reason. Otherwise a holder of the permission would be told nothing
    # (it is not "excess" for them) and only a non-holder would be refused —
    # which is the wrong rule entirely.
    human_only = set(scopes) & HUMAN_ONLY_PERMISSIONS
    if human_only:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "these permissions administer people and can never be granted to "
            f"a credential: {sorted(human_only)}",
        )
    excess = set(scopes) - set(principal.permissions)
    if excess:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"cannot grant scopes you do not hold: {sorted(excess)}",
        )


def require_any_permission(*permissions: str):
    """As `require_permission`, but any ONE of `permissions` is enough (P1-8).

    WHY A DISJUNCTION, AND WHY IT IS NOT A SECOND AUTHORIZATION SYSTEM. Three
    identity-administration routes are reachable two genuinely different ways:
    an Application administrator holding `manage_roles`/`manage_members`, and a
    Developer administrator holding `manage_developer_access`. Both are real
    answers to "may you call this route" — the first administers the
    organization, the second administers the Developer team inside it — and
    neither implies the other.

    ADDITIVE BY CONSTRUCTION. Every existing caller passed on the Application
    permission and still does; the alternative is only ever consulted after
    that check fails. So no route becomes harder to reach and no credential
    loses anything.

    The input is still `Principal.permissions`, resolved by the one
    `effective_permissions`. This asks the existing evaluator a question with
    an `or` in it, which is the only thing it adds — the same shape, and the
    same reasoning, as `developer.gateway.require_developer_any`.

    The denial names every acceptable permission rather than just the first:
    an operator refused here needs to know which of the two grants would fix
    it, and reporting only one sends a Developer admin to ask for
    `manage_roles` — the Application permission this route exists to stop
    requiring.
    """
    if not permissions:  # a gate that accepts nothing would deny everyone
        raise ValueError("require_any_permission needs at least one permission")

    async def _dep(principal: Principal = Depends(get_principal)) -> Principal:
        if not any(principal.has(p) for p in permissions):
            wanted = " or ".join(permissions)
            logger.warning(
                "permission denied: principal=%s kind=%s org=%s perm=%s",
                principal.subject_id, principal.kind, principal.org_id, wanted,
            )
            # Labelled by the whole disjunction, so the metric distinguishes
            # this gate from the single-permission ones it replaced.
            metrics.incr("auth.permission_denied", labels={"permission": wanted})
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, f"missing permission: {wanted}"
            )
        return principal

    return _dep


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
