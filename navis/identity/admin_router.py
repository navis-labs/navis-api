"""Delegated-administration surface — all permission-gated (never membership-only)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.agents import credential_status_of
from navis.identity.apikeys import ApiKeyService
from navis.identity.audit import AuditAction, AuditService
from navis.identity.authz import (
    Principal,
    ensure_grantable_scopes,
    require_any_permission,
    require_permission,
)
from navis.identity.members import MemberConflict, MemberError, MemberForbidden, MemberService
from navis.identity.models import DeveloperRole, Role, User
from navis.identity.permissions import Permission
from navis.identity.service_accounts import ServiceAccountService
from navis.identity.sso import SsoConnectionService
from navis.platform.database import get_session
from navis.platform.events import DomainEvent, EventType, event_bus

router = APIRouter(prefix="/identity", tags=["identity-admin"])


# The escalation guard now lives in `identity.authz` — the Agents API needs the
# same rule, and two copies of "may this principal grant this scope" is one copy
# that gets a fix and one that does not. Re-exported under the module-private
# name this file has always used so its call sites (and their tests) are unchanged.
_ensure_grantable_scopes = ensure_grantable_scopes


async def _emit(event_type: EventType, org_id: uuid.UUID, actor_id: uuid.UUID, **payload: Any) -> None:
    """Publish an identity-admin event after a successful mutation (best-effort)."""
    await event_bus.publish(
        DomainEvent(
            type=event_type,
            org_id=org_id,
            source="identity",
            payload={"actor_id": str(actor_id), **{k: str(v) for k, v in payload.items()}},
        )
    )


# ---- members ----

#: Seeing the team and adding to it (P1-8).
#:
#: TWO GRANTS, ONE ROUTE, AND NEITHER IMPLIES THE OTHER. `manage_members` is
#: the Application administrator's power over the organization's people.
#: `manage_developer_access` is the Developer administrator's power over the
#: Developer team. A Developer-first company has the second and frequently not
#: the first: its Developer admin is an Application `employee`, and before this
#: could not list their own team or invite the engineer who needed an agent.
#:
#: STRICTLY ADDITIVE. Every caller that passed before still passes on
#: `manage_members` alone; the alternative is consulted only after that fails.
#:
#: WHAT THIS DOES NOT WIDEN. `MemberService._ensure_grantable` still requires
#: the actor to already hold every permission the invitation would confer, so a
#: Developer admin can invite a Developer and cannot smuggle in an Application
#: role they do not have themselves. And a personal workspace still refuses a
#: second member regardless of who is asking.
_MAY_ADMINISTER_THE_TEAM = require_any_permission(
    Permission.MANAGE_MEMBERS, Permission.MANAGE_DEVELOPER_ACCESS
)

#: Granting, changing or revoking a member's Developer access (P1-8).
#:
#: Paired with `MANAGE_ROLES` for the same reason: an Application administrator
#: reaches this route as they always did, and a Developer administrator now
#: reaches it too. The Application half is deliberately NOT sufficient on its
#: own to succeed — `_ensure_grantable` refuses an Application admin who holds
#: no Developer authority, which is the P1-6 escalation guard and is unchanged.
#:
#: NOT EXTENDED TO `DELETE /members/{id}`, deliberately. Removing a member
#: evicts them from the ORGANIZATION — it drops their Application access and
#: bulk-revokes their tokens — and that is an organizational act, not a
#: Developer one. The Developer-scoped equivalent is revoking Developer access,
#: which is this route with `developer_role: null`.
_MAY_ASSIGN_DEVELOPER_ACCESS = require_any_permission(
    Permission.MANAGE_ROLES, Permission.MANAGE_DEVELOPER_ACCESS
)


class InviteBody(BaseModel):
    email: EmailStr  # format-validated by pydantic at the edge
    role: Role = Role.EMPLOYEE
    #: Developer access for the invitee, or null for Application-only (P1-6).
    #:
    #: DEFAULTS TO NULL, which is the requirement expressed as a default: an
    #: invite that does not mention the Developer surface grants nothing on it,
    #: so Developer access can never be acquired by omission. Every caller that
    #: predates this field therefore keeps its exact previous behaviour.
    developer_role: DeveloperRole | None = None


class RoleBody(BaseModel):
    role: Role


class DeveloperRoleBody(BaseModel):
    """The Developer-access assignment.

    `developer_role: null` REVOKES access and is an explicit, valid value — not
    an omission. It is required rather than defaulted precisely so that "revoke"
    has to be written down: a caller cannot strip somebody's Developer access by
    sending an empty body.
    """

    developer_role: DeveloperRole | None = Field(...)


def _member_out(m: Any, user: Any | None = None) -> dict[str, Any]:
    """One membership on the wire, both surfaces.

    `developer_role` is reported alongside `role` because the console's members
    view has to show who can reach the Developer surface; with only `role` it
    could not tell an Application `employee` from a Developer.

    WHO THE PERSON IS, NOT JUST WHICH ROW THEY ARE (P0-8). This used to report
    `user_id` and two role strings, so the console's members table rendered a
    column of raw UUIDs — an operator granting Developer access could not tell
    which colleague they were granting it to, which makes the safest available
    action "grant it to everyone and see who complains". Email is the identity
    key the invite is addressed to, so it is the right label.

    STILL NO PERMISSION LIST. `/auth/me` publishes permissions for the ACTIVE
    session only; adding identity does not change that, and this response
    remains unable to answer "what may that other person do".

    `user` is optional because the three single-member routes below have a
    membership in hand and load the user beside it; a caller that cannot
    supply one gets the roles alone rather than a fabricated address.
    """
    out = {
        "user_id": str(m.user_id),
        "role": m.role.value if m.role else None,
        "developer_role": m.developer_role,
    }
    if user is not None:
        out["email"] = user.email
        out["full_name"] = user.full_name
        out["status"] = _member_status(user)
    return out


def _member_status(user: Any) -> str:
    """`invited` until the person has signed in even once, then `active`.

    DERIVED, NOT STORED, AND THAT IS DELIBERATE. There is no invitations table
    and this does not invent one: `MemberService.invite` creates the real
    `users` row and the real `memberships` row immediately, so an invitee has
    live access from the moment they are invited. The honest thing to report is
    therefore not "pending approval" — nothing is pending — but whether they
    have ever authenticated.

    A user with neither a password nor a Google subject has no way to have
    signed in, so they cannot have. Both fields are set by the paths that
    establish a credential (`signup`/`authenticate` and `google_signin`'s
    backfill), and `invite` sets neither. `is_active=False` is a different
    axis — a deactivated member who HAS signed in before still reads `active`
    here, because this describes onboarding, not standing.
    """
    signed_in = bool(getattr(user, "hashed_password", None)) or bool(
        getattr(user, "google_sub", None)
    )
    return "active" if signed_in else "invited"


@router.get("/members")
async def list_members(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    p: Principal = Depends(_MAY_ADMINISTER_THE_TEAM),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    # The JOINed read that already existed for SCIM, rather than N+1 user
    # loads or a second query — `list_members` and `list_members_with_users`
    # order identically, so this changes the response body and not the page.
    rows = await MemberService(db).list_members_with_users(
        p.org_id, limit=limit, offset=offset
    )
    return {
        "members": [_member_out(m, u) for m, u in rows],
        "limit": limit, "offset": offset, "count": len(rows),
    }


@router.post("/members/invite", status_code=status.HTTP_201_CREATED)
async def invite_member(
    body: InviteBody,
    p: Principal = Depends(_MAY_ADMINISTER_THE_TEAM),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        m = await MemberService(db).invite(
            org_id=p.org_id, email=body.email, role=body.role,
            actor_id=p.subject_id, actor_permissions=p.permissions,
            developer_role=body.developer_role,
        )
    except MemberForbidden as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc))
    except MemberError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    await _emit(
        EventType.MEMBER_INVITED, p.org_id, p.subject_id,
        user_id=m.user_id, role=m.role.value,
        developer_role=m.developer_role or "none",
    )
    return _member_out(m, await db.get(User, m.user_id))


@router.delete("/members/{user_id}")
async def remove_member(
    user_id: uuid.UUID,  # FastAPI validates the UUID path param; bad input -> 422
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        await MemberService(db).remove(
            org_id=p.org_id, user_id=user_id, actor_id=p.subject_id,
            actor_permissions=p.permissions,
        )
    except MemberForbidden as exc:
        # `remove` is now target-aware, so it can refuse on privilege. Listed FIRST:
        # MemberForbidden subclasses MemberError, and falling through to the 404 arm
        # would report an authorization decision as a missing member.
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc))
    except MemberConflict as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    except MemberError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    await _emit(EventType.MEMBER_REMOVED, p.org_id, p.subject_id, user_id=user_id)
    return {"status": "removed"}


@router.post("/members/{user_id}/role")
async def assign_role(
    user_id: uuid.UUID,
    body: RoleBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_ROLES)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        m = await MemberService(db).assign_role(
            org_id=p.org_id, user_id=user_id, role=body.role,
            actor_id=p.subject_id, actor_permissions=p.permissions,
        )
    except MemberForbidden as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc))
    except MemberConflict as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    except MemberError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    await _emit(EventType.ROLE_ASSIGNED, p.org_id, p.subject_id, user_id=user_id, role=m.role.value)
    return _member_out(m, await db.get(User, m.user_id))


@router.post("/members/{user_id}/access")
async def assign_surface_access(
    user_id: uuid.UUID,
    body: DeveloperRoleBody,
    p: Principal = Depends(_MAY_ASSIGN_DEVELOPER_ACCESS),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Grant, change or revoke a member's Developer access (P1-6).

    NAMED `/access` AND NOT `/developer-role`, deliberately. Every path
    containing the word "developer" must live under `/api/v1/developer/` —
    `test_the_developer_api_is_mounted_on_the_one_application` asserts exactly
    that, because the Developer API owns that segment of the URL space and a
    second place where "developer" appears is how "one boundary, one schema"
    starts to erode. This is an identity-administration route that happens to
    ASSIGN Developer access; it is not part of the Developer API, is gated on an
    Application permission, and belongs where its siblings are.

    GATED ON THE EXISTING `MANAGE_ROLES`, and deliberately not on a new
    permission. "Who may change what a member may do" is one question, and the
    permission that answers it already exists; a `manage_developer_access`
    permission would be a second answer to it, free to disagree.

    THAT ALONE WOULD NOT BE SUFFICIENT, AND IS NOT WHAT AUTHORIZES THE GRANT.
    `MANAGE_ROLES` gets the caller through the door; `MemberService.
    _ensure_grantable` then requires them to already hold every permission the
    assignment would confer. Since the surface split, an Application `admin`
    holds no Developer permission at all, so an Application administrator
    reaches this route and is refused 403 — which is exactly the requirement
    that Developer authority is never inherited from an Application role.

    A SEPARATE ROUTE FROM `POST /members/{id}/role`, so that neither surface's
    contract changes when the other is assigned. This is additive: the existing
    role route is untouched.
    """
    try:
        m = await MemberService(db).assign_developer_role(
            org_id=p.org_id, user_id=user_id, developer_role=body.developer_role,
            actor_id=p.subject_id, actor_permissions=p.permissions,
        )
    except MemberForbidden as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc))
    except MemberConflict as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    except MemberError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    await _emit(
        EventType.ROLE_ASSIGNED, p.org_id, p.subject_id,
        user_id=user_id, developer_role=m.developer_role or "none",
    )
    return _member_out(m, await db.get(User, m.user_id))


# ---- api keys ----

class ApiKeyBody(BaseModel):
    name: str
    scopes: list[str] = Field(default_factory=list)
    #: When this credential stops working. None — the default — means it does
    #: not expire, so an existing caller that omits the field gets exactly the
    #: key it got before. Must be in the future; `ApiKeyService.create` is the
    #: one place that decides, and it refuses an instant already past.
    expires_at: datetime | None = None


@router.get("/api-keys")
async def list_api_keys(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    p: Principal = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The organization's keys, with the SAME four-state status the agent
    credential list reports.

    `revoked` alone could not answer "will this key work on the next request"
    once a key could carry an expiry: a credential that aged out yesterday has
    `revoked: false` and authenticates nothing. `status` comes from
    `credential_status_of` — the identical derivation
    `/developer/agents/{id}/credentials` uses — so the two lists cannot disagree
    about a row that appears in both, which every agent credential does.

    `revoked` is kept alongside it rather than replaced: it is in the contract
    the console and its tests already read, and the two fields answer different
    questions (`revoked` is whether somebody cut this off, `status` is whether it
    works).
    """
    rows = await ApiKeyService(db).list_for_org(p.org_id, limit=limit, offset=offset)
    # ONE wall-clock reading for the whole page, so two keys with adjacent
    # expiries cannot be stamped against different "now"s.
    now = datetime.now(timezone.utc)
    return {"api_keys": [
        {"id": str(k.id), "name": k.name, "prefix": k.prefix, "scopes": k.scopes,
         "revoked": k.revoked_at is not None,
         "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
         # Null means "does not expire", which is what every key minted before
         # 0052 is — NOT a missing value.
         "expires_at": k.expires_at.isoformat() if k.expires_at else None,
         "status": credential_status_of(k.revoked_at, k.expires_at, now=now)}
        for k in rows
    ], "limit": limit, "offset": offset, "count": len(rows)}


@router.post("/api-keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: ApiKeyBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        _ensure_grantable_scopes(body.scopes, p)
        key, plaintext = await ApiKeyService(db).create(
            org_id=p.org_id,
            name=body.name,
            scopes=body.scopes,
            created_by=p.subject_id,
            expires_at=body.expires_at,
        )
    except ValueError as exc:
        # An expiry already in the past. 400 rather than 422, matching the two
        # Developer minting routes: the field parsed, the instant it names is
        # the problem. `_ensure_grantable_scopes` raises `HTTPException`, not
        # `ValueError`, so its 400/403 still pass through untouched.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    await AuditService(db).record(
        AuditAction.API_KEY_CREATE, org_id=p.org_id, actor_id=p.subject_id, target_id=str(key.id),
        metadata={"expires_at": key.expires_at.isoformat() if key.expires_at else None},
    )
    await _emit(EventType.API_KEY_CREATED, p.org_id, p.subject_id, key_id=key.id)
    # `expires_at` is echoed so a caller knows what it got without a second
    # read. The secret is still the only sensitive field, and still shown ONCE.
    return {
        "id": str(key.id),
        "prefix": key.prefix,
        "secret": plaintext,
        "expires_at": key.expires_at.isoformat() if key.expires_at else None,
    }


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: uuid.UUID,
    p: Principal = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    revoked = await ApiKeyService(db).revoke(key_id, org_id=p.org_id)
    if revoked is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "api key not found")
    await AuditService(db).record(
        AuditAction.API_KEY_REVOKE, org_id=p.org_id, actor_id=p.subject_id, target_id=str(key_id)
    )
    await _emit(EventType.API_KEY_REVOKED, p.org_id, p.subject_id, key_id=key_id)
    return {"status": "revoked"}


# ---- service accounts ----

class ServiceAccountBody(BaseModel):
    name: str
    scopes: list[str] = Field(default_factory=list)
    #: Bounds the FIRST CREDENTIAL, not the account. A service account has no
    #: expiry column and must not grow one — it is the identity that outlives
    #: its secrets. Named as it is on `POST /developer/agents` for the same
    #: reason, since both create an identity plus one credential.
    credential_expires_at: datetime | None = None


class RotateBody(BaseModel):
    """Optional body for a rotation. The route accepted none before this."""

    #: Lifetime for the REPLACEMENT credential. Omitted means it does not
    #: expire, which is what every rotation produced before — `rotate` never
    #: inherits the retired credential's expiry, deliberately. See
    #: `ServiceAccountService.rotate`.
    expires_at: datetime | None = None


@router.get("/service-accounts")
async def list_service_accounts(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    p: Principal = Depends(require_permission(Permission.MANAGE_SERVICE_ACCOUNTS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = await ServiceAccountService(db).list_for_org(p.org_id, limit=limit, offset=offset)
    return {"service_accounts": [
        {"id": str(s.id), "name": s.name, "scopes": s.scopes, "revoked": s.revoked_at is not None}
        for s in rows
    ], "limit": limit, "offset": offset, "count": len(rows)}


@router.post("/service-accounts", status_code=status.HTTP_201_CREATED)
async def create_service_account(
    body: ServiceAccountBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_SERVICE_ACCOUNTS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        _ensure_grantable_scopes(body.scopes, p)
        sa, plaintext = await ServiceAccountService(db).create(
            org_id=p.org_id,
            name=body.name,
            scopes=body.scopes,
            created_by=p.subject_id,
            expires_at=body.credential_expires_at,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    await AuditService(db).record(
        AuditAction.SERVICE_ACCOUNT_CREATE, org_id=p.org_id, actor_id=p.subject_id, target_id=str(sa.id)
    )
    await _emit(EventType.SERVICE_ACCOUNT_CREATED, p.org_id, p.subject_id, service_account_id=sa.id)
    return {"id": str(sa.id), "secret": plaintext}  # shown ONCE


@router.post("/service-accounts/{sa_id}/rotate")
async def rotate_service_account(
    sa_id: uuid.UUID,
    # OPTIONAL body, so every existing caller — which sends none — is unchanged.
    body: RotateBody | None = None,
    p: Principal = Depends(require_permission(Permission.MANAGE_SERVICE_ACCOUNTS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        plaintext = await ServiceAccountService(db).rotate(
            sa_id, org_id=p.org_id, expires_at=body.expires_at if body else None
        )
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    await AuditService(db).record(
        AuditAction.SERVICE_ACCOUNT_ROTATE, org_id=p.org_id, actor_id=p.subject_id, target_id=str(sa_id)
    )
    await _emit(EventType.SERVICE_ACCOUNT_ROTATED, p.org_id, p.subject_id, service_account_id=sa_id)
    return {"secret": plaintext}


@router.delete("/service-accounts/{sa_id}")
async def revoke_service_account(
    sa_id: uuid.UUID,
    p: Principal = Depends(require_permission(Permission.MANAGE_SERVICE_ACCOUNTS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        await ServiceAccountService(db).revoke(sa_id, org_id=p.org_id)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    await AuditService(db).record(
        AuditAction.SERVICE_ACCOUNT_REVOKE, org_id=p.org_id, actor_id=p.subject_id, target_id=str(sa_id)
    )
    await _emit(EventType.SERVICE_ACCOUNT_REVOKED, p.org_id, p.subject_id, service_account_id=sa_id)
    return {"status": "revoked"}


# ---- audit + sso ----

@router.get("/audit")
async def list_audit(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    since: datetime | None = Query(None, description="Only events at/after this UTC timestamp"),
    p: Principal = Depends(require_permission(Permission.VIEW_AUDIT)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = await AuditService(db).list_for_org(p.org_id, limit=limit, offset=offset, since=since)
    return {"audit": [
        {"action": r.action, "actor_id": str(r.actor_id) if r.actor_id else None,
         "actor_kind": r.actor_kind, "target_id": r.target_id, "metadata": r.audit_metadata,
         "occurred_at": r.occurred_at.isoformat() if r.occurred_at else None}
        for r in rows
    ], "limit": limit, "offset": offset, "count": len(rows)}


class SsoBody(BaseModel):
    provider: str
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = False


@router.put("/sso")
async def configure_sso(
    body: SsoBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_SSO)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        conn = await SsoConnectionService(db).upsert(
            org_id=p.org_id, provider=body.provider, config=body.config, enabled=body.enabled
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    # SSO controls who can log into the whole org — a high-value mutation that MUST
    # be on the audit trail. Metadata carries only non-secret facts (the config blob
    # itself holds the IdP secret and is encrypted at rest by the SSO service).
    await AuditService(db).record(
        AuditAction.SSO_UPDATED,
        org_id=p.org_id,
        actor_id=p.subject_id,
        target_id=str(p.org_id),
        metadata={"provider": conn.provider, "enabled": conn.enabled},
    )
    await _emit(
        EventType.SSO_UPDATED, p.org_id, p.subject_id,
        provider=conn.provider, enabled=conn.enabled,
    )
    return {"provider": conn.provider, "enabled": conn.enabled}
