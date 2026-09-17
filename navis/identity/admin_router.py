"""Delegated-administration surface — all permission-gated (never membership-only)."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.apikeys import ApiKeyService
from navis.identity.audit import AuditAction, AuditService
from navis.identity.authz import Principal, require_permission
from navis.identity.members import MemberConflict, MemberError, MemberForbidden, MemberService
from navis.identity.models import Role
from navis.identity.permissions import Permission, validate_permission
from navis.identity.service_accounts import ServiceAccountService
from navis.identity.sso import SsoConnectionService
from navis.platform.database import get_session
from navis.platform.events import DomainEvent, EventType, event_bus

router = APIRouter(prefix="/identity", tags=["identity-admin"])


def _ensure_grantable_scopes(scopes: list[str], principal: Principal) -> None:
    """A machine credential may never hold a permission its creator lacks — closes
    the 'mint an api key with execute_actions you don't have' escalation. Also rejects
    typo/unknown scopes (400) so garbage never persists into a credential."""
    for scope in scopes:
        try:
            validate_permission(scope)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    excess = set(scopes) - set(principal.permissions)
    if excess:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"cannot grant scopes you do not hold: {sorted(excess)}",
        )


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

class InviteBody(BaseModel):
    email: EmailStr  # format-validated by pydantic at the edge
    role: Role = Role.EMPLOYEE


class RoleBody(BaseModel):
    role: Role


@router.get("/members")
async def list_members(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = await MemberService(db).list_members(p.org_id, limit=limit, offset=offset)
    return {
        "members": [{"user_id": str(m.user_id), "role": m.role.value if m.role else None} for m in rows],
        "limit": limit, "offset": offset, "count": len(rows),
    }


@router.post("/members/invite", status_code=status.HTTP_201_CREATED)
async def invite_member(
    body: InviteBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        m = await MemberService(db).invite(
            org_id=p.org_id, email=body.email, role=body.role,
            actor_id=p.subject_id, actor_permissions=p.permissions,
        )
    except MemberForbidden as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc))
    except MemberError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    await _emit(EventType.MEMBER_INVITED, p.org_id, p.subject_id, user_id=m.user_id, role=m.role.value)
    return {"user_id": str(m.user_id), "role": m.role.value}


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
    return {"user_id": str(m.user_id), "role": m.role.value}


# ---- api keys ----

class ApiKeyBody(BaseModel):
    name: str
    scopes: list[str] = Field(default_factory=list)


@router.get("/api-keys")
async def list_api_keys(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    p: Principal = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = await ApiKeyService(db).list_for_org(p.org_id, limit=limit, offset=offset)
    return {"api_keys": [
        {"id": str(k.id), "name": k.name, "prefix": k.prefix, "scopes": k.scopes,
         "revoked": k.revoked_at is not None, "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None}
        for k in rows
    ], "limit": limit, "offset": offset, "count": len(rows)}


@router.post("/api-keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: ApiKeyBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _ensure_grantable_scopes(body.scopes, p)
    key, plaintext = await ApiKeyService(db).create(
        org_id=p.org_id, name=body.name, scopes=body.scopes, created_by=p.subject_id
    )
    await AuditService(db).record(
        AuditAction.API_KEY_CREATE, org_id=p.org_id, actor_id=p.subject_id, target_id=str(key.id)
    )
    await _emit(EventType.API_KEY_CREATED, p.org_id, p.subject_id, key_id=key.id)
    return {"id": str(key.id), "prefix": key.prefix, "secret": plaintext}  # shown ONCE


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
    _ensure_grantable_scopes(body.scopes, p)
    sa, plaintext = await ServiceAccountService(db).create(
        org_id=p.org_id, name=body.name, scopes=body.scopes, created_by=p.subject_id
    )
    await AuditService(db).record(
        AuditAction.SERVICE_ACCOUNT_CREATE, org_id=p.org_id, actor_id=p.subject_id, target_id=str(sa.id)
    )
    await _emit(EventType.SERVICE_ACCOUNT_CREATED, p.org_id, p.subject_id, service_account_id=sa.id)
    return {"id": str(sa.id), "secret": plaintext}  # shown ONCE


@router.post("/service-accounts/{sa_id}/rotate")
async def rotate_service_account(
    sa_id: uuid.UUID,
    p: Principal = Depends(require_permission(Permission.MANAGE_SERVICE_ACCOUNTS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        plaintext = await ServiceAccountService(db).rotate(sa_id, org_id=p.org_id)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
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
