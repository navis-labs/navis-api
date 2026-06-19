"""Delegated-administration surface — all permission-gated (never membership-only)."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.apikeys import ApiKeyService
from navis.identity.audit import AuditAction, AuditService
from navis.identity.authz import Principal, require_permission
from navis.identity.members import MemberError, MemberService
from navis.identity.models import Role
from navis.identity.permissions import Permission
from navis.identity.service_accounts import ServiceAccountService
from navis.identity.sso import SsoConnectionService
from navis.platform.database import get_session

router = APIRouter(prefix="/identity", tags=["identity-admin"])


# ---- members ----

class InviteBody(BaseModel):
    email: str
    role: Role = Role.EMPLOYEE


class RoleBody(BaseModel):
    role: Role


@router.get("/members")
async def list_members(
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = await MemberService(db).list_members(p.org_id)
    return {"members": [{"user_id": str(m.user_id), "role": m.role.value if m.role else None} for m in rows]}


@router.post("/members/invite", status_code=status.HTTP_201_CREATED)
async def invite_member(
    body: InviteBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        m = await MemberService(db).invite(org_id=p.org_id, email=body.email, role=body.role, actor_id=p.subject_id)
    except MemberError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return {"user_id": str(m.user_id), "role": m.role.value}


@router.delete("/members/{user_id}")
async def remove_member(
    user_id: str,
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        await MemberService(db).remove(org_id=p.org_id, user_id=uuid.UUID(user_id), actor_id=p.subject_id)
    except MemberError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return {"status": "removed"}


@router.post("/members/{user_id}/role")
async def assign_role(
    user_id: str,
    body: RoleBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_ROLES)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        m = await MemberService(db).assign_role(
            org_id=p.org_id, user_id=uuid.UUID(user_id), role=body.role, actor_id=p.subject_id
        )
    except MemberError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return {"user_id": str(m.user_id), "role": m.role.value}


# ---- api keys ----

class ApiKeyBody(BaseModel):
    name: str
    scopes: list[str] = []


@router.get("/api-keys")
async def list_api_keys(
    p: Principal = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = await ApiKeyService(db).list_for_org(p.org_id)
    return {"api_keys": [
        {"id": str(k.id), "name": k.name, "prefix": k.prefix, "scopes": k.scopes,
         "revoked": k.revoked_at is not None, "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None}
        for k in rows
    ]}


@router.post("/api-keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: ApiKeyBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    key, plaintext = await ApiKeyService(db).create(
        org_id=p.org_id, name=body.name, scopes=body.scopes, created_by=p.subject_id
    )
    await AuditService(db).record(
        AuditAction.API_KEY_CREATE, org_id=p.org_id, actor_id=p.subject_id, target_id=str(key.id)
    )
    return {"id": str(key.id), "prefix": key.prefix, "secret": plaintext}  # shown ONCE


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    p: Principal = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await ApiKeyService(db).revoke(uuid.UUID(key_id))
    await AuditService(db).record(
        AuditAction.API_KEY_REVOKE, org_id=p.org_id, actor_id=p.subject_id, target_id=key_id
    )
    return {"status": "revoked"}


# ---- service accounts ----

class ServiceAccountBody(BaseModel):
    name: str
    scopes: list[str] = []


@router.get("/service-accounts")
async def list_service_accounts(
    p: Principal = Depends(require_permission(Permission.MANAGE_SERVICE_ACCOUNTS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = await ServiceAccountService(db).list_for_org(p.org_id)
    return {"service_accounts": [
        {"id": str(s.id), "name": s.name, "scopes": s.scopes, "revoked": s.revoked_at is not None}
        for s in rows
    ]}


@router.post("/service-accounts", status_code=status.HTTP_201_CREATED)
async def create_service_account(
    body: ServiceAccountBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_SERVICE_ACCOUNTS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    sa, plaintext = await ServiceAccountService(db).create(
        org_id=p.org_id, name=body.name, scopes=body.scopes, created_by=p.subject_id
    )
    await AuditService(db).record(
        AuditAction.SERVICE_ACCOUNT_CREATE, org_id=p.org_id, actor_id=p.subject_id, target_id=str(sa.id)
    )
    return {"id": str(sa.id), "secret": plaintext}  # shown ONCE


@router.post("/service-accounts/{sa_id}/rotate")
async def rotate_service_account(
    sa_id: str,
    p: Principal = Depends(require_permission(Permission.MANAGE_SERVICE_ACCOUNTS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        plaintext = await ServiceAccountService(db).rotate(uuid.UUID(sa_id))
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    await AuditService(db).record(
        AuditAction.SERVICE_ACCOUNT_ROTATE, org_id=p.org_id, actor_id=p.subject_id, target_id=sa_id
    )
    return {"secret": plaintext}


@router.delete("/service-accounts/{sa_id}")
async def revoke_service_account(
    sa_id: str,
    p: Principal = Depends(require_permission(Permission.MANAGE_SERVICE_ACCOUNTS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        await ServiceAccountService(db).revoke(uuid.UUID(sa_id))
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    await AuditService(db).record(
        AuditAction.SERVICE_ACCOUNT_REVOKE, org_id=p.org_id, actor_id=p.subject_id, target_id=sa_id
    )
    return {"status": "revoked"}


# ---- audit + sso ----

@router.get("/audit")
async def list_audit(
    p: Principal = Depends(require_permission(Permission.VIEW_AUDIT)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = await AuditService(db).list_for_org(p.org_id)
    return {"audit": [
        {"action": r.action, "actor_id": str(r.actor_id) if r.actor_id else None,
         "actor_kind": r.actor_kind, "target_id": r.target_id, "metadata": r.audit_metadata,
         "occurred_at": r.occurred_at.isoformat() if r.occurred_at else None}
        for r in rows
    ]}


class SsoBody(BaseModel):
    provider: str
    config: dict[str, Any] = {}
    enabled: bool = False


@router.put("/sso")
async def configure_sso(
    body: SsoBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_ROLES)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        conn = await SsoConnectionService(db).upsert(
            org_id=p.org_id, provider=body.provider, config=body.config, enabled=body.enabled
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    return {"provider": conn.provider, "enabled": conn.enabled}
