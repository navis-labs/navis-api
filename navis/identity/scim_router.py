"""Minimal SCIM 2.0 surface (enterprise provisioning foundation).

Users → app users+memberships; Groups → scim_groups. This is a viable shell
(list/create/deprovision) gated by MANAGE_MEMBERS, not a full RFC 7644
implementation (no PATCH ops, filtering, or pagination cursors yet).
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.authz import Principal, require_permission
from navis.identity.enterprise_models import ScimGroup
from navis.identity.members import MemberError, MemberService
from navis.identity.models import Membership, Role, User
from navis.identity.permissions import Permission
from navis.identity.tokens import TokenRevocationService
from navis.platform.database import get_session

router = APIRouter(prefix="/scim/v2", tags=["scim"])

_USER_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:User"
_GROUP_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:Group"


def _user_resource(user: User, membership: Membership) -> dict[str, Any]:
    return {
        "schemas": [_USER_SCHEMA],
        "id": str(user.id),
        "userName": user.email,
        "active": user.is_active,
        "name": {"formatted": user.full_name},
        "roles": [membership.role.value] if membership.role else [],
    }


class ScimUserBody(BaseModel):
    userName: str
    active: bool = True


@router.get("/Users")
async def list_users(
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    memberships = await MemberService(db).list_members(p.org_id)
    resources = []
    for m in memberships:
        user = await db.get(User, m.user_id)
        if user:
            resources.append(_user_resource(user, m))
    return {"schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            "totalResults": len(resources), "Resources": resources}


@router.post("/Users", status_code=status.HTTP_201_CREATED)
async def provision_user(
    body: ScimUserBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        m = await MemberService(db).invite(
            org_id=p.org_id, email=body.userName, role=Role.EMPLOYEE, actor_id=p.subject_id
        )
    except MemberError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    user = await db.get(User, m.user_id)
    return _user_resource(user, m)


@router.delete("/Users/{user_id}")
async def deprovision_user(
    user_id: str,
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    uid = uuid.UUID(user_id)
    try:
        await MemberService(db).remove(org_id=p.org_id, user_id=uid, actor_id=p.subject_id)
    except MemberError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    # Deprovisioning also deactivates the user and revokes their tokens.
    user = await db.get(User, uid)
    if user:
        user.is_active = False
    await TokenRevocationService(db).revoke_all_for_user(uid)
    return {"status": "deprovisioned"}


class ScimGroupBody(BaseModel):
    displayName: str
    externalId: str | None = None
    members: list[str] = []


@router.get("/Groups")
async def list_groups(
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = (await db.scalars(select(ScimGroup).where(ScimGroup.organization_id == p.org_id))).all()
    return {
        "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
        "totalResults": len(rows),
        "Resources": [
            {"schemas": [_GROUP_SCHEMA], "id": str(g.id), "displayName": g.display_name,
             "members": [{"value": m} for m in (g.members or [])]}
            for g in rows
        ],
    }


@router.post("/Groups", status_code=status.HTTP_201_CREATED)
async def create_group(
    body: ScimGroupBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    group = ScimGroup(
        organization_id=p.org_id, display_name=body.displayName,
        external_id=body.externalId, members=body.members,
    )
    db.add(group)
    await db.flush()
    return {"schemas": [_GROUP_SCHEMA], "id": str(group.id), "displayName": group.display_name}
