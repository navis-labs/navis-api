"""Minimal SCIM 2.0 surface (enterprise provisioning foundation).

Users → app users+memberships; Groups → scim_groups. This is a viable shell
(list/create/deprovision) gated by MANAGE_MEMBERS, not a full RFC 7644
implementation (no PATCH ops, filtering, or pagination cursors yet).
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.authz import Principal, require_permission
from navis.identity.enterprise_models import ScimGroup
from navis.identity.members import (
    MemberConflict,
    MemberError,
    MemberForbidden,
    MemberService,
)
from navis.identity.models import Membership, Role, User
from navis.identity.permissions import Permission
from navis.identity.tokens import TokenRevocationService
from navis.platform.database import get_session

router = APIRouter(prefix="/scim/v2", tags=["scim"])

_USER_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:User"
_GROUP_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:Group"
_ERROR_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:Error"
_LIST_RESPONSE = "urn:ietf:params:scim:api:messages:2.0:ListResponse"

# Minimal SCIM filter: `attribute eq "value"` (what Okta/Entra send to check existence
# before provisioning). Full RFC 7644 filter grammar (and/or/co/sw) is future work.
_EQ_FILTER = re.compile(r'^\s*(\w+)\s+eq\s+"(.*)"\s*$', re.IGNORECASE)


def _scim_error(status_code: int, detail: str) -> "HTTPException":
    """SCIM error envelope (urn:...:Error) — IdPs parse this, not FastAPI's default."""
    return HTTPException(
        status_code,
        detail={"schemas": [_ERROR_SCHEMA], "detail": detail, "status": str(status_code)},
    )


def _parse_eq_filter(filter_str: str | None) -> tuple[str, str] | None:
    if not filter_str:
        return None
    m = _EQ_FILTER.match(filter_str)
    if not m:
        raise _scim_error(status.HTTP_400_BAD_REQUEST, f"unsupported filter: {filter_str!r}")
    return m.group(1), m.group(2)


def _excludes_members(excluded_attributes: str | None) -> bool:
    return bool(excluded_attributes) and "members" in excluded_attributes.lower()


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
    userName: EmailStr  # SCIM userName is an email for Navis; validate the format
    active: bool = True


@router.get("/Users")
async def list_users(
    startIndex: int = Query(1, ge=1),       # RFC 7644: 1-based
    count: int = Query(100, ge=1, le=500),  # page size
    filter: str | None = Query(None),       # e.g. userName eq "user@x.com"
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    svc = MemberService(db)

    parsed = _parse_eq_filter(filter)
    if parsed is not None:
        attr, value = parsed
        if attr.lower() != "username":
            raise _scim_error(status.HTTP_400_BAD_REQUEST, f"unsupported filter attribute: {attr}")
        # IdP existence check: return the single matching user (or an empty page).
        found = await svc.find_member_by_email(p.org_id, value)
        resources = [_user_resource(found[1], found[0])] if found else []
        return {"schemas": [_LIST_RESPONSE], "totalResults": len(resources),
                "startIndex": 1, "itemsPerPage": len(resources), "Resources": resources}

    total = await svc.count_members(p.org_id)
    # ONE join (not 1 + N user lookups), paginated per the SCIM list contract.
    pairs = await svc.list_members_with_users(p.org_id, limit=count, offset=startIndex - 1)
    resources = [_user_resource(u, m) for m, u in pairs]
    return {
        "schemas": [_LIST_RESPONSE],
        "totalResults": total,
        "startIndex": startIndex,
        "itemsPerPage": len(resources),
        "Resources": resources,
    }


@router.post("/Users", status_code=status.HTTP_201_CREATED)
async def provision_user(
    body: ScimUserBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        m = await MemberService(db).invite(
            org_id=p.org_id, email=body.userName, role=Role.EMPLOYEE,
            actor_id=p.subject_id, actor_permissions=p.permissions,
        )
    except MemberForbidden as exc:
        # Subclasses MemberError, so it MUST be caught first — otherwise a privilege
        # denial is reported as a provisioning conflict and an IdP retries it forever.
        raise _scim_error(status.HTTP_403_FORBIDDEN, str(exc))
    except MemberError as exc:
        raise _scim_error(status.HTTP_409_CONFLICT, str(exc))
    user = await db.get(User, m.user_id)
    if user is None:  # invariant: invite just flushed this user into the session
        raise _scim_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "user not found after provisioning"
        )
    return _user_resource(user, m)


@router.delete("/Users/{user_id}")
async def deprovision_user(
    user_id: uuid.UUID,
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        await MemberService(db).remove(
            org_id=p.org_id, user_id=user_id, actor_id=p.subject_id,
            actor_permissions=p.permissions,
        )
    except MemberForbidden as exc:
        # Ordering matters: both of these subclass MemberError. Without them a
        # refusal to remove a higher-privileged member, and the last-founder
        # governance guard, both surfaced as 404 "not a member" — which reads as
        # "already gone" to an IdP and hides a real authorization decision.
        raise _scim_error(status.HTTP_403_FORBIDDEN, str(exc))
    except MemberConflict as exc:
        raise _scim_error(status.HTTP_409_CONFLICT, str(exc))
    except MemberError as exc:
        raise _scim_error(status.HTTP_404_NOT_FOUND, str(exc))
    # Revoke this user's access (forces re-login; they keep access only to orgs they
    # still belong to). Only GLOBALLY deactivate when NO memberships remain — a SCIM
    # deprovision in one org must never lock a multi-org user out of their other orgs.
    await TokenRevocationService(db).revoke_all_for_user(user_id)
    remaining = await db.scalar(
        select(func.count()).select_from(Membership).where(Membership.user_id == user_id)
    )
    if not remaining:
        user = await db.get(User, user_id)
        if user:
            user.is_active = False
    return {"status": "deprovisioned"}


class ScimGroupBody(BaseModel):
    displayName: str = Field(max_length=255)
    externalId: str | None = Field(default=None, max_length=255)
    members: list[str] = Field(default_factory=list)


def _group_resource(g: ScimGroup, *, include_members: bool) -> dict[str, Any]:
    res = {"schemas": [_GROUP_SCHEMA], "id": str(g.id), "displayName": g.display_name}
    if include_members:  # excludedAttributes=members lets IdPs skip huge member lists
        res["members"] = [{"value": m} for m in (g.members or [])]
    return res


@router.get("/Groups")
async def list_groups(
    startIndex: int = Query(1, ge=1),
    count: int = Query(100, ge=1, le=500),
    filter: str | None = Query(None),                 # e.g. displayName eq "Engineers"
    excludedAttributes: str | None = Query(None),     # e.g. members
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    include_members = not _excludes_members(excludedAttributes)
    where = [ScimGroup.organization_id == p.org_id]

    parsed = _parse_eq_filter(filter)
    if parsed is not None:
        attr, value = parsed
        if attr.lower() == "displayname":
            where.append(ScimGroup.display_name == value)
        elif attr.lower() == "externalid":
            where.append(ScimGroup.external_id == value)
        else:
            raise _scim_error(status.HTTP_400_BAD_REQUEST, f"unsupported filter attribute: {attr}")

    total = int(
        await db.scalar(select(func.count()).select_from(ScimGroup).where(*where)) or 0
    )
    rows = (
        await db.scalars(
            select(ScimGroup).where(*where)
            .order_by(ScimGroup.created_at.desc(), ScimGroup.id)
            .limit(count)
            .offset(startIndex - 1)
        )
    ).all()
    return {
        "schemas": [_LIST_RESPONSE],
        "totalResults": total,
        "startIndex": startIndex,
        "itemsPerPage": len(rows),
        "Resources": [_group_resource(g, include_members=include_members) for g in rows],
    }


async def _validate_group_members(db: AsyncSession, org_id: uuid.UUID, members: list[str]) -> None:
    """Members must be user ids of THIS org — keeps the SCIM group graph consistent."""
    if not members:
        return
    try:
        ids = [uuid.UUID(m) for m in members]
    except (ValueError, TypeError):
        raise _scim_error(status.HTTP_400_BAD_REQUEST, "group members must be user ids")
    valid = set(
        (
            await db.scalars(
                select(Membership.user_id).where(
                    Membership.organization_id == org_id, Membership.user_id.in_(ids)
                )
            )
        ).all()
    )
    missing = [str(i) for i in ids if i not in valid]
    if missing:
        raise _scim_error(
            status.HTTP_400_BAD_REQUEST, f"members are not in this organization: {missing}"
        )


@router.post("/Groups", status_code=status.HTTP_201_CREATED)
async def create_group(
    body: ScimGroupBody,
    p: Principal = Depends(require_permission(Permission.MANAGE_MEMBERS)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await _validate_group_members(db, p.org_id, body.members)
    group = ScimGroup(
        organization_id=p.org_id, display_name=body.displayName,
        external_id=body.externalId, members=body.members,
    )
    db.add(group)
    await db.flush()
    return _group_resource(group, include_members=True)
