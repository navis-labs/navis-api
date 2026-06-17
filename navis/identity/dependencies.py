"""FastAPI auth dependencies. Every protected route resolves a CurrentContext.

`CurrentContext` is the tenancy boundary the rest of the system relies on:
it carries the authenticated user AND the active org_id.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.models import Membership, User
from navis.platform.database import get_session
from navis.platform.security import TokenError, decode_token
from sqlalchemy import select

_bearer = HTTPBearer(auto_error=True)


@dataclass(slots=True)
class CurrentContext:
    user: User
    org_id: uuid.UUID
    role: str


async def get_current_context(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(get_session),
) -> CurrentContext:
    try:
        payload = decode_token(credentials.credentials, expected_type="access")
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))

    user_id = payload.get("sub")
    org_id = payload.get("org")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "malformed token")

    user = await db.get(User, uuid.UUID(user_id))
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found")

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

    return CurrentContext(user=user, org_id=membership.organization_id, role=membership.role.value)
