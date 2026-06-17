"""Auth + identity HTTP routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

import uuid

from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity.models import Organization, Role, User
from navis.identity.schemas import (
    LoginRequest,
    MeOut,
    OrganizationOut,
    RefreshRequest,
    SignupRequest,
    TokenPair,
    UserOut,
)
from navis.identity.service import AuthError, IdentityService
from navis.platform.database import get_session
from navis.platform.security import TokenError, decode_token

router = APIRouter(prefix="/auth", tags=["identity"])


@router.post("/signup", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
async def signup(data: SignupRequest, db: AsyncSession = Depends(get_session)) -> TokenPair:
    svc = IdentityService(db)
    try:
        user, org, _ = await svc.signup(data)
    except AuthError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    access, refresh = await svc.issue_tokens(user, org.id)
    return TokenPair(access_token=access, refresh_token=refresh)


@router.post("/login", response_model=TokenPair)
async def login(data: LoginRequest, db: AsyncSession = Depends(get_session)) -> TokenPair:
    svc = IdentityService(db)
    try:
        user = await svc.authenticate(data.email, data.password)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))
    membership = await svc.primary_membership(user)
    org_id = membership.organization_id if membership else None
    access, refresh = await svc.issue_tokens(user, org_id)
    return TokenPair(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenPair)
async def refresh(data: RefreshRequest, db: AsyncSession = Depends(get_session)) -> TokenPair:
    svc = IdentityService(db)
    try:
        payload = decode_token(data.refresh_token, expected_type="refresh")
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))

    user = await db.get(User, uuid.UUID(payload["sub"]))
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found")
    org_id = uuid.UUID(payload["org"]) if payload.get("org") else None
    try:
        access, new_refresh = await svc.rotate_refresh(data.refresh_token, user, org_id)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))
    return TokenPair(access_token=access, refresh_token=new_refresh)


@router.get("/me", response_model=MeOut)
async def me(
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> MeOut:
    org = await db.get(Organization, ctx.org_id)
    return MeOut(
        user=UserOut.model_validate(ctx.user),
        organization=OrganizationOut.model_validate(org),
        role=Role(ctx.role),
    )
