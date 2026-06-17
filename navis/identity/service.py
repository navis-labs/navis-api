"""Identity service: signup, login, token refresh. Pure business logic."""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.models import Membership, Organization, Role, Session, User
from navis.identity.schemas import SignupRequest
from navis.platform.config import settings
from navis.platform.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)


class AuthError(Exception):
    """Authentication/registration failure (maps to 400/401 at the edge)."""


def _slugify(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "org"
    return f"{base}-{uuid.uuid4().hex[:6]}"


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class IdentityService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def signup(self, data: SignupRequest) -> tuple[User, Organization, Membership]:
        existing = await self.db.scalar(select(User).where(User.email == data.email))
        if existing:
            raise AuthError("email already registered")

        user = User(
            email=data.email,
            full_name=data.full_name,
            hashed_password=hash_password(data.password),
            is_verified=False,
        )
        org = Organization(name=data.organization_name, slug=_slugify(data.organization_name))
        self.db.add_all([user, org])
        await self.db.flush()

        # First user of an org is the founder.
        membership = Membership(user_id=user.id, organization_id=org.id, role=Role.FOUNDER)
        self.db.add(membership)
        await self.db.flush()
        return user, org, membership

    async def authenticate(self, email: str, password: str) -> User:
        user = await self.db.scalar(select(User).where(User.email == email))
        if not user or not user.hashed_password:
            raise AuthError("invalid credentials")
        if not verify_password(password, user.hashed_password):
            raise AuthError("invalid credentials")
        if not user.is_active:
            raise AuthError("account disabled")
        return user

    async def primary_membership(self, user: User) -> Membership | None:
        return await self.db.scalar(
            select(Membership).where(Membership.user_id == user.id).limit(1)
        )

    async def issue_tokens(self, user: User, org_id: uuid.UUID | None) -> tuple[str, str]:
        org_str = str(org_id) if org_id else None
        access = create_access_token(str(user.id), org_str)
        refresh = create_refresh_token(str(user.id), org_str)

        session = Session(
            user_id=user.id,
            refresh_token_hash=_hash_token(refresh),
            expires_at=datetime.now(timezone.utc)
            + timedelta(days=settings.refresh_token_expire_days),
        )
        self.db.add(session)
        await self.db.flush()
        return access, refresh

    async def rotate_refresh(self, refresh_token: str, user: User, org_id: uuid.UUID | None) -> tuple[str, str]:
        token_hash = _hash_token(refresh_token)
        session = await self.db.scalar(
            select(Session).where(
                Session.refresh_token_hash == token_hash,
                Session.revoked.is_(False),
            )
        )
        if not session:
            raise AuthError("refresh token revoked or unknown")
        session.revoked = True
        return await self.issue_tokens(user, org_id)
