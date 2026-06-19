"""Identity service: signup, login, token refresh. Pure business logic."""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.audit import AuditAction, AuditService
from navis.identity.models import Membership, Organization, Role, Session, User
from navis.identity.org_service import OrganizationService
from navis.identity.schemas import SignupRequest
from navis.identity.tokens import TokenRevocationService
from navis.platform.config import settings
from navis.platform.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)


class AuthError(Exception):
    """Authentication/registration failure (maps to 400/401 at the edge)."""


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
        self.db.add(user)
        # Org lifecycle goes through the seam (P2 extraction prep), not inline here.
        org = await OrganizationService(self.db).create(name=data.organization_name)
        await self.db.flush()

        # First user of an org is the founder.
        membership = Membership(user_id=user.id, organization_id=org.id, role=Role.FOUNDER)
        self.db.add(membership)
        await self.db.flush()
        await AuditService(self.db).record(
            AuditAction.SIGNUP, org_id=org.id, actor_id=user.id, target_id=str(org.id)
        )
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
        # Deterministic: oldest membership wins (stable across requests).
        return await self.db.scalar(
            select(Membership)
            .where(Membership.user_id == user.id)
            .order_by(Membership.created_at.asc(), Membership.id.asc())
            .limit(1)
        )

    async def logout(self, *, jti: str | None, user_id: uuid.UUID, expires_at: datetime | None) -> None:
        """Revoke the presented access token and audit the logout."""
        revocation = TokenRevocationService(self.db)
        if jti:
            await revocation.revoke_jti(jti, user_id=user_id, expires_at=expires_at)
        await AuditService(self.db).record(
            AuditAction.LOGOUT, actor_id=user_id, target_id=str(user_id)
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
