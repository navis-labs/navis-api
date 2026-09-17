"""Identity service: signup, login, token refresh. Pure business logic."""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.audit import AuditAction, AuditService
from navis.identity.members import normalize_email
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
        email = normalize_email(data.email)  # canonical identity key (see members.normalize_email)
        existing = await self.db.scalar(select(User).where(User.email == email))
        if existing:
            raise AuthError("email already registered")  # fast path; not authoritative

        user = User(
            email=email,
            full_name=data.full_name,
            hashed_password=hash_password(data.password),
            is_verified=False,
        )
        # Race-safe: the pre-check above can lose a race; the UNIQUE(email)/lower(email)
        # constraint is the authority. Insert inside a SAVEPOINT so a concurrent
        # duplicate rolls back ONLY the user insert and surfaces as a clean 400.
        try:
            async with self.db.begin_nested():
                self.db.add(user)
                await self.db.flush()
        except IntegrityError:
            raise AuthError("email already registered")
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
        from navis.identity.ratelimit import TooManyLoginAttempts, login_rate_limiter

        email_normalized = normalize_email(email)

        # Brute-force guard: fail fast before touching the DB if the account is locked.
        # Raises TooManyLoginAttempts (maps to 429 at the router) if the limit is reached.
        await login_rate_limiter.check(email_normalized)

        user = await self.db.scalar(select(User).where(User.email == email_normalized))
        if not user or not user.hashed_password:
            # Still record a failure so an attacker can't enumerate existing emails by
            # observing different lock-out behaviour for known vs unknown addresses.
            await login_rate_limiter.record_failure(email_normalized)
            raise AuthError("invalid credentials")
        if not verify_password(password, user.hashed_password):
            await login_rate_limiter.record_failure(email_normalized)
            raise AuthError("invalid credentials")
        if not user.is_active:
            raise AuthError("account disabled")

        # Success: reset the strike counter so a returning user with a fresh password
        # doesn't hit a stale lockout.
        await login_rate_limiter.reset(email_normalized)
        return user

    async def primary_membership(self, user: User) -> Membership | None:
        # Deterministic: oldest membership wins (stable across requests).
        return await self.db.scalar(
            select(Membership)
            .where(Membership.user_id == user.id)
            .order_by(Membership.created_at.asc(), Membership.id.asc())
            .limit(1)
        )

    async def logout(
        self, *, jti: str | None, user_id: uuid.UUID, expires_at: datetime | None, sid: str | None = None
    ) -> None:
        """End the session: deny the access jti AND revoke the bound refresh Session
        (so /refresh can't mint a new token after logout). `sid` ties the access token
        to its session; tokens issued before sid existed fall back to jti-only revoke."""
        revocation = TokenRevocationService(self.db)
        if jti:
            await revocation.revoke_jti(jti, user_id=user_id, expires_at=expires_at)
        if sid:
            session = await self.db.get(Session, uuid.UUID(sid))
            if session is not None and session.user_id == user_id:
                session.revoked = True
        await AuditService(self.db).record(
            AuditAction.LOGOUT, actor_id=user_id, target_id=str(user_id)
        )

    async def issue_tokens(self, user: User, org_id: uuid.UUID | None) -> tuple[str, str]:
        org_str = str(org_id) if org_id else None
        # Assign the session id up front (Python-side) so its sid can be embedded in
        # both tokens BEFORE the single insert — no empty-hash placeholder is ever
        # persisted, so the UNIQUE(refresh_token_hash) constraint is safe.
        session = Session(
            id=uuid.uuid4(),
            user_id=user.id,
            refresh_token_hash="",
            revoked=False,
            expires_at=datetime.now(timezone.utc)
            + timedelta(days=settings.refresh_token_expire_days),
        )
        sid = str(session.id)
        access = create_access_token(str(user.id), org_str, sid=sid)
        refresh = create_refresh_token(str(user.id), org_str, sid=sid)
        session.refresh_token_hash = _hash_token(refresh)  # real hash before insert
        self.db.add(session)
        await self.db.flush()
        return access, refresh

    async def rotate_refresh(self, refresh_token: str, user: User, org_id: uuid.UUID | None) -> tuple[str, str]:
        token_hash = _hash_token(refresh_token)
        # `FOR UPDATE` makes rotation a one-time-use atomic claim: two concurrent
        # refreshes of the same token serialize — the second re-reads the row (now
        # revoked) and is filtered out, so it can't mint a parallel refresh chain
        # (classic refresh-token replay race).
        session = await self.db.scalar(
            select(Session)
            .where(
                Session.refresh_token_hash == token_hash,
                Session.revoked.is_(False),
            )
            .with_for_update()
        )
        if session is None:
            # Reuse detection: a hash that maps to an ALREADY-revoked session means an
            # already-rotated (likely stolen) refresh token is being replayed. Treat as
            # compromise — kill the user's whole session family + record a security event.
            reused = await self.db.scalar(
                select(Session).where(
                    Session.refresh_token_hash == token_hash, Session.revoked.is_(True)
                )
            )
            if reused is not None:
                await TokenRevocationService(self.db).revoke_all_for_user(reused.user_id)
                await AuditService(self.db).record(
                    AuditAction.REFRESH_REUSE_DETECTED,
                    actor_id=reused.user_id, target_id=str(reused.id),
                )
            raise AuthError("refresh token revoked or unknown")
        # Defense in depth at the trust boundary: the session MUST belong to the
        # presented user — never trust the router to have bound them correctly.
        if session.user_id != user.id:
            raise AuthError("refresh token does not belong to this user")
        # Service-layer expiry enforcement — never rely solely on the JWT exp check.
        if session.expires_at <= datetime.now(timezone.utc):
            raise AuthError("refresh token expired")
        session.revoked = True
        return await self.issue_tokens(user, org_id)

    async def purge_expired_sessions(self, *, now: datetime | None = None) -> int:
        """Reaper for the sessions table (served by ix_sessions_expires_at). Deletes
        sessions past expiry; a refresh token can't be used after expiry anyway."""
        cutoff = now or datetime.now(timezone.utc)
        result = await self.db.execute(delete(Session).where(Session.expires_at < cutoff))
        return result.rowcount or 0
