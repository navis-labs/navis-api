"""Identity service: signup, login, token refresh. Pure business logic."""
from __future__ import annotations

import hashlib
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
from navis.organization.contract import OnboardingStage, stage_index
from navis.organization.repository import OrganizationRepository
from navis.platform.config import settings
from navis.platform.monitoring import get_request_id
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
        user = await self.db.scalar(select(User).where(User.email == normalize_email(email)))
        if not user or not user.hashed_password:
            raise AuthError("invalid credentials")
        if not verify_password(password, user.hashed_password):
            raise AuthError("invalid credentials")
        if not user.is_active:
            raise AuthError("account disabled")
        return user

    async def google_signin(
        self,
        *,
        email: str,
        google_sub: str,
        full_name: str | None,
    ) -> tuple[User, uuid.UUID | None, bool, bool]:
        """Find-or-create a user identified by their Google account.

        Returns a 4-tuple:
          (user, org_id, is_new_user, onboarding_completed)

        Behavior:
          • Existing user with google_sub match  → fast path, no writes.
          • Existing user with email match        → backfill google_sub.
          • Unknown email                         → create User + Org + Membership.

        `onboarding_completed` is True when the org's onboarding stage is at
        or beyond PROFILE_COMPLETE (i.e. the org form was already submitted).
        """
        email = normalize_email(email)
        is_new_user = False

        # 1. Look up by google_sub first (most specific — immutable Google identifier).
        user = await self.db.scalar(
            select(User).where(User.google_sub == google_sub)
        )

        # 2. Fall back to email lookup (covers pre-invited users + password-signup users).
        if user is None:
            user = await self.db.scalar(
                select(User).where(User.email == email)
            )

        if user is not None:
            # Backfill google_sub so future lookups hit the fast path.
            if user.google_sub != google_sub:
                user.google_sub = google_sub
            if not user.is_active:
                raise AuthError("account disabled")
            # Backfill full_name if the user has none (e.g. pre-invited stub).
            if not user.full_name and full_name:
                user.full_name = full_name
            await self.db.flush()
        else:
            # Brand-new user — create User + Org + Membership atomically.
            is_new_user = True
            try:
                async with self.db.begin_nested():  # SAVEPOINT: rolls back only on conflict
                    user = User(
                        email=email,
                        full_name=full_name,
                        google_sub=google_sub,
                        is_verified=True,   # Google already verified the email
                        is_active=True,
                    )
                    self.db.add(user)
                    await self.db.flush()
            except IntegrityError:
                raise AuthError("email already registered")

            # Create the org. Slug collisions are handled by OrganizationService's retry loop.
            org_name = full_name or email.split("@")[0]
            org = await OrganizationService(self.db).create(name=org_name)
            await self.db.flush()

            membership = Membership(user_id=user.id, organization_id=org.id, role=Role.FOUNDER)
            self.db.add(membership)
            await self.db.flush()

            await AuditService(self.db).record(
                AuditAction.SIGNUP,
                org_id=org.id,
                actor_id=user.id,
                target_id=str(org.id),
                metadata={"method": "google_sso"},
            )

        # Resolve org membership for token issuance.
        membership = await self.primary_membership(user)
        org_id = membership.organization_id if membership else None

        # Determine if onboarding has been completed (PROFILE_COMPLETE or beyond).
        onboarding_completed = False
        if org_id is not None:
            onboarding_row = await OrganizationRepository(self.db).get_onboarding(org_id)
            if onboarding_row is not None:
                current_stage = OnboardingStage(onboarding_row.stage)
                onboarding_completed = (
                    stage_index(current_stage) >= stage_index(OnboardingStage.PROFILE_COMPLETE)
                )

        await AuditService(self.db).record(
            AuditAction.SSO_LOGIN,
            org_id=org_id,
            actor_id=user.id,
            target_id=str(user.id),
            metadata={"method": "google", "request_id": get_request_id()},
        )

        return user, org_id, is_new_user, onboarding_completed

    async def record_login(self, user: User, org_id: uuid.UUID | None, *, method: str = "password") -> None:
        """Audit a SUCCESSFUL login exactly once (IDENTITY-002). Called only from the
        login route after authentication succeeds — never on failure, and never from the
        refresh flow (which mints tokens without re-authenticating). Carries user_id,
        org_id, the request_id (if any), the auth method, and an auto timestamp."""
        await AuditService(self.db).record(
            AuditAction.LOGIN,
            org_id=org_id,
            actor_id=user.id,
            target_id=str(user.id),
            metadata={"method": method, "request_id": get_request_id()},
        )

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
