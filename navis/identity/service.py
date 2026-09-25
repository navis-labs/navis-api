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
from navis.identity.models import (
    DeveloperRole,
    Membership,
    Organization,
    OrganizationKind,
    Role,
    Session,
    User,
)
from navis.identity.org_service import OrgLifecycleSeam
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


class NoSuchAccountError(AuthError):
    """A verified identity that Navis has never seen, on a call that was told
    not to create one (maps to 404 at the edge).

    NOT AN AUTHENTICATION FAILURE, which is why it is a distinct type and gets
    a distinct status. Google verified this person; there is simply no Navis
    account behind the address yet. Reporting it as the generic 401 would tell
    them their credentials were wrong, which is both false and unactionable —
    the action they need is to choose what kind of workspace to create.

    It does not enable account enumeration the way a 404 on the PASSWORD path
    would: reaching this line requires a Google credential the caller has
    already proved they control, so the only account whose existence it reveals
    is their own.
    """


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


#: Ceiling on a derived workspace name, matching `organizations.name`.
_MAX_ORG_NAME = 255


def _personal_workspace_name(full_name: str | None, email: str) -> str:
    """What to call a workspace nobody named.

    The person's own name when they gave one, otherwise the local part of their
    address — which is the only other thing signup knows about them. Never the
    full address: the name is rendered in the console and in an agent's
    `/developer/me`, and an email is more than a label needs to carry.

    Always non-empty, because `OrgLifecycleSeam.create` rejects a blank name and
    a signup must not fail for lack of a display string. An address with no
    local part cannot reach here — `EmailStr` has already validated it — but the
    fallback is kept so this function is total rather than nearly total.
    """
    candidate = (full_name or "").strip() or email.split("@", 1)[0].strip()
    return (candidate or "Personal workspace")[:_MAX_ORG_NAME]


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
        #
        # NO ORGANIZATION NAME MEANS A PERSONAL WORKSPACE (P0-4). A solo
        # developer has no company to name, and requiring one produced a tenant
        # that claimed to be a company and was not. The workspace is named after
        # the person because that is what it is; the name is a LABEL and carries
        # no authority — the tenant is still identified by its id everywhere.
        personal = data.organization_name is None
        org = await OrgLifecycleSeam(self.db).create(
            name=_personal_workspace_name(data.full_name, email)
            if personal
            else data.organization_name,
            kind=OrganizationKind.PERSONAL if personal else OrganizationKind.COMPANY,
        )
        await self.db.flush()

        # First user of an org is the founder — on BOTH surfaces (P1-6).
        #
        # WHY THE CREATOR IS THE DEVELOPER OWNER, AND WHY THAT IS NOT "SILENT
        # INHERITANCE". Developer access is explicit everywhere else: it comes
        # from `memberships.developer_role`, it defaults to NULL on an invite,
        # and an Application administrator cannot grant it to themselves
        # (`MemberService._ensure_grantable`). But SOMEBODY has to be able to
        # grant it first, and the only person who exists when an organization is
        # created is the person creating it. Leaving them NULL would produce a
        # tenant whose Developer surface no one in the world could ever open —
        # not even its owner — which is the same unrecoverable state the
        # last-founder guard exists to prevent.
        #
        # This is also what makes the Developer-solo and Developer-first-company
        # flows work with no signup change at all: whoever creates the workspace
        # can immediately register an agent and call the Developer API, and never
        # touches Application onboarding to do it.
        membership = Membership(
            user_id=user.id,
            organization_id=org.id,
            role=Role.FOUNDER,
            developer_role=DeveloperRole.OWNER.value,
        )
        self.db.add(membership)
        await self.db.flush()
        await AuditService(self.db).record(
            AuditAction.SIGNUP, org_id=org.id, actor_id=user.id, target_id=str(org.id),
            metadata={"role": Role.FOUNDER.value, "developer_role": DeveloperRole.OWNER.value},
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
        workspace_kind: OrganizationKind | None = None,
        organization_name: str | None = None,
        create_workspace: bool = True,
    ) -> tuple[User, uuid.UUID | None, bool, bool]:
        """Find-or-create a user identified by their Google account.

        Returns a 4-tuple:
          (user, org_id, is_new_user, onboarding_completed)

        Behavior:
          • Existing user with google_sub match  → fast path, no writes.
          • Existing user with email match        → backfill google_sub.
          • Unknown email, create_workspace=True  → create User + Org + Membership.
          • Unknown email, create_workspace=False → NoSuchAccountError, no writes.

        `create_workspace=False` IS WHAT MAKES A SIGN-IN A SIGN-IN (P0-8).
        Authentication proves who somebody is; it is not a request to found a
        company. With this False the unknown-identity branch below is not
        reached at all, so a person arriving at the sign-in page cannot end up
        owning a tenant they were never asked about — which is exactly how
        three of the four organizations in production came to be `company`
        rows named after a person, classified by a default nobody chose.

        IT DOES NOT AFFECT AN INVITED PERSON. `MemberService.invite` writes a
        real `users` row, so an invitee is found by the email lookup below on
        their very first Google sign-in and resolves to the organization that
        invited them. The refusal reaches only an identity with no user row
        and therefore no membership and no invitation.

        `onboarding_completed` is True when the org's onboarding stage is at
        or beyond PROFILE_COMPLETE (i.e. the org form was already submitted).

        THE WORKSPACE ARGUMENTS ARE CONSULTED ONLY ON THE CREATE BRANCH, AND
        THAT IS THE LOAD-BEARING PART. A Google account that already holds a
        membership resolves to the organization it already belongs to — the
        `primary_membership` read at the bottom — and neither argument is
        reachable from there. So a console that sends the same body for
        sign-in and sign-up (which it must, because one Google button cannot
        know which it will turn out to be) can never create a second tenant
        for somebody who already has one, and can never rename the one they
        have.

        DEFAULTS TO COMPANY when the caller says nothing, which is what this
        method has always done. The alternative — inferring `personal` from
        the absence of a name, as the password path does — would silently
        reclassify every existing Google signup, and `0051_surface_rbac_columns`
        already declined to infer product meaning from an artefact of a signup
        flow. Intent is honoured when it is stated and guessed never.
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
            # A verified identity with no account. Whether that is a sign-up or
            # a dead end is the CALLER's decision, and it has to be made before
            # anything is written — once the organization exists there is no
            # in-product way to ask the question retroactively.
            if not create_workspace:
                raise NoSuchAccountError("no Navis account exists for this Google identity")

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

            # Create the org. Slug collisions are handled by OrgLifecycleSeam's retry loop.
            #
            # COMPANY UNLESS THE CALLER SAID OTHERWISE (P0-5). This flow used to
            # create a company unconditionally, because it never asked: a founder
            # bringing their company to the Application and a solo developer with
            # no company reached this line identically, and the result was a
            # tenant named after a person that claimed to be a company.
            #
            # The fix is to let intent be STATED, not to start inferring it. An
            # absent `workspace_kind` still produces a company, so every existing
            # caller is unchanged and no existing workspace is reclassified;
            # `personal` produces the same one-person space the password path
            # creates when `organization_name` is omitted. Nothing here looks at
            # the email domain — two people at the same company signing up on
            # their own get two workspaces, which is the truthful answer until
            # one of them invites the other.
            kind = workspace_kind or OrganizationKind.COMPANY
            org = await OrgLifecycleSeam(self.db).create(
                # A personal workspace is named after its owner; a company is
                # named by whoever created it, falling back to the same label
                # because it is the only other thing Google told us.
                name=(
                    organization_name.strip()
                    if kind is OrganizationKind.COMPANY and organization_name
                    else _personal_workspace_name(full_name, email)
                ),
                kind=kind,
            )
            await self.db.flush()

            # Founder on both surfaces, for the reason given in `signup` above:
            # the workspace creator is the only person who can bootstrap
            # Developer access, so leaving them NULL would make the surface
            # permanently unreachable for this tenant.
            membership = Membership(
                user_id=user.id,
                organization_id=org.id,
                role=Role.FOUNDER,
                developer_role=DeveloperRole.OWNER.value,
            )
            self.db.add(membership)
            await self.db.flush()

            await AuditService(self.db).record(
                AuditAction.SIGNUP,
                org_id=org.id,
                actor_id=user.id,
                target_id=str(org.id),
                # `workspace_kind` records what was CREATED, so an audit can tell
                # a stated choice from the default without re-reading the org row
                # — which may have been renamed since.
                metadata={
                    "method": "google_sso",
                    "workspace_kind": kind.value,
                    "role": Role.FOUNDER.value,
                    "developer_role": DeveloperRole.OWNER.value,
                },
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
