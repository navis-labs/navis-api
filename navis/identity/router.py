"""Auth + identity HTTP routes."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import uuid

from navis.identity.authz import Principal, get_principal
from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity.google_auth import GoogleAuthError, GoogleAuthService
from navis.identity.microsoft_auth import MicrosoftAuthError, MicrosoftAuthService
from navis.identity.models import Membership, Organization, User
from navis.identity.schemas import (
    GoogleAuthRequest,
    MicrosoftAuthRequest,
    LoginRequest,
    MeOut,
    OrganizationOut,
    OrgMembershipOut,
    RefreshRequest,
    SignupRequest,
    TokenPair,
    UserOut,
)
from navis.identity.service import AuthError, IdentityService
from navis.identity.throttle import LoginThrottle
from navis.identity.tokens import TokenRevocationService, token_predates_bulk_revoke
from navis.platform.cache import redis_client
from navis.platform.database import get_session
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.security import TokenError, decode_token
from navis.organization.contract import OnboardingStage, stage_index
from navis.organization.repository import OrganizationRepository

router = APIRouter(prefix="/auth", tags=["identity"])

# One opaque message for EVERY auth failure (bad password / unknown email / disabled /
# locked) so the endpoint never enables account enumeration.
_GENERIC_AUTH_ERROR = "invalid email or password"


def _client_ip(request: Request) -> str:
    # Behind a trusted proxy/LB the first X-Forwarded-For hop is the client; fall back to
    # the direct peer. The per-email counter is the primary defense if XFF is spoofable.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/signup", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
async def signup(data: SignupRequest, db: AsyncSession = Depends(get_session)) -> TokenPair:
    svc = IdentityService(db)
    try:
        user, org, _ = await svc.signup(data)
    except AuthError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    access, refresh = await svc.issue_tokens(user, org.id)
    # Commit so the org row is visible, THEN announce it. Organization is provisioned
    # by an orchestration subscriber to this event — Identity never imports Organization.
    await db.commit()
    await event_bus.publish(
        DomainEvent(type=EventType.ORG_CREATED, org_id=org.id, source="identity",
                    payload={"organization_id": str(org.id), "user_id": str(user.id)})
    )
    return TokenPair(access_token=access, refresh_token=refresh)


@router.post("/login", response_model=TokenPair)
async def login(data: LoginRequest, request: Request, db: AsyncSession = Depends(get_session)) -> TokenPair:
    svc = IdentityService(db)
    throttle = LoginThrottle(redis_client)
    ip = _client_ip(request)

    # IDENTITY-003: brute-force gate BEFORE any password work. Locked -> generic 401.
    if await throttle.is_locked(data.email, ip):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _GENERIC_AUTH_ERROR)

    try:
        user = await svc.authenticate(data.email, data.password)
    except AuthError:
        await throttle.record_failure(data.email, ip)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _GENERIC_AUTH_ERROR)

    # Success: clear strike counters, resolve org, issue tokens, audit exactly once.
    await throttle.reset(data.email, ip)
    membership = await svc.primary_membership(user)
    org_id = membership.organization_id if membership else None
    access, refresh = await svc.issue_tokens(user, org_id)
    await svc.record_login(user, org_id)  # IDENTITY-002 — login route only; refresh never calls this
    return TokenPair(access_token=access, refresh_token=refresh)


@router.post("/google", response_model=TokenPair)
async def google_login(
    data: GoogleAuthRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> TokenPair:
    """Google SSO login — accepts an OAuth2 authorization code OR a raw id_token.

    Flow:
      1. IP throttle gate (before any crypto / Google round-trip).
      2. Verify the Google credential (code exchange or id_token verification).
      3. Find-or-create the user + org (IdentityService.google_signin).
      4. Issue a JWT access + refresh token pair.
      5. Emit ORG_CREATED for new users so orchestration bootstraps the org.
      6. Return TokenPair with is_new_user + onboarding_completed flags.

    Frontend routing logic:
      is_new_user=True or onboarding_completed=False  →  /onboarding
      onboarding_completed=True                        →  /dashboard
    """
    throttle = LoginThrottle(redis_client)
    ip = _client_ip(request)

    # SSO gate: we don't know the email until AFTER Google verifies the credential,
    # so the IP counter is the only thing we can check up front. Same generic 401
    # the password path uses — no enumeration, no lock-state leak.
    if await throttle.is_ip_locked(ip):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _GENERIC_AUTH_ERROR)

    google_svc = GoogleAuthService()
    try:
        if data.code:
            identity = await google_svc.verify_code(data.code)
        else:
            identity = await google_svc.verify_id_token(data.id_token)  # type: ignore[arg-type]
    except GoogleAuthError as exc:
        # Bogus code / expired token / Google unreachable — count it as a failure so
        # a flood can't burn Google's token-endpoint quota or our CPU on JWT verify.
        await throttle.record_ip_failure(ip)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))

    svc = IdentityService(db)
    try:
        user, org_id, is_new_user, onboarding_completed = await svc.google_signin(
            email=identity.email,
            google_sub=identity.google_sub,
            full_name=identity.full_name,
        )
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))

    access, refresh = await svc.issue_tokens(user, org_id)
    await db.commit()

    # Emit ORG_CREATED for brand-new users so the orchestration layer bootstraps
    # lifecycle/onboarding/entitlements. Emit AFTER commit — the org row must be
    # durable before subscribers try to read it (commit-before-publish pattern).
    if is_new_user and org_id is not None:
        await event_bus.publish(
            DomainEvent(
                type=EventType.ORG_CREATED,
                org_id=org_id,
                source="identity",
                payload={"organization_id": str(org_id), "user_id": str(user.id)},
            )
        )

    return TokenPair(
        access_token=access,
        refresh_token=refresh,
        is_new_user=is_new_user,
        onboarding_completed=onboarding_completed,
    )


@router.post("/microsoft", response_model=TokenPair)
async def microsoft_login(
    data: MicrosoftAuthRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> TokenPair:
    """Microsoft (Entra ID) SSO login — authorization code only.

    The exact mirror of `POST /auth/google`: same TokenPair, same routing
    flags, same throttle gate. Two things differ, and both are deliberate.

    IDENTITY SCOPES ONLY (`openid profile email offline_access User.Read`).
    The connector's mailbox consent is a separate app-registration flow behind
    a separate feature flag — see navis.signals.microsoft_router. Signing in
    and handing over an inbox are two decisions a founder makes at two
    different moments, and merging them would ask for mailbox access at the
    login screen.

    ACCOUNT LINKING IS GATED ON VERIFICATION. Entra's `email` claim is
    attacker-controlled for personal Microsoft accounts, so
    `IdentityService.microsoft_signin` links to an existing account only when
    Entra asserts domain ownership via `xms_edov`. See that method for the full
    reasoning.

    Frontend routing logic (unchanged from the Google route):
      is_new_user=True or onboarding_completed=False  ->  /onboarding
      onboarding_completed=True                        ->  /dashboard
    """
    throttle = LoginThrottle(redis_client)
    ip = _client_ip(request)

    # SSO gate: the email is unknown until Entra verifies the credential, so
    # the IP counter is the only thing checkable up front. Same generic 401 the
    # password path uses — no enumeration, no lock-state leak.
    if await throttle.is_ip_locked(ip):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _GENERIC_AUTH_ERROR)

    ms_svc = MicrosoftAuthService()
    if not ms_svc.configured:
        # A specific 503 rather than a generic 401: this is our misconfiguration,
        # not a bad credential, and telling the user "sign-in failed" would send
        # them retrying something that cannot work.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Microsoft sign-in is not configured on this deployment.",
        )

    try:
        identity = await ms_svc.verify_code(data.code)
    except MicrosoftAuthError as exc:
        # Bogus/expired code or Entra unreachable — count it as a failure so a
        # flood cannot burn Entra's token-endpoint quota or our CPU.
        await throttle.record_ip_failure(ip)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))

    svc = IdentityService(db)
    try:
        user, org_id, is_new_user, onboarding_completed = await svc.microsoft_signin(
            email=identity.email,
            microsoft_sub=identity.microsoft_sub,
            tenant_id=identity.tenant_id,
            full_name=identity.full_name,
            email_verified=identity.email_verified,
        )
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))

    access, refresh = await svc.issue_tokens(user, org_id)
    await db.commit()

    # Emit ORG_CREATED for brand-new users so orchestration bootstraps
    # lifecycle/onboarding/entitlements. AFTER commit — the org row must be
    # durable before subscribers read it (commit-before-publish).
    if is_new_user and org_id is not None:
        await event_bus.publish(
            DomainEvent(
                type=EventType.ORG_CREATED,
                org_id=org_id,
                source="identity",
                payload={"organization_id": str(org_id), "user_id": str(user.id)},
            )
        )

    return TokenPair(
        access_token=access,
        refresh_token=refresh,
        is_new_user=is_new_user,
        onboarding_completed=onboarding_completed,
    )


@router.post("/refresh", response_model=TokenPair)
async def refresh(data: RefreshRequest, db: AsyncSession = Depends(get_session)) -> TokenPair:
    svc = IdentityService(db)
    try:
        payload = decode_token(data.refresh_token, expected_type="refresh")
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))

    user = await db.get(User, uuid.UUID(payload["sub"]))
    # Refresh MUST honor revocation too — otherwise a deactivated/removed user (or a
    # future "revoke my sessions" event) could mint a fresh access token whose new iat
    # sails past the watermark. Mirrors the checks in authz.get_principal.
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found or inactive")
    if await TokenRevocationService(db).is_jti_revoked(payload.get("jti")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token revoked")
    if token_predates_bulk_revoke(payload, user):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token revoked")
    org_id = uuid.UUID(payload["org"]) if payload.get("org") else None
    try:
        access, new_refresh = await svc.rotate_refresh(data.refresh_token, user, org_id)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))
    return TokenPair(access_token=access, refresh_token=new_refresh)


_logout_bearer = HTTPBearer(auto_error=True)


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(_logout_bearer),
    db: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    try:
        payload = decode_token(credentials.credentials, expected_type="access")
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))
    exp = payload.get("exp")
    expires_at = datetime.fromtimestamp(int(exp), tz=timezone.utc) if exp else None
    await IdentityService(db).logout(
        jti=payload.get("jti"), user_id=uuid.UUID(payload["sub"]),
        expires_at=expires_at, sid=payload.get("sid"),
    )
    return {"status": "logged_out"}


@router.get("/me", response_model=MeOut)
async def me(
    ctx: CurrentContext = Depends(get_current_context),
    principal: Principal = Depends(get_principal),
    db: AsyncSession = Depends(get_session),
) -> MeOut:
    """The caller's identity, active organization, and EFFECTIVE PERMISSIONS.

    This is the frontend's authorization READ contract. It reports what the
    caller may do; it does not decide it. Enforcement stays where it has always
    been — `require_permission` on each protected route — and this endpoint
    simply publishes the same resolved `Principal.permissions` the gate uses, so
    the UI stops carrying a second, drifting copy of the permission table.

    WHAT "CURRENT ORGANIZATION" MEANS, AND WHY IT IS THE ONLY ONE DESCRIBED
    ---------------------------------------------------------------------
    `organization`, `role` and `permissions` all describe ONE tenant: the one
    this access token authenticated into. That tenant is `Principal.org_id`,
    which is not the JWT's `org` claim — `authz._principal_from_jwt` uses the
    claim only to LOOK UP a membership row and then takes
    `membership.organization_id` as the answer. A token naming an org the user
    has no seat in is rejected 403 and never reaches this route, so the org
    reported here is always one the database agrees the caller belongs to.

    `organizations` lists every seat the user holds, current one flagged. It
    exists so a client can distinguish "this user has one organization" from
    "this response only shows one" — a distinction the previous shape erased.
    It deliberately carries NO role and NO permissions per entry: authorization
    is published for the active tenant only, so nothing here can be assembled
    into an answer about another tenant.

    MULTI-ORG SWITCHING IS NOT IMPLEMENTED AND IS NOT IMPLIED. There is no
    switch endpoint and no way to re-scope an existing token; changing the
    active organization means authenticating into it. The full multi-org
    experience (an active-org token mechanism, an org picker, per-org
    onboarding) is Phase 7 work. This list is a read, and nothing more.

    Machine principals (API key / service account) are rejected 401 by
    `get_current_context`, unchanged: /auth/me describes a human session.
    """
    # `get_current_context` already depends on `get_principal`, and FastAPI caches a
    # dependency per request, so asking for both resolves the principal ONCE — no
    # second trip through the JWT decode, revocation check, membership lookup and
    # grant query. `ctx` is what enforces the human-only contract; `principal` is
    # what carries the permissions this response exists to publish.
    org = await db.get(Organization, ctx.org_id)

    # Every seat this user holds. Two queries, not a join: the fan-out is a
    # handful of rows (one per organization the user belongs to) and the ORM
    # models are what the response schema validates from.
    memberships = (
        await db.scalars(select(Membership).where(Membership.user_id == ctx.user.id))
    ).all()
    member_org_ids = [m.organization_id for m in memberships]
    member_orgs = (
        (await db.scalars(select(Organization).where(Organization.id.in_(member_org_ids)))).all()
        if member_org_ids
        else []
    )

    # Determine onboarding_completed from the org's onboarding stage.
    onboarding_completed = False
    if ctx.org_id is not None:
        onboarding_row = await OrganizationRepository(db).get_onboarding(ctx.org_id)
        if onboarding_row is not None:
            current_stage = OnboardingStage(onboarding_row.stage)
            onboarding_completed = (
                stage_index(current_stage) >= stage_index(OnboardingStage.PROFILE_COMPLETE)
            )

    return MeOut(
        user=UserOut.model_validate(ctx.user),
        organization=OrganizationOut.model_validate(org),
        # Passed as the raw string; MeOut resolves it and reports an unusable
        # role as null rather than raising (see MeOut._tolerate_unusable_role).
        role=principal.role,
        # The resolved set the gate itself checks — code defaults PLUS this org's
        # `role_permissions` grants. Scoped to `principal.org_id` by construction:
        # `grants_for` filters on that org id, so a grant made in another tenant
        # cannot appear here.
        permissions=sorted(principal.permissions),
        organizations=[
            OrgMembershipOut(
                id=o.id, name=o.name, slug=o.slug, is_current=(o.id == ctx.org_id)
            )
            for o in member_orgs
        ],
        onboarding_completed=onboarding_completed,
    )
