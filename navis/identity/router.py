"""Auth + identity HTTP routes."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

import uuid

from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity.google_auth import GoogleAuthError, GoogleAuthService
from navis.identity.models import Organization, Role, User
from navis.identity.schemas import (
    GoogleAuthRequest,
    LoginRequest,
    MeOut,
    OrganizationOut,
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
    db: AsyncSession = Depends(get_session),
) -> MeOut:
    org = await db.get(Organization, ctx.org_id)
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
        role=Role(ctx.role),
        onboarding_completed=onboarding_completed,
    )
