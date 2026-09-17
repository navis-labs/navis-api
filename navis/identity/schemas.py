"""Pydantic request/response schemas for identity."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from navis.identity.models import Role
from navis.platform.config import settings

# Bounds on untrusted input — reject oversized payloads at the edge before they reach
# the DB/bcrypt/JWT parser (cheap DoS protection).
_MAX_NAME = 255
_MAX_PASSWORD = 128
_MAX_TOKEN = 4096  # a normal JWT is < 2KB; anything larger is abuse

# A tiny blocklist of the most-common weak passwords. NOTE: this is deliberately a
# blocklist, NOT composition rules (upper/lower/digit) — NIST 800-63B advises length
# + breach/common-password screening over composition rules (which just produce
# "Password1!"). The production upgrade is a full breached-password check (HIBP).
_WEAK_PASSWORDS = frozenset({
    "password", "12345678", "123456789", "qwertyui", "qwerty123",
    "11111111", "password1", "iloveyou", "admin123", "letmein1",
})


def _normalize_email(value: str) -> str:
    return value.strip().lower()


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=_MAX_PASSWORD)
    full_name: str | None = Field(default=None, max_length=_MAX_NAME)
    organization_name: str = Field(min_length=2, max_length=_MAX_NAME)

    # Normalize at the schema boundary so EVERY path that builds a SignupRequest gets
    # one canonical identity key (the service layer normalizes too — defense in depth).
    @field_validator("email")
    @classmethod
    def _canon_email(cls, v: str) -> str:
        return _normalize_email(v)

    @field_validator("password")
    @classmethod
    def _reject_weak_password(cls, v: str) -> str:
        if v.lower() in _WEAK_PASSWORDS or len(set(v)) == 1:
            raise ValueError("password is too common or weak")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(max_length=_MAX_PASSWORD)

    @field_validator("email")
    @classmethod
    def _canon_email(cls, v: str) -> str:
        return _normalize_email(v)


class GoogleAuthRequest(BaseModel):
    """Body for POST /auth/google.

    The frontend uses the OAuth2 authorization-code flow (preferred), which
    returns a `code` from `google.accounts.oauth2.initCodeClient`. The legacy
    id_token path is kept for backward compatibility.
    Exactly one of `code` or `id_token` must be present.
    """

    code: str | None = Field(None, description="OAuth2 authorization code (preferred)")
    id_token: str | None = Field(None, description="Google id_token (legacy fallback)")

    @model_validator(mode="after")
    def _require_exactly_one(self) -> "GoogleAuthRequest":
        if not self.code and not self.id_token:
            raise ValueError("provide either 'code' or 'id_token'")
        if self.code and self.id_token:
            raise ValueError("provide either 'code' or 'id_token', not both")
        return self


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    # Access-token lifetime in seconds, so clients (web/mobile/agents) needn't decode
    # the JWT to schedule a refresh.
    expires_in: int = Field(default_factory=lambda: settings.access_token_expire_minutes * 60)
    # Set by the Google SSO route so the frontend knows where to redirect:
    #   is_new_user=True  → /onboarding  (first-time signup via Google)
    #   is_new_user=False, onboarding_completed=False → /onboarding (started but not done)
    #   is_new_user=False, onboarding_completed=True  → /dashboard
    is_new_user: bool = False
    onboarding_completed: bool = False


class RefreshRequest(BaseModel):
    refresh_token: str = Field(max_length=_MAX_TOKEN)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    full_name: str | None
    is_active: bool
    is_verified: bool
    created_at: datetime


class OrganizationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str


class OrgMembershipOut(BaseModel):
    """One organization this user holds a seat in.

    Carries IDENTITY ONLY — no role, no permissions. That omission is the point:
    permissions are published for the ACTIVE organization and nowhere else, so a
    client physically cannot assemble an authorization answer for a tenant it is
    not currently authenticated into. See `MeOut.organizations`.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str
    # True for the organization this access token authenticated into — the one
    # `MeOut.organization`, `MeOut.role` and `MeOut.permissions` all describe.
    is_current: bool = False


class MeOut(BaseModel):
    """The authenticated caller's identity AND their effective authorization state.

    `permissions` is the whole reason this response is not just `role`. A client
    that knows only the role has to own a copy of the backend's role->permission
    table to decide what to render, and that copy is wrong in two ways it cannot
    detect: it misses per-org `role_permissions` grants entirely (they exist only
    in the database), and it drifts the moment the backend's defaults change.
    This field is the resolved `Principal.permissions` — defaults AND org grants,
    computed by the same code that enforces them — so there is exactly one
    permission table in the system and the UI reads it rather than reimplements it.

    It remains a PRESENTATION input. Every protected route re-checks the same
    permission server-side; holding a string here has never been what grants
    access.
    """

    user: UserOut
    organization: OrganizationOut
    # Nullable BY CONTRACT. A membership whose role is missing (pre-0043 data) or
    # whose stored value this build does not know (a role added to the enum ahead
    # of the code) is reported as `null` with an empty `permissions` list, rather
    # than crashing the one endpoint a client needs to establish its session.
    role: Role | None
    # Sorted for a stable wire shape — clients may diff or cache this.
    permissions: list[str] = Field(default_factory=list)
    # Every organization this user is a member of, current one flagged. Present so
    # a client can TELL whether the account is single- or multi-org instead of
    # inferring "no others exist" from a response that only ever shows one.
    # Read-only: there is no switch endpoint (see the route docstring).
    organizations: list[OrgMembershipOut] = Field(default_factory=list)
    onboarding_completed: bool = False

    @field_validator("role", mode="before")
    @classmethod
    def _tolerate_unusable_role(cls, v: object) -> object:
        """Resolve an unusable role to None instead of raising.

        `Role("")` and `Role("some_future_role")` both raise ValueError, and a
        ValueError raised while BUILDING a response is a 500. The membership row
        that produces it is already handled fail-closed everywhere authorization
        is decided (`effective_permissions` returns the empty set for an unknown
        role), so the honest representation is "no usable role" — never a guess,
        and never an exception.
        """
        if v is None or isinstance(v, Role):
            return v
        try:
            return Role(v)
        except ValueError:
            return None
