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


class MeOut(BaseModel):
    user: UserOut
    organization: OrganizationOut
    role: Role
    onboarding_completed: bool = False
