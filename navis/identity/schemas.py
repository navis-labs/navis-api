"""Pydantic request/response schemas for identity."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from navis.identity.models import DeveloperRole, OrganizationKind, Role
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
    #: OPTIONAL, AND OMITTING IT IS A REAL CHOICE (P0-4).
    #:
    #: Naming an organization was required, so a solo developer with no company
    #: had to invent one to reach the Developer surface at all — and the tenant
    #: that came out was indistinguishable from a real company. Omit this and
    #: signup creates a PERSONAL workspace instead: still an ordinary
    #: `organizations` row, isolated by the same `organization_id` predicate,
    #: but classified `kind='personal'` so the product can tell the difference.
    #:
    #: Backward compatible by construction: a required field becoming optional
    #: cannot break a caller that was already sending it, and every caller that
    #: sends one still gets a company.
    organization_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=_MAX_NAME,
        description=(
            "Name of the company to create. Omit it to create a personal "
            "workspace for this user instead — no company required."
        ),
    )

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

    THE TWO WORKSPACE FIELDS ARE CONSULTED ONLY FOR A BRAND-NEW ACCOUNT. A
    Google account that already holds a membership resolves to the workspace it
    already belongs to, and both fields are ignored — see
    `IdentityService.google_signin`. That is what makes it safe for the console
    to send the same body for sign-in and sign-up: there is one Google button,
    and the caller cannot know in advance which of the two it will turn out to
    be.
    """

    code: str | None = Field(None, description="OAuth2 authorization code (preferred)")
    id_token: str | None = Field(None, description="Google id_token (legacy fallback)")
    create_workspace: bool = Field(
        True,
        description=(
            "Whether an unknown Google account may be PROVISIONED by this call. "
            "True — the default, and what every caller before P0-8 did — keeps "
            "the find-or-create behaviour. Send False to make this a pure "
            "SIGN-IN: an identity Navis does not know is refused 404 and "
            "nothing is written, so the caller can ask the person how their "
            "workspace should be set up before one exists. A person who was "
            "INVITED already has a user row, so they resolve normally either "
            "way — this refuses only a genuinely unknown identity."
        ),
    )
    workspace_kind: OrganizationKind | None = Field(
        None,
        description=(
            "What to create if this Google account is new to Navis. Omit it and "
            "a company workspace is created, which is what this route has always "
            "done — so an existing caller is unaffected. Send `personal` for a "
            "developer with no company; send `company` with "
            "`organization_name` to name the workspace."
        ),
    )
    organization_name: str | None = Field(
        None,
        min_length=2,
        max_length=_MAX_NAME,
        description=(
            "Name for the company workspace to create, when this account is new. "
            "Omitted, the workspace is named after the person — which is all "
            "Google tells us about them."
        ),
    )

    @model_validator(mode="after")
    def _require_exactly_one(self) -> "GoogleAuthRequest":
        if not self.code and not self.id_token:
            raise ValueError("provide either 'code' or 'id_token'")
        if self.code and self.id_token:
            raise ValueError("provide either 'code' or 'id_token', not both")
        # A personal workspace is one person's own space, so a company name for
        # it is not a detail to ignore — it is a caller that believes it is
        # creating something other than what it asked for. Refused rather than
        # silently dropped.
        if self.workspace_kind is OrganizationKind.PERSONAL and self.organization_name:
            raise ValueError(
                "a personal workspace is named after its owner; "
                "omit 'organization_name' or send workspace_kind='company'"
            )
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


class SwitchOrganizationRequest(BaseModel):
    """Body for POST /auth/switch — which of the caller's own seats to use.

    THE ID IS A SELECTION, NEVER AN AUTHORITY, and the distinction is the whole
    security argument for this route. The caller names an organization; the
    route then looks for a `memberships` row joining THIS authenticated user to
    it and refuses when there is none. Exactly the pattern
    `authz._principal_from_jwt` already uses for the JWT's own `org` claim: the
    claim selects a row, and the row — not the claim — is the answer. A token
    minted here is therefore re-validated against the same membership on every
    subsequent request, so even a mistake in this handler could not widen
    access beyond a seat the database agrees the person holds.
    """

    organization_id: uuid.UUID


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
    #: `company` or `personal` — what this tenant IS, never what it may do.
    #:
    #: PUBLISHED SO A SURFACE CAN STOP ASKING FOR THINGS THAT CANNOT APPLY.
    #: `OnboardingStage` is an Application ladder — profile, Gmail, calendar,
    #: first brief — and a personal workspace never climbs it, so a console
    #: that cannot tell the two apart reports "onboarding is not finished"
    #: forever at a developer who has nothing to finish. That is the fake
    #: Application activation step this field exists to remove.
    #:
    #: NOT AN AUTHORIZATION INPUT, and it must never become one. No gate reads
    #: it, no query branches on it, and isolation is the same
    #: `organization_id` predicate for both kinds. See `OrganizationKind`.
    kind: OrganizationKind


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
    #: This member's role on the DEVELOPER surface, or null for no Developer
    #: access (P1-6).
    #:
    #: PUBLISHED SO A SURFACE CAN DESCRIBE ITSELF, NOT SO IT CAN DECIDE. It is
    #: already folded into `permissions` below — the Developer Console reads
    #: that list to choose which controls to OFFER, exactly as it does today,
    #: and the Core re-checks every request regardless. This field exists
    #: because "you have no Developer access" and "your Developer role is
    #: viewer" are different things to say to a person, and a permission list
    #: alone cannot distinguish them from a failed read.
    #:
    #: Null is a real, common and correct value: every Application-only member
    #: has it.
    developer_role: DeveloperRole | None = None
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

    @field_validator("developer_role", mode="before")
    @classmethod
    def _tolerate_unusable_developer_role(cls, v: object) -> object:
        """Same tolerance as `role`, for the same reason.

        A `developer_role` value this build does not recognise resolves to null
        rather than 500ing the one endpoint a client needs to establish its
        session. Authorization already treats it as zero privilege
        (`default_developer_permissions_for_role` returns the empty set), so
        null is the honest report and not a downgrade of anything.
        """
        if v is None or isinstance(v, DeveloperRole):
            return v
        try:
            return DeveloperRole(v)
        except ValueError:
            return None
