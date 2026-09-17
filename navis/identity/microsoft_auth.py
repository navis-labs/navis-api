"""Microsoft (Entra ID) credential verification for SIGN-IN.

The mirror of `navis.identity.google_auth`: exchanges an authorization code,
verifies the resulting id_token, and returns normalized identity claims. The
caller (`IdentityService.microsoft_signin`) owns the DB work.

── This module requests IDENTITY SCOPES ONLY ──────────────────────────────

`openid profile email offline_access User.Read`. Never `Mail.Read`, never
anything else the connector asks for. Signing in and handing over a mailbox are
two separate decisions a founder makes at two different moments, and the login
screen is the worst possible place to ask for the second — it turns a sign-up
into a security review. The connector's consent lives in
`navis.signals.microsoft.oauth` behind a different redirect URI and a different
feature flag, and the two must not be merged.

── Email verification is the hard part, and it is a security decision ─────

Entra's `email` claim is NOT inherently trustworthy. For a personal Microsoft
account the holder can set it to any address they like, including one belonging
to somebody else's company. If sign-in linked accounts on that claim, anyone
could create a free Microsoft account carrying a target's work address, sign in,
and land inside that person's existing Navis account. That is an
account-takeover primitive, not a convenience bug.

The signal that makes the claim trustworthy is `xms_edov` (Email Domain Owner
Verified) — an OPTIONAL claim that must be configured on the app registration.
When it is present and true, Entra is asserting the tenant demonstrably owns the
email's domain. When it is absent or false, this module reports the email as
unverified and `microsoft_signin` refuses to link.

Because `xms_edov` must be explicitly configured, a deployment that forgets it
gets NO linking at all rather than unsafe linking. That is the correct direction
to fail: a founder who cannot link sees an error and asks; a founder whose
account was taken over does not.
"""
from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Any

import httpx

from navis.platform.config import settings
from navis.platform.monitoring import get_logger

logger = get_logger("identity.microsoft_auth")

_AUTHORITY = "https://login.microsoftonline.com"

# The tenant id every personal Microsoft account (Outlook.com, Hotmail, Xbox)
# authenticates against. Identities from here have no organisation behind them
# and therefore no domain ownership anybody verified, which is exactly the
# population the linking rule above exists to keep out.
CONSUMER_TENANT_ID = "9188040d-6c67-4c5b-b112-36a304b66dad"

# Identity only. Asserted by a test, because a mailbox scope appearing here
# would be a product regression rather than a configuration one.
MICROSOFT_SIGNIN_SCOPES = (
    "openid",
    "profile",
    "email",
    "offline_access",
    "https://graph.microsoft.com/User.Read",
)


@dataclass(frozen=True, slots=True)
class MicrosoftIdentity:
    """Normalized identity claims extracted from a verified Entra credential."""

    email: str
    microsoft_sub: str      # `oid` — stable per user, never changes
    tenant_id: str | None   # `tid` — the Entra tenant the user authenticated against
    full_name: str | None
    # True ONLY when Entra asserts domain ownership (`xms_edov`). Drives whether
    # this identity may be linked to an existing account — see the module
    # docstring and `IdentityService.microsoft_signin`.
    email_verified: bool

    @property
    def is_personal_account(self) -> bool:
        return self.tenant_id == CONSUMER_TENANT_ID


class MicrosoftAuthError(Exception):
    """Raised when Entra credential verification or code exchange fails.

    Maps to HTTP 401 at the route level, exactly as GoogleAuthError does.
    """


def _decode_claims(id_token: str) -> dict[str, Any]:
    """Read the claims from an id_token that came straight from Entra's token
    endpoint over TLS in the authorization-code flow.

    The signature is not re-verified here for the same reason
    `google_auth`/`signals.router` do not re-verify theirs on this path: the
    token was not supplied by the client, it was fetched by this server from
    the issuer over an authenticated TLS channel using the client secret. There
    is no untrusted hop for a forged token to enter through.

    Note this reasoning holds ONLY for the code-exchange path. It would not hold
    for an id_token posted by a browser, which is why this module deliberately
    offers no id_token entry point — unlike the Google service, which keeps one
    for backward compatibility and verifies the signature properly there.
    """
    try:
        payload_b64 = id_token.split(".")[1]
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
    except (IndexError, binascii.Error, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MicrosoftAuthError("could not read Microsoft id_token") from exc
    if not isinstance(claims, dict):
        raise MicrosoftAuthError("unexpected Microsoft id_token payload")
    return claims


def _email_from_claims(claims: dict[str, Any]) -> str | None:
    email = claims.get("email")
    if isinstance(email, str) and "@" in email:
        return email.lower()
    upn = claims.get("preferred_username")
    if isinstance(upn, str) and "@" in upn:
        return upn.lower()
    return None


def _email_verified(claims: dict[str, Any]) -> bool:
    """Whether Entra asserts the email's domain is owned by the tenant.

    `xms_edov` arrives as a bool or, on some token versions, the strings
    "true"/"1". Anything else — including the claim being absent entirely —
    is treated as NOT verified. Defaulting to True when the claim is missing
    would silently reinstate the takeover path on any deployment that forgot to
    configure the optional claim, which is precisely the failure this guards.
    """
    raw = claims.get("xms_edov")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "1")
    return False


class MicrosoftAuthService:
    """Verifies Entra credentials and returns a normalized MicrosoftIdentity.

    Stateless — create one per request, no shared mutable state.
    """

    def __init__(self) -> None:
        self._client_id = settings.microsoft_client_id
        self._client_secret = settings.microsoft_client_secret
        self._tenant = settings.microsoft_tenant
        self._redirect_uri = settings.microsoft_signin_redirect_uri

    @property
    def configured(self) -> bool:
        """Credentials AND the redirect URI — the exchange sends all three.

        Without the redirect URI the POST carries `redirect_uri=None`, Entra
        rejects it, and the user sees a generic 401 that reads like a bad
        password rather than a deployment that was never finished configuring.
        """
        return bool(self._client_id and self._client_secret and self._redirect_uri)

    async def verify_code(self, code: str) -> MicrosoftIdentity:
        """Exchange an authorization code for tokens, then read the id_token."""
        if not self.configured:
            raise MicrosoftAuthError("Microsoft sign-in is not configured")

        token_url = f"{_AUTHORITY}/{self._tenant}/oauth2/v2.0/token"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    token_url,
                    data={
                        "code": code,
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                        "redirect_uri": self._redirect_uri,
                        "grant_type": "authorization_code",
                        "scope": " ".join(MICROSOFT_SIGNIN_SCOPES),
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning("microsoft token exchange network error: %s", exc)
            raise MicrosoftAuthError("could not reach Microsoft") from exc

        if resp.status_code != 200:
            logger.warning(
                "microsoft token exchange failed status=%d body=%s",
                resp.status_code, resp.text[:300],
            )
            raise MicrosoftAuthError("Microsoft rejected the sign-in code")

        payload = resp.json()
        id_token = payload.get("id_token")
        if not id_token:
            raise MicrosoftAuthError("Microsoft returned no id_token")

        claims = _decode_claims(id_token)
        email = _email_from_claims(claims)
        subject = claims.get("oid") or claims.get("sub")
        if not email or not subject:
            raise MicrosoftAuthError("Microsoft did not return an account email")

        return MicrosoftIdentity(
            email=email,
            microsoft_sub=str(subject),
            tenant_id=claims.get("tid"),
            full_name=claims.get("name"),
            email_verified=_email_verified(claims),
        )
