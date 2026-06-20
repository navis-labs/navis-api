"""SSO foundation: a provider-agnostic abstraction + connection management.

This is the future-ready shell the audit asked for — a uniform interface across
Google OAuth / OIDC / SAML and a per-org connection record. Full SAML assertion
parsing and OIDC discovery are intentionally NOT implemented here; each provider
is a class that maps an already-verified identity assertion to (email, subject).
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.enterprise_models import SsoConnection
from navis.platform.security import field_cipher

SUPPORTED_PROVIDERS = ("google", "oidc", "saml")
# SSO config holds provider secrets (OIDC client_secret, SAML keys). It is encrypted
# at rest as a single opaque blob: the whole dict is JSON-serialized + Fernet-encrypted
# and stored under one key. This (a) protects nested secrets at any depth, and (b)
# sidesteps SQLAlchemy's shallow JSONB mutation tracking entirely — the column is
# always reassigned, never mutated in place.
_ENC_KEY = "__enc__"


def _encrypt_config(config: dict[str, Any]) -> dict[str, Any]:
    return {_ENC_KEY: field_cipher.encrypt(json.dumps(config, default=str))}


def _decrypt_config(stored: dict[str, Any] | None) -> dict[str, Any]:
    if not stored or _ENC_KEY not in stored:
        return stored or {}  # legacy plaintext config (pre-encryption rows)
    raw = field_cipher.decrypt(stored[_ENC_KEY])
    return json.loads(raw) if raw else {}


# Hard cap on assertion attribute size — a hostile/buggy IdP must not be able to
# push a multi-MB attribute bag into memory/DB.
_MAX_ATTRIBUTE_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class VerifiedAssertion:
    """An SSO assertion whose signature, issuer, audience, and expiry/replay have
    ALREADY been verified by the callback. Constructing one is a *claim* that those
    checks passed — `normalize()` accepts ONLY this type, never a raw dict, so an
    unverified payload can never structurally reach identity mapping (the exact class
    of bug — `normalize(request.json())` — that causes SAML account takeover later).

    The SSO callback is responsible for building this; do NOT construct it from
    untrusted input without verification.
    """

    provider: str
    claims: dict[str, Any]                 # the verified attribute bag
    issuer: str | None = None              # identity-graph + trust fields
    audience: str | None = None
    auth_time: int | None = None
    assertion_id: str | None = None        # for replay tracking

    def __post_init__(self) -> None:
        import json

        if len(json.dumps(self.claims, default=str)) > _MAX_ATTRIBUTE_BYTES:
            raise ValueError("sso assertion attributes exceed size limit")


@dataclass(frozen=True, slots=True)
class SsoIdentity:
    """The normalized result of a verified SSO assertion (identity-graph shaped)."""

    email: str
    external_subject: str
    provider: str
    attributes: dict[str, Any]
    issuer: str | None = None
    audience: str | None = None
    auth_time: int | None = None


class SsoProvider(Protocol):
    name: str

    def normalize(self, assertion: VerifiedAssertion) -> SsoIdentity:
        """Map a VERIFIED assertion to SsoIdentity. Cannot be passed a raw dict."""


def _identity(provider: str, email: str, subject: str, a: VerifiedAssertion, attributes: dict[str, Any]) -> SsoIdentity:
    return SsoIdentity(
        email=email, external_subject=subject, provider=provider, attributes=attributes,
        issuer=a.issuer, audience=a.audience, auth_time=a.auth_time,
    )


class GoogleOAuthProvider:
    name = "google"

    def normalize(self, assertion: VerifiedAssertion) -> SsoIdentity:
        c = assertion.claims
        return _identity(self.name, c["email"], c["sub"], assertion,
                         {k: c.get(k) for k in ("name", "hd")})


class OidcProvider:
    name = "oidc"

    def normalize(self, assertion: VerifiedAssertion) -> SsoIdentity:
        c = assertion.claims
        return _identity(self.name, c["email"], c["sub"], assertion, c.get("claims", {}))


class SamlProvider:
    name = "saml"

    def normalize(self, assertion: VerifiedAssertion) -> SsoIdentity:
        c = assertion.claims
        return _identity(self.name, c["email"], c["name_id"], assertion, c.get("attributes", {}))


_PROVIDERS: dict[str, SsoProvider] = {
    p.name: p for p in (GoogleOAuthProvider(), OidcProvider(), SamlProvider())
}


def get_provider(name: str) -> SsoProvider:
    if name not in _PROVIDERS:
        raise ValueError(f"unsupported sso provider: {name}")
    return _PROVIDERS[name]


class SsoConnectionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def upsert(self, *, org_id: uuid.UUID, provider: str, config: dict[str, Any], enabled: bool) -> SsoConnection:
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"unsupported sso provider: {provider}")
        conn = await self.db.scalar(
            select(SsoConnection).where(
                SsoConnection.organization_id == org_id, SsoConnection.provider == provider
            )
        )
        encrypted = _encrypt_config(config)  # secrets encrypted before they touch the DB
        if conn is None:
            conn = SsoConnection(organization_id=org_id, provider=provider, config=encrypted, enabled=enabled)
            self.db.add(conn)
        else:
            conn.config = encrypted
            conn.enabled = enabled
        await self.db.flush()
        return conn

    async def list_for_org(self, org_id: uuid.UUID) -> list[SsoConnection]:
        return list(
            (await self.db.scalars(select(SsoConnection).where(SsoConnection.organization_id == org_id))).all()
        )

    async def get_config(self, org_id: uuid.UUID, provider: str) -> dict[str, Any]:
        """Decrypted config for use by an SSO flow (never exposes ciphertext)."""
        conn = await self.db.scalar(
            select(SsoConnection).where(
                SsoConnection.organization_id == org_id, SsoConnection.provider == provider
            )
        )
        return _decrypt_config(conn.config) if conn else {}
