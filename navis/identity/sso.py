"""SSO foundation: a provider-agnostic abstraction + connection management.

This is the future-ready shell the audit asked for — a uniform interface across
Google OAuth / OIDC / SAML and a per-org connection record. Full SAML assertion
parsing and OIDC discovery are intentionally NOT implemented here; each provider
is a class that maps an already-verified identity assertion to (email, subject).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.enterprise_models import SsoConnection

SUPPORTED_PROVIDERS = ("google", "oidc", "saml")


@dataclass(frozen=True, slots=True)
class SsoIdentity:
    """The normalized result of a successful SSO assertion."""

    email: str
    external_subject: str
    provider: str
    attributes: dict[str, Any]


class SsoProvider(Protocol):
    name: str

    def normalize(self, raw_assertion: dict[str, Any]) -> SsoIdentity:
        """Map a provider-specific (already-verified) assertion to SsoIdentity."""


class GoogleOAuthProvider:
    name = "google"

    def normalize(self, raw_assertion: dict[str, Any]) -> SsoIdentity:
        return SsoIdentity(
            email=raw_assertion["email"],
            external_subject=raw_assertion["sub"],
            provider=self.name,
            attributes={k: raw_assertion.get(k) for k in ("name", "hd")},
        )


class OidcProvider:
    name = "oidc"

    def normalize(self, raw_assertion: dict[str, Any]) -> SsoIdentity:
        return SsoIdentity(
            email=raw_assertion["email"],
            external_subject=raw_assertion["sub"],
            provider=self.name,
            attributes=raw_assertion.get("claims", {}),
        )


class SamlProvider:
    name = "saml"

    def normalize(self, raw_assertion: dict[str, Any]) -> SsoIdentity:
        # raw_assertion is expected to be the parsed, signature-verified attributes.
        return SsoIdentity(
            email=raw_assertion["email"],
            external_subject=raw_assertion["name_id"],
            provider=self.name,
            attributes=raw_assertion.get("attributes", {}),
        )


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
        if conn is None:
            conn = SsoConnection(organization_id=org_id, provider=provider, config=config, enabled=enabled)
            self.db.add(conn)
        else:
            conn.config = config
            conn.enabled = enabled
        await self.db.flush()
        return conn

    async def list_for_org(self, org_id: uuid.UUID) -> list[SsoConnection]:
        return list(
            (await self.db.scalars(select(SsoConnection).where(SsoConnection.organization_id == org_id))).all()
        )
