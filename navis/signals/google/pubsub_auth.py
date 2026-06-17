"""Verify that a Pub/Sub push genuinely came from Google.

Google signs each push with an OIDC JWT in the `Authorization: Bearer <jwt>`
header. We verify the signature (Google certs), then the claims:
  * issuer is accounts.google.com,
  * audience matches our configured push audience,
  * the service-account email matches the one we authorized (if configured),
  * email_verified is true.

The signature decoder is injectable so the claim logic is unit-testable without
network access to Google's certs.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from navis.platform.monitoring import get_logger

logger = get_logger("signals.pubsub_auth")

_VALID_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}

Decoder = Callable[[str, str], dict[str, Any]]  # (token, audience) -> claims


class PubSubAuthError(Exception):
    """The push could not be authenticated as a genuine Google Pub/Sub request."""


def verify_claims(claims: dict[str, Any], *, audience: str | None, sa_email: str | None) -> None:
    """App-level claim checks (signature/exp already verified by the decoder)."""
    if claims.get("iss") not in _VALID_ISSUERS:
        raise PubSubAuthError(f"bad issuer: {claims.get('iss')!r}")
    if audience and claims.get("aud") != audience:
        raise PubSubAuthError("audience mismatch")
    if sa_email and claims.get("email") != sa_email:
        raise PubSubAuthError("service-account email mismatch")
    if claims.get("email") and claims.get("email_verified") is False:
        raise PubSubAuthError("service-account email not verified")


def _default_decoder(token: str, audience: str) -> dict[str, Any]:  # pragma: no cover - network
    from google.auth.transport import requests as g_requests
    from google.oauth2 import id_token

    return id_token.verify_oauth2_token(token, g_requests.Request(), audience or None)


def verify_pubsub_request(
    authorization_header: str | None,
    *,
    audience: str | None = None,
    sa_email: str | None = None,
    decoder: Decoder | None = None,
) -> dict[str, Any]:
    """Verify the bearer JWT and return its claims, or raise PubSubAuthError."""
    audience = audience or os.environ.get("GOOGLE_PUBSUB_AUDIENCE")
    sa_email = sa_email or os.environ.get("GOOGLE_PUBSUB_SA_EMAIL")

    if not authorization_header or not authorization_header.lower().startswith("bearer "):
        raise PubSubAuthError("missing bearer token")
    token = authorization_header.split(" ", 1)[1].strip()

    decode = decoder or _default_decoder
    try:
        claims = decode(token, audience or "")
    except PubSubAuthError:
        raise
    except Exception as exc:  # signature/exp failure from the decoder
        raise PubSubAuthError(f"invalid token signature: {exc}") from exc

    verify_claims(claims, audience=audience, sa_email=sa_email)
    return claims


def pubsub_auth_configured() -> bool:
    return bool(os.environ.get("GOOGLE_PUBSUB_AUDIENCE"))
