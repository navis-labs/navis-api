"""JWT issue/verify. Tokens carry the user id (`sub`) and org id (`org`).

Every token gets a unique `jti` so it can be individually revoked (a denylist
keyed on `jti` belongs to the identity domain — platform only mints the claim).
`nbf`/`iat` are set and a small `leeway` tolerates clock skew across hosts.

Signing is algorithm-driven (Platform V2): HS256 (shared secret) by default, or an
asymmetric algorithm (RS256/ES256/PS256) when ALGORITHM + JWT_PRIVATE_KEY/PUBLIC_KEY
are configured. Asymmetric signing lets independent services verify tokens with only
the public key and enables key rotation via `kid`/JWKS — the enterprise posture —
with NO change to these function signatures. Migration is config-only.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from navis.platform.config import settings

# Tolerate up to 30s of clock drift between issuer and verifier.
_LEEWAY_SECONDS = 30


class TokenError(Exception):
    """Raised when a token is invalid, expired, or of the wrong type."""


def _encode(claims: dict[str, Any], expires: timedelta, token_type: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        **claims,
        "type": token_type,
        "iat": now,
        "nbf": now,
        "exp": now + expires,
        "iss": settings.app_name,
        "jti": uuid.uuid4().hex,
    }
    headers = {"kid": settings.jwt_key_id} if settings.jwt_key_id else None
    return jwt.encode(
        payload, settings.jwt_signing_key, algorithm=settings.algorithm, headers=headers
    )


def create_access_token(subject: str, org_id: str | None = None, sid: str | None = None) -> str:
    # `sid` links the token to its server-side Session so logout can revoke that exact
    # session (single-device), not just deny the access jti.
    claims = {"sub": subject, "org": org_id}
    if sid is not None:
        claims["sid"] = sid
    return _encode(
        claims,
        timedelta(minutes=settings.access_token_expire_minutes),
        "access",
    )


def create_refresh_token(subject: str, org_id: str | None = None, sid: str | None = None) -> str:
    claims: dict[str, Any] = {"sub": subject, "org": org_id}
    if sid is not None:
        claims["sid"] = sid
    return _encode(
        claims,
        timedelta(days=settings.refresh_token_expire_days),
        "refresh",
    )


def decode_token(token: str, expected_type: str | None = None) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_verify_key,
            algorithms=[settings.algorithm],
            issuer=settings.app_name,
            options={
                "leeway": _LEEWAY_SECONDS,
                "require_exp": True,
                "require_iat": True,
            },
        )
    except JWTError as exc:
        raise TokenError(str(exc)) from exc
    if expected_type and payload.get("type") != expected_type:
        raise TokenError(f"expected {expected_type} token")
    return payload
