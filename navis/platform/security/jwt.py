"""JWT issue/verify. Tokens carry the user id (`sub`) and org id (`org`).

Every token gets a unique `jti` so it can be individually revoked (a denylist
keyed on `jti` belongs to the identity domain — platform only mints the claim).
`nbf`/`iat` are set and a small `leeway` tolerates clock skew across hosts.
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
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def create_access_token(subject: str, org_id: str | None = None) -> str:
    return _encode(
        {"sub": subject, "org": org_id},
        timedelta(minutes=settings.access_token_expire_minutes),
        "access",
    )


def create_refresh_token(subject: str, org_id: str | None = None) -> str:
    return _encode(
        {"sub": subject, "org": org_id},
        timedelta(days=settings.refresh_token_expire_days),
        "refresh",
    )


def decode_token(token: str, expected_type: str | None = None) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
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
