"""Password hashing (bcrypt via passlib).

Two production guards:
  * cost factor pinned to 12 rounds (tune with hardware);
  * bcrypt silently truncates input at 72 bytes — a real footgun, since a long
    passphrase would be accepted but only its first 72 bytes verified. We
    pre-hash any input longer than 72 bytes through SHA-256 (base64, 44 bytes)
    so the full secret always contributes. This is transparent and applied
    identically on hash and verify.
"""
from __future__ import annotations

import base64
import hashlib

from passlib.context import CryptContext

_BCRYPT_MAX_BYTES = 72

_pwd = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)


def _prepare(secret: str) -> str:
    """Collapse over-long secrets so bcrypt's 72-byte limit can't truncate them."""
    if len(secret.encode("utf-8")) <= _BCRYPT_MAX_BYTES:
        return secret
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def hash_password(plain: str) -> str:
    return _pwd.hash(_prepare(plain))


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd.verify(_prepare(plain), hashed)
