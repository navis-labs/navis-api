"""Application-level field encryption at rest (platform-wide, reusable).

Generalizes the proven signals token-cipher to any domain that must encrypt a
sensitive value before it touches the database — e.g. Identity's SSO client secrets.
Living in `platform.security` lets Identity use it WITHOUT importing a business
domain (signals), preserving the layer law.

  * Fernet (AES-128-CBC + HMAC) via `cryptography`, MultiFernet for key rotation.
  * Keys from env `FIELD_ENCRYPTION_KEYS` = comma-separated Fernet keys, newest first.
  * Ciphertext tagged `fenc:1:`. Untagged values are treated as legacy plaintext and
    returned as-is (and re-encrypted on next write) — backward compatible.
  * No keys configured -> passthrough + loud warning (dev works; prod MUST set keys).

Generate a key:  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
from __future__ import annotations

import os

from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("platform.crypto")

_PREFIX = "fenc:1:"
_ENV_VAR = "FIELD_ENCRYPTION_KEYS"


class FieldDecryptError(Exception):
    """Ciphertext could not be decrypted with any configured key (tampered/rotated out)."""


class FieldCipher:
    def __init__(self, keys: list[str]) -> None:
        self._multifernet = None
        if keys:
            try:
                from cryptography.fernet import Fernet, MultiFernet

                self._multifernet = MultiFernet([Fernet(k.encode()) for k in keys])
            except Exception as exc:  # invalid key material
                logger.error("invalid %s; field encryption DISABLED: %r", _ENV_VAR, exc)
                self._multifernet = None

    @classmethod
    def from_env(cls) -> "FieldCipher":
        # Real exported env var wins (prod secret managers); else fall back to settings,
        # which loads .env like the rest of config.
        from navis.platform.config import settings

        raw = os.environ.get(_ENV_VAR) or settings.field_encryption_keys or ""
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            # CR-1 / SIGNAL-CRYPTO-001 parity: production must never store CONTENT (Gmail
            # bodies, Slack messages, attachment/channel/user names, SSO secrets) unencrypted.
            # Fail fast at startup exactly like TokenCipher — a missing key is a hard stop, not
            # a warning, so plaintext can never reach the DB, backups, or replicas.
            if settings.is_production:
                raise RuntimeError(
                    f"{_ENV_VAR} is required in production — refusing to start with "
                    f"sensitive content stored UNENCRYPTED. Set {_ENV_VAR} "
                    f"(comma-separated Fernet keys, newest first)."
                )
            logger.warning(
                "%s not set; sensitive content/fields stored UNENCRYPTED (non-production only)", _ENV_VAR
            )
        return cls(keys)

    @property
    def enabled(self) -> bool:
        return self._multifernet is not None

    def encrypt(self, plaintext: str | None) -> str | None:
        if plaintext is None or self._multifernet is None:
            return plaintext
        return f"{_PREFIX}{self._multifernet.encrypt(plaintext.encode()).decode()}"

    def decrypt(self, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith(_PREFIX):
            return value  # legacy plaintext (or cipher disabled when written)
        if self._multifernet is None:
            raise FieldDecryptError("encrypted value present but no keys configured")
        from cryptography.fernet import InvalidToken

        try:
            return self._multifernet.decrypt(value[len(_PREFIX):].encode()).decode()
        except InvalidToken as exc:
            raise FieldDecryptError("value failed decryption (rotated out or tampered)") from exc

    def is_encrypted(self, value: str | None) -> bool:
        return bool(value) and value.startswith(_PREFIX)


def field_encryption_healthy() -> bool:
    """Liveness/readiness signal: content encryption must be ON in production. In dev it may
    run passthrough (returns False there) — callers/health endpoints decide what to do with
    that per environment."""
    from navis.platform.config import settings

    healthy = field_cipher.enabled or not settings.is_production
    metrics.incr("crypto.field_cipher.health", labels={"enabled": str(field_cipher.enabled)})
    return healthy


field_cipher = FieldCipher.from_env()
metrics.incr("crypto.field_cipher.init", labels={"enabled": str(field_cipher.enabled)})
