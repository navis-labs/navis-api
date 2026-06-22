"""Token encryption at rest for integration OAuth tokens.

  * Fernet (AES-128-CBC + HMAC) via `cryptography` (proven, boring). Authenticated, so
    tampered ciphertext is detected (InvalidToken -> TokenDecryptError).
  * Keys come from the environment: `SIGNALS_TOKEN_KEYS` = comma-separated Fernet keys,
    newest first. `MultiFernet` encrypts with the first and decrypts with any — the
    rotation story: prepend a new key, old ciphertexts still decrypt, re-encryption
    happens lazily on next write (or eagerly via `rotate()` for a bulk key-rotation job).
  * Backward compatible: ciphertext is tagged `enc:1:`. An untagged value is treated as
    legacy plaintext and returned as-is — UNLESS the sunset switch
    (`signals_reject_legacy_plaintext`) is on, which rejects it (SIGNAL-CRYPTO-002).
  * Keys are mandatory in PRODUCTION: missing keys fail at startup rather than silently
    storing tokens unencrypted (SIGNAL-CRYPTO-001). dev/staging may run passthrough.

Generate a key:  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

if TYPE_CHECKING:
    from cryptography.fernet import MultiFernet

logger = get_logger("signals.crypto")

_PREFIX = "enc:1:"  # version tag; a future scheme would be enc:2: with its own key set
_ENV_VAR = "SIGNALS_TOKEN_KEYS"


class TokenDecryptError(Exception):
    """Ciphertext could not be decrypted with any configured key (tampered/rotated out),
    or a legacy-plaintext token was seen while the sunset switch is enabled."""


class TokenCipher:
    def __init__(self, keys: list[str]) -> None:
        self._multifernet: MultiFernet | None = None
        if keys:
            try:
                from cryptography.fernet import Fernet, MultiFernet

                self._multifernet = MultiFernet([Fernet(k.encode()) for k in keys])
            except Exception as exc:  # invalid key material
                logger.error("invalid %s; token encryption DISABLED: %r", _ENV_VAR, exc)
                self._multifernet = None

    @classmethod
    def from_env(cls) -> "TokenCipher":
        # Real exported env var wins (prod secret managers); else fall back to settings,
        # which loads .env like the rest of config.
        raw = os.environ.get(_ENV_VAR) or settings.signals_token_keys or ""
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            # SIGNAL-CRYPTO-001: production must never store tokens unencrypted. Fail fast
            # at startup instead of silently degrading to passthrough.
            if settings.is_production:
                raise RuntimeError(
                    f"{_ENV_VAR} is required in production — refusing to start with "
                    f"integration tokens stored UNENCRYPTED. Set {_ENV_VAR} "
                    f"(comma-separated Fernet keys, newest first)."
                )
            logger.warning(
                "%s not set; integration tokens stored UNENCRYPTED (non-production only)", _ENV_VAR
            )
        return cls(keys)

    @property
    def enabled(self) -> bool:
        return self._multifernet is not None

    def encrypt(self, plaintext: str | None) -> str | None:
        if plaintext is None or self._multifernet is None:
            return plaintext
        token = self._multifernet.encrypt(plaintext.encode()).decode()
        return f"{_PREFIX}{token}"

    def decrypt(self, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith(_PREFIX):
            # Legacy plaintext (or written while the cipher was disabled in dev).
            metrics.incr("signals.token.legacy_plaintext_seen")
            if settings.signals_reject_legacy_plaintext:
                raise TokenDecryptError("legacy plaintext token rejected (sunset enabled)")
            return value
        if self._multifernet is None:
            raise TokenDecryptError("encrypted token present but no keys configured")
        from cryptography.fernet import InvalidToken

        try:
            plaintext = self._multifernet.decrypt(value[len(_PREFIX):].encode()).decode()
            metrics.incr("signals.token.decrypt_success")
            return plaintext
        except InvalidToken as exc:
            metrics.incr("signals.token.decrypt_failure")
            raise TokenDecryptError("token failed decryption (rotated out or tampered)") from exc

    def rotate(self, value: str | None) -> str | None:
        """Re-encrypt an existing token under the NEWEST key without exposing plaintext —
        the primitive a bulk key-rotation job calls after prepending a new key
        (SIGNAL-CRYPTO-004). Legacy plaintext is encrypted fresh. No-op when disabled."""
        if value is None or self._multifernet is None:
            return value
        if not value.startswith(_PREFIX):
            return self.encrypt(value)  # bring legacy plaintext under encryption
        from cryptography.fernet import InvalidToken

        try:
            rotated = self._multifernet.rotate(value[len(_PREFIX):].encode()).decode()
            metrics.incr("signals.token.rotation_performed")
            return f"{_PREFIX}{rotated}"
        except InvalidToken as exc:
            metrics.incr("signals.token.decrypt_failure")
            raise TokenDecryptError("token failed rotation (rotated out or tampered)") from exc

    def is_encrypted(self, value: str | None) -> bool:
        return bool(value) and value.startswith(_PREFIX)


token_cipher = TokenCipher.from_env()
