"""Token encryption at rest for integration OAuth tokens.

  * Fernet (AES-128-CBC + HMAC) via `cryptography` (proven, boring).
  * Keys come from the environment: `SIGNALS_TOKEN_KEYS` = comma-separated Fernet
    keys, newest first. `MultiFernet` encrypts with the first and decrypts with
    any — that is the rotation story: prepend a new key, old ciphertexts still
    decrypt, re-encryption happens lazily on next write.
  * Backward compatible: ciphertext is tagged `enc:1:`. A value without the tag is
    treated as legacy plaintext and returned as-is (and re-encrypted on next write).
  * If no keys are configured the cipher is disabled (passthrough) and logs loudly —
    so dev still works, but production should always set keys.

Generate a key:  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
from __future__ import annotations

import os

from navis.platform.monitoring import get_logger

logger = get_logger("signals.crypto")

_PREFIX = "enc:1:"
_ENV_VAR = "SIGNALS_TOKEN_KEYS"


class TokenDecryptError(Exception):
    """Ciphertext could not be decrypted with any configured key (tampered/rotated out)."""


class TokenCipher:
    def __init__(self, keys: list[str]) -> None:
        self._multifernet = None
        if keys:
            try:
                from cryptography.fernet import Fernet, MultiFernet

                self._multifernet = MultiFernet([Fernet(k.encode()) for k in keys])
            except Exception as exc:  # invalid key material
                logger.error("invalid %s; token encryption DISABLED: %r", _ENV_VAR, exc)
                self._multifernet = None

    @classmethod
    def from_env(cls) -> "TokenCipher":
        raw = os.environ.get(_ENV_VAR, "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            logger.warning("%s not set; integration tokens stored UNENCRYPTED", _ENV_VAR)
        return cls(keys)

    @property
    def enabled(self) -> bool:
        return self._multifernet is not None

    def encrypt(self, plaintext: str | None) -> str | None:
        if plaintext is None or not self.enabled:
            return plaintext
        token = self._multifernet.encrypt(plaintext.encode()).decode()
        return f"{_PREFIX}{token}"

    def decrypt(self, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith(_PREFIX):
            return value  # legacy plaintext (or cipher disabled when written)
        if not self.enabled:
            raise TokenDecryptError("encrypted token present but no keys configured")
        from cryptography.fernet import InvalidToken

        try:
            return self._multifernet.decrypt(value[len(_PREFIX):].encode()).decode()
        except InvalidToken as exc:
            raise TokenDecryptError("token failed decryption (rotated out or tampered)") from exc

    def is_encrypted(self, value: str | None) -> bool:
        return bool(value) and value.startswith(_PREFIX)


token_cipher = TokenCipher.from_env()
