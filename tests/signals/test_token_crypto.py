"""P1-1: token encryption — round-trip, legacy passthrough, rotation, tamper."""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from navis.signals.crypto import TokenCipher, TokenDecryptError

KEY_A = Fernet.generate_key().decode()
KEY_B = Fernet.generate_key().decode()


def test_round_trip_and_tagging():
    c = TokenCipher([KEY_A])
    ct = c.encrypt("ya29.secret-access-token")
    assert c.is_encrypted(ct)
    assert ct != "ya29.secret-access-token"
    assert c.decrypt(ct) == "ya29.secret-access-token"


def test_legacy_plaintext_passthrough():
    c = TokenCipher([KEY_A])
    # A value without the enc: tag is treated as legacy plaintext.
    assert c.decrypt("legacy-plaintext-token") == "legacy-plaintext-token"


def test_disabled_cipher_is_noop():
    c = TokenCipher([])  # no keys
    assert c.enabled is False
    assert c.encrypt("tok") == "tok"
    assert c.decrypt("tok") == "tok"


def test_rotation_decrypts_old_encrypts_new():
    old = TokenCipher([KEY_A])
    ct_old = old.encrypt("secret")

    # Rotate: new primary KEY_B, KEY_A retained for decryption.
    rotated = TokenCipher([KEY_B, KEY_A])
    assert rotated.decrypt(ct_old) == "secret"  # old ciphertext still readable

    ct_new = rotated.encrypt("secret")
    # A cipher holding only the old key can no longer read the new ciphertext.
    with pytest.raises(TokenDecryptError):
        old.decrypt(ct_new)


def test_tampered_ciphertext_raises():
    c = TokenCipher([KEY_A])
    ct = c.encrypt("secret")
    tampered = ct[:-4] + "AAAA"
    with pytest.raises(TokenDecryptError):
        c.decrypt(tampered)


def test_encrypted_token_without_keys_raises():
    ct = TokenCipher([KEY_A]).encrypt("secret")
    with pytest.raises(TokenDecryptError):
        TokenCipher([]).decrypt(ct)  # keys rotated away / misconfigured


# ---- SIGNAL-CRYPTO-002: legacy-plaintext sunset switch ----

def test_legacy_sunset_rejects_plaintext(monkeypatch):
    from navis.platform.config import settings

    monkeypatch.setattr(settings, "signals_reject_legacy_plaintext", True)
    c = TokenCipher([KEY_A])
    with pytest.raises(TokenDecryptError):
        c.decrypt("legacy-plaintext-token")
    # Properly-tagged ciphertext is unaffected by the sunset switch.
    assert c.decrypt(c.encrypt("ok")) == "ok"


# ---- SIGNAL-CRYPTO-004: rotate() primitive (bulk key rotation) ----

def test_rotate_reencrypts_under_newest_key():
    old = TokenCipher([KEY_A])
    ct_a = old.encrypt("secret")
    rotator = TokenCipher([KEY_B, KEY_A])  # KEY_B newest, KEY_A retained
    ct_b = rotator.rotate(ct_a)
    assert ct_b != ct_a                       # genuinely re-encrypted
    assert rotator.decrypt(ct_b) == "secret"
    only_b = TokenCipher([KEY_B])
    assert only_b.decrypt(ct_b) == "secret"   # readable with just the new key
    with pytest.raises(TokenDecryptError):
        only_b.decrypt(ct_a)                  # the old ciphertext is not


def test_rotate_brings_legacy_plaintext_under_encryption():
    c = TokenCipher([KEY_A])
    ct = c.rotate("legacy-plaintext")
    assert c.is_encrypted(ct)
    assert c.decrypt(ct) == "legacy-plaintext"


# ---- SIGNAL-CRYPTO-001: production must not run unencrypted ----

def test_production_without_keys_fails_fast(monkeypatch):
    from navis.platform.config import settings

    monkeypatch.delenv("SIGNALS_TOKEN_KEYS", raising=False)
    monkeypatch.setattr(settings, "signals_token_keys", None)
    monkeypatch.setattr(settings, "environment", "production")
    with pytest.raises(RuntimeError):
        TokenCipher.from_env()


def test_non_production_without_keys_is_passthrough(monkeypatch):
    from navis.platform.config import settings

    monkeypatch.delenv("SIGNALS_TOKEN_KEYS", raising=False)
    monkeypatch.setattr(settings, "signals_token_keys", None)
    monkeypatch.setattr(settings, "environment", "development")
    c = TokenCipher.from_env()
    assert c.enabled is False
    assert c.encrypt("tok") == "tok"
