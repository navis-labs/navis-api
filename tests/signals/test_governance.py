"""Signal governance: field-encryption prod enforcement (CR-1), content-retention policy +
tombstone (M-1), and per-subject erasure transforms (CR-2). Pure logic — no DB."""
from __future__ import annotations

import pytest

from navis.signals.governance import (
    TOMBSTONE,
    content_retention_days,
    scrub_content_fields,
    scrub_subject,
    subject_matches,
)


# ---- CR-1: FIELD_ENCRYPTION_KEYS must be a production startup hard-fail ----

def test_field_cipher_hard_fails_in_production_without_keys(monkeypatch):
    import navis.platform.config as cfg
    from navis.platform.security.crypto import FieldCipher

    monkeypatch.delenv("FIELD_ENCRYPTION_KEYS", raising=False)
    monkeypatch.setattr(cfg.settings, "field_encryption_keys", None)
    monkeypatch.setattr(cfg.settings, "environment", "production")
    with pytest.raises(RuntimeError, match="FIELD_ENCRYPTION_KEYS"):
        FieldCipher.from_env()


def test_field_cipher_passthrough_allowed_in_dev(monkeypatch):
    import navis.platform.config as cfg
    from navis.platform.security.crypto import FieldCipher

    monkeypatch.delenv("FIELD_ENCRYPTION_KEYS", raising=False)
    monkeypatch.setattr(cfg.settings, "field_encryption_keys", None)
    monkeypatch.setattr(cfg.settings, "environment", "development")
    assert FieldCipher.from_env().enabled is False  # warns, does not raise


def test_field_cipher_health(monkeypatch):
    import navis.platform.config as cfg
    from navis.platform.security.crypto import field_encryption_healthy

    monkeypatch.setattr(cfg.settings, "environment", "development")
    assert field_encryption_healthy() is True  # dev tolerates passthrough


# ---- M-1: retention policy + tombstone ----

def test_retention_windows_by_class_and_source_fallback():
    assert content_retention_days("slack", "slack_message") == 90
    assert content_retention_days("slack", "slack_reaction") == 30
    assert content_retention_days("slack", "slack_identity") is None    # snapshots: not TTL'd
    assert content_retention_days("gmail", None) == 90                  # source fallback (no class)
    assert content_retention_days("calendar", None) is None


def test_scrub_content_is_idempotent():
    payload = {"text": "fenc:secret", "channel_id": "C1", "_content_fields": ["text"]}
    once, changed = scrub_content_fields(payload)
    assert changed and once["text"] == TOMBSTONE and once["_scrubbed"] is True
    assert "_content_fields" not in once
    assert once["channel_id"] == "C1"          # non-content kept (channel is not personal data)
    twice, changed2 = scrub_content_fields(once)
    assert changed2 is False and twice == once  # idempotent -> replay/retry safe


# ---- CR-2: subject match + scrub ----

def test_subject_match_by_email_and_slack_user():
    assert subject_matches({"sender_email": "a@x.com"}, email="A@X.com", slack_user_id=None)
    assert subject_matches({"to_emails": ["b@x.com", "a@x.com"]}, email="a@x.com", slack_user_id=None)
    assert subject_matches({"slack_user_id": "U1"}, email=None, slack_user_id="U1")
    assert not subject_matches({"sender_email": "z@x.com"}, email="a@x.com", slack_user_id=None)


def test_scrub_subject_erases_one_person_keeps_others():
    payload = {
        "text": "fenc:body", "_content_fields": ["text"],
        "sender_email": "a@x.com", "to_emails": ["a@x.com", "keep@x.com"],
        "slack_user_id": "U1",
    }
    out, changed = scrub_subject(payload, email="a@x.com", slack_user_id="U1")
    assert changed
    assert out["text"] == TOMBSTONE                       # content gone
    assert out["sender_email"] == TOMBSTONE               # subject's own identity gone
    assert out["to_emails"] == [TOMBSTONE, "keep@x.com"]  # OTHER recipients preserved
    assert out["slack_user_id"] == TOMBSTONE


def test_erase_subject_requires_an_identifier():
    import asyncio

    from navis.signals.governance import erase_subject

    with pytest.raises(ValueError):
        asyncio.run(erase_subject(None, __import__("uuid").uuid4()))  # type: ignore[arg-type]


# ---- P0: exact matching only (no substring / alias / domain collision) ----

def test_email_match_is_exact_not_substring():
    # exact succeeds
    assert subject_matches({"sender_email": "alice@example.com"}, email="alice@example.com", slack_user_id=None)
    assert subject_matches({"to_emails": ["x@y.com", "alice@example.com"]}, email="ALICE@example.com", slack_user_id=None)
    # domain-suffix collision FAILS
    assert not subject_matches({"sender_email": "alice@example.com.attacker.com"},
                               email="alice@example.com", slack_user_id=None)
    # plus-alias / local-part collision FAILS
    assert not subject_matches({"to_emails": ["bob+alice@example.com"]},
                               email="alice@example.com", slack_user_id=None)
    # a raw header is NEVER matched (audit-only), even on exact content
    assert not subject_matches({"from": "alice@example.com", "to": "alice@example.com"},
                               email="alice@example.com", slack_user_id=None)


# ---- P1: BCC + content tombstoned on erasure ----

def test_erasure_tombstones_bcc_content_field():
    # Gmail seals raw `bcc` as a content field; scrub_subject must tombstone it.
    payload = {
        "sender_email": "a@x.com", "to_emails": ["a@x.com"],
        "bcc": "fenc:hidden", "body_text": "fenc:body", "_content_fields": ["bcc", "body_text"],
    }
    out, changed = scrub_subject(payload, email="a@x.com", slack_user_id=None)
    assert changed
    assert out["bcc"] == TOMBSTONE and out["body_text"] == TOMBSTONE  # raw bcc + body gone
    assert out["sender_email"] == TOMBSTONE and out["to_emails"] == [TOMBSTONE]


# ---- P1: retention fails LOUD on unknown source, never silent None ----

def test_unknown_source_is_loud_not_silent(monkeypatch):
    from navis.platform.monitoring.metrics import metrics
    import navis.signals.governance as gov

    seen = []
    monkeypatch.setattr(metrics, "incr", lambda name, **k: seen.append(name))
    assert gov.content_retention_days("brand_new_connector", None) is None  # conservative keep
    assert "signals.retention.unknown_source" in seen                       # but LOUD


# ---- P1: cascade contract emission seam ----

async def test_subject_erasure_cascade_seam():
    import navis.signals.governance as gov

    received = []

    async def _sub(notice, db):
        received.append(notice)

    saved = list(gov._ERASURE_SUBSCRIBERS)
    gov._ERASURE_SUBSCRIBERS.clear()  # isolate from globally-registered subscribers
    gov.on_subject_erasure(_sub)
    try:
        await gov._notify_erasure(
            gov.SubjectErasureNotice(org_id=__import__("uuid").uuid4(), subject_handle="subj:x", erased=3),
            None,  # type: ignore[arg-type]
        )
        assert received and received[0].erased == 3
    finally:
        gov._ERASURE_SUBSCRIBERS[:] = saved


# ---- P2: audit handle is keyed (HMAC), non-reversible, deterministic ----

def test_subject_handle_is_keyed_and_non_reversible():
    from navis.signals.governance import _subject_handle

    h = _subject_handle("alice@example.com", None)
    assert h.startswith("subj:") and "alice" not in h and "example.com" not in h
    assert h == _subject_handle("ALICE@example.com", None)            # deterministic + normalized
    assert h != _subject_handle("bob@example.com", None)              # distinct subjects differ
    # keyed: a bare sha256 of the same input must NOT equal our handle (proves the secret salt)
    import hashlib
    bare = "subj:" + hashlib.sha256(b"alice@example.com|").hexdigest()[:32]
    assert h != bare
