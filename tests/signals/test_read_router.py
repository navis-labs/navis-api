"""Read router shape + decrypt contract.

The endpoints themselves are thin SQL adapters — the risk-carrying logic is
(1) transparent decryption of sealed content fields and (2) the per-source
payload shaping the frontend depends on. These tests pin both against
regression. Full end-to-end SQL is exercised in the Postgres integration
suite (DISTINCT ON, tenant scoping); here we lock the pure-Python contract.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from navis.platform.security import field_cipher
from navis.signals import read_router as R
from navis.signals.base import SignalEventType


@dataclass
class _Row:
    """SignalEventRecord stand-in — the shapers only read these attributes."""
    id: uuid.UUID
    external_id: str
    occurred_at: datetime | None
    payload: dict[str, Any]
    event_type: str = SignalEventType.EMAIL_RECEIVED.value


# ── _decrypt_content ────────────────────────────────────────────────────────

def test_decrypt_content_no_sealed_fields_is_identity():
    """A payload without `_content_fields` (e.g. calendar / drive) MUST pass
    through unchanged — decryption is a Gmail/Slack-only concern."""
    p = {"summary": "1:1 with Alice", "attendees": ["a@x.com"]}
    assert R._decrypt_content(p) == p


def test_decrypt_content_roundtrips_sealed_gmail_fields():
    """`_content_fields` is the self-description a Gmail signal carries. The
    reader must decrypt each listed field transparently so no consumer has to
    know the storage-at-rest scheme."""
    body = "Confirming the demo at 3pm"
    snippet = "Confirming the demo…"
    sealed = {
        "subject": "Re: demo",
        "body_text": field_cipher.encrypt(body),
        "snippet":   field_cipher.encrypt(snippet),
        "_content_fields": ["body_text", "snippet"],
    }
    out = R._decrypt_content(sealed)
    assert out["body_text"] == body and out["snippet"] == snippet
    # Original payload is not mutated — the DB copy must never be touched.
    assert sealed["body_text"] != body


def test_decrypt_content_degrades_on_decrypt_failure(monkeypatch):
    """A rotated-out key OR tampered ciphertext MUST NOT 500 the whole list.
    The one bad field degrades to empty; all sibling fields keep decrypting."""

    class _BoomCipher:
        def decrypt(self, v):
            from navis.platform.security import FieldDecryptError
            raise FieldDecryptError("rotated out")

    monkeypatch.setattr(R, "field_cipher", _BoomCipher())
    out = R._decrypt_content({
        "body_text": "cipher-bytes",
        "snippet":   "cipher-bytes",
        "_content_fields": ["body_text", "snippet"],
    })
    # Both fields degraded to empty — no exception propagated.
    assert out["body_text"] == "" and out["snippet"] == ""


# ── _shape_gmail ────────────────────────────────────────────────────────────

def test_shape_gmail_curates_the_frontend_fields():
    when = datetime(2026, 7, 25, 9, 30, tzinfo=timezone.utc)
    row = _Row(
        id=uuid.uuid4(),
        external_id="msg-123",
        occurred_at=when,
        payload={
            "subject":       "Board update",
            "sender_email":  "chair@acme.com",
            "sender_name":   "The Chair",
            "to_emails":     ["founder@navis.io"],
            "snippet":       "TL;DR — we approved…",
            "thread_id":     "thr-1",
            "label_ids":     ["INBOX", "IMPORTANT"],
            "attachments":   [{"filename": "deck.pdf"}],
        },
    )
    shaped = R._shape_gmail(row)
    assert shaped["subject"] == "Board update"
    assert shaped["from_email"] == "chair@acme.com"
    assert shaped["from_name"] == "The Chair"
    assert shaped["to_emails"] == ["founder@navis.io"]
    assert shaped["thread_id"] == "thr-1"
    assert shaped["has_attachments"] is True
    assert shaped["received_at"] == when.isoformat()
    assert shaped["external_id"] == "msg-123"
    # Sensitive routing fields (bcc raw header, encrypted bcc_emails) MUST NOT
    # leak — the endpoint intentionally omits them.
    assert "bcc" not in shaped and "bcc_emails" not in shaped


def test_shape_gmail_missing_attachments_reports_false():
    row = _Row(id=uuid.uuid4(), external_id="m", occurred_at=None, payload={})
    assert R._shape_gmail(row)["has_attachments"] is False


# ── _shape_calendar ────────────────────────────────────────────────────────

def test_shape_calendar_flattens_conference_and_organizer():
    when = datetime(2026, 8, 1, 15, 0, tzinfo=timezone.utc)
    row = _Row(
        id=uuid.uuid4(),
        external_id="evt-9",
        occurred_at=when,
        payload={
            "summary":         "Design review",
            "status":          "confirmed",
            "start":           {"dateTime": when.isoformat()},
            "end":             {"dateTime": (when + timedelta(hours=1)).isoformat()},
            "attendee_count":  4,
            "organizer":       {"email": "eng-lead@acme.com"},
            "conference_data": {"conference_id": "abc", "has_video": True},
            "hangout_link":    "https://meet.google.com/xyz-abc",
        },
        event_type=SignalEventType.CALENDAR_EVENT_CREATED.value,
    )
    shaped = R._shape_calendar(row)
    assert shaped["summary"] == "Design review"
    assert shaped["attendees_count"] == 4
    assert shaped["organizer_email"] == "eng-lead@acme.com"
    assert shaped["has_conference"] is True
    assert shaped["hangout_link"].endswith("xyz-abc")


def test_shape_calendar_has_conference_is_true_for_hangout_only():
    """`conference_data` may be None while `hangout_link` is set (legacy events).
    The frontend chip lights up for either — one truthy indicator is enough."""
    row = _Row(
        id=uuid.uuid4(),
        external_id="evt-legacy",
        occurred_at=datetime.now(timezone.utc),
        payload={"hangout_link": "https://meet.google.com/legacy"},
    )
    assert R._shape_calendar(row)["has_conference"] is True


# ── _shape_drive ────────────────────────────────────────────────────────────

def test_shape_drive_extracts_owner_emails_and_status_flags():
    when = datetime(2026, 7, 20, tzinfo=timezone.utc)
    row = _Row(
        id=uuid.uuid4(),
        external_id="file-abc",
        occurred_at=when,
        payload={
            "name":      "Q3 forecast.xlsx",
            "mime_type": "application/vnd.google-apps.spreadsheet",
            "owners":    [{"email": "cfo@acme.com"}, {"email": None}, {}],
            "shared":    True,
            "trashed":   False,
        },
        event_type=SignalEventType.DRIVE_FILE_UPDATED.value,
    )
    shaped = R._shape_drive(row)
    assert shaped["name"] == "Q3 forecast.xlsx"
    assert shaped["owners"] == ["cfo@acme.com"]  # None + {} filtered out
    assert shaped["shared"] is True
    assert shaped["event_type"] == SignalEventType.DRIVE_FILE_UPDATED.value
    assert shaped["modified_at"] == when.isoformat()


# ── _score_email ───────────────────────────────────────────────────────────

def test_score_email_hard_skips_promotions():
    """CATEGORY_PROMOTIONS is 100% junk for a 'top important' list — the
    scorer must return None so the endpoint drops it entirely, not merely
    down-rank it below other traffic."""
    assert R._score_email({"label_ids": ["CATEGORY_PROMOTIONS", "INBOX"]}) is None


def test_score_email_hard_skips_social():
    assert R._score_email({"label_ids": ["CATEGORY_SOCIAL"]}) is None


def test_score_email_hard_skips_mailing_list():
    """List-ID header means 'this came off a mailing list' — newsletters,
    digests, ops-list broadcasts. Never important for the top-10 view."""
    assert R._score_email({"list_id": "<devs.list@acme.com>"}) is None


def test_score_email_hard_skips_auto_submitted():
    """RFC 3834 auto-submitted mail is bots / CI / monitoring. Skip."""
    assert R._score_email({"auto_submitted": "auto-generated"}) is None


def test_score_email_hard_skips_bulk_precedence():
    assert R._score_email({"precedence": "bulk"}) is None


def test_score_email_important_label_dominates():
    """Gmail's IMPORTANT label encodes Google's own inbox ML — it must
    outweigh a bare inbox email with nothing else going for it."""
    plain = R._score_email({"label_ids": ["INBOX"], "sender_email": "x@acme.com"})
    important = R._score_email({"label_ids": ["INBOX", "IMPORTANT"], "sender_email": "x@acme.com"})
    assert important > plain
    assert important >= 10  # IMPORTANT alone is worth 10


def test_score_email_starred_is_strong():
    """User explicitly starred the mail — a stronger signal than any
    heuristic we compute, but slightly under Google's own IMPORTANT."""
    starred = R._score_email({"label_ids": ["STARRED"]})
    assert starred is not None and starred >= 8


def test_score_email_commitment_language_boosts():
    """A promise from the sender ('I'll send it by EOD') means the thread
    is actionable — outrank a plain inbox mail from the same domain."""
    plain = R._score_email({"label_ids": ["INBOX"], "body_text": "hello"})
    commit = R._score_email({
        "label_ids": ["INBOX"],
        "body_text": "I'll send the deck by EOD tomorrow.",
    })
    assert commit > plain


def test_score_email_decision_language_boosts_more_than_commitment():
    """A 'we decided' / 'approved' beats a mere commitment — it's a fact
    the org needs to record, not a pending action."""
    commit = R._score_email({"body_text": "I'll follow up next week."})
    decision = R._score_email({"body_text": "We've decided to go with option B."})
    assert decision > commit


def test_score_email_free_domain_gets_no_business_boost():
    """A @gmail.com sender is more often personal noise than a real
    counterparty — the domain bonus applies only to business domains."""
    biz = R._score_email({"sender_email": "ceo@acme.com", "label_ids": ["INBOX"]})
    free = R._score_email({"sender_email": "someone@gmail.com", "label_ids": ["INBOX"]})
    assert biz > free


def test_score_email_direct_addressed_beats_mass_cc():
    """One recipient, no CC → for-you-specifically. 20 people on CC →
    broadcast FYI. The scorer has to distinguish the two."""
    direct = R._score_email({"to_emails": ["me@acme.com"], "cc_emails": []})
    mass = R._score_email({
        "to_emails": ["a@x.com"] * 8,
        "cc_emails": ["b@x.com"] * 8,
    })
    assert direct > mass


def test_score_email_missing_fields_are_safe():
    """A minimal payload (no body, no labels, no headers) must score
    without raising — real payloads sometimes miss optional fields."""
    assert R._score_email({}) == 0


# ── _important_sort_key ────────────────────────────────────────────────────

def test_important_sort_orders_newest_first_regardless_of_score():
    """The regression the user hit: score-first sort surfaced 4-week-old
    emails above yesterday's mail because IMPORTANT/STARRED labels
    accumulate over time. Time MUST win the sort so the feed walks
    backwards chronologically."""
    old_hi_score = _Row(
        id=uuid.uuid4(), external_id="old",
        occurred_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        payload={},
    )
    fresh_lo_score = _Row(
        id=uuid.uuid4(), external_id="fresh",
        occurred_at=datetime(2026, 7, 25, tzinfo=timezone.utc),
        payload={},
    )
    entries = [(21, old_hi_score, {}), (8, fresh_lo_score, {})]
    entries.sort(key=R._important_sort_key, reverse=True)
    assert entries[0][1].external_id == "fresh"
    assert entries[1][1].external_id == "old"


def test_important_sort_score_breaks_ties_on_same_timestamp():
    """When two emails share the exact occurred_at (batched ingest),
    score picks the winner — deterministic ordering matters for a
    paged UI that trusts array position."""
    ts = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
    lo = _Row(id=uuid.uuid4(), external_id="lo", occurred_at=ts, payload={})
    hi = _Row(id=uuid.uuid4(), external_id="hi", occurred_at=ts, payload={})
    entries = [(5, lo, {}), (12, hi, {})]
    entries.sort(key=R._important_sort_key, reverse=True)
    assert entries[0][1].external_id == "hi"


def test_important_sort_tolerates_missing_occurred_at():
    """A row with `occurred_at=None` (malformed capture) MUST NOT raise a
    naive-vs-aware TypeError in the sort key. It sinks to the bottom
    of the list — never blocking the endpoint."""
    real = _Row(
        id=uuid.uuid4(), external_id="real",
        occurred_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
        payload={},
    )
    missing = _Row(id=uuid.uuid4(), external_id="missing", occurred_at=None, payload={})
    entries = [(20, missing, {}), (5, real, {})]
    # Should not raise even though scores would suggest missing comes first.
    entries.sort(key=R._important_sort_key, reverse=True)
    assert entries[0][1].external_id == "real"
    assert entries[1][1].external_id == "missing"


# ── _RowShim ────────────────────────────────────────────────────────────────

def test_row_shim_proxies_dict_lookups_as_attributes():
    """The DISTINCT ON path returns SQLAlchemy row mappings, not ORM
    instances. The shim adapts them to the same attribute interface the
    shapers use so one shaper covers both code paths."""
    row = {"id": "u1", "external_id": "x", "payload": {"k": "v"}}
    shim = R._RowShim(row)
    assert shim.id == "u1" and shim.external_id == "x"
    assert shim.payload == {"k": "v"}
    with pytest.raises(AttributeError):
        _ = shim.does_not_exist
