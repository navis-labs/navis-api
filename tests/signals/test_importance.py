"""Deterministic signal importance (Phase 1).

The measurement that drove this design: in the reference corpus every calendar
RSVP and every invoice is *automated*, so excluding automated mail would delete
the meetings. Importance is a separate axis from automated-ness, and these tests
pin that separation.
"""
from __future__ import annotations

import uuid

import pytest

from navis.signals.base import SignalSource
from navis.signals.importance import (
    IMPORTANCE_VERSION,
    ImportanceClass,
    is_automated,
    score_payload,
    score_signal,
)
from navis.signals.models import compute_importance_id


def gmail(**over):
    payload = {"from": "Alice <alice@acme.com>", "to": "founder@navis.com",
               "subject": "hello", "label_ids": ["CATEGORY_PERSONAL"]}
    payload.update(over)
    return payload


def s(payload):
    return score_signal(SignalSource.GMAIL, payload)


# ── automated is NOT a proxy for unimportant ───────────────────────────────


def test_calendar_rsvp_is_important_despite_being_automated():
    """The finding that killed the exclusion proposal: RSVPs are automated."""
    p = gmail(subject="Accepted: partnership @ Wed Aug 5, 2026 9am - 9:30am (IST)",
              **{"auto_submitted": "auto-generated"})
    assert is_automated(p) is True
    r = s(p)
    assert r.classification == ImportanceClass.MEETING
    assert r.score >= 0.8


def test_invoice_is_mid_importance_despite_being_automated():
    p = gmail(subject="Payment reminder", **{"from": "slice <noreply@slice.bank.in>"})
    assert is_automated(p) is True
    r = s(p)
    assert r.classification == ImportanceClass.INVOICE
    assert 0.5 <= r.score <= 0.8


def test_marketing_scores_low():
    r = s(gmail(subject="Monsoon Madness Ends Today!", label_ids=["CATEGORY_PROMOTIONS"]))
    assert r.classification == ImportanceClass.MARKETING
    assert r.score <= 0.2


def test_spam_scores_zero():
    r = s(gmail(subject="New Survey Alert", label_ids=["SPAM"]))
    assert r.classification == ImportanceClass.SPAM
    assert r.score == 0.0


def test_otp_scores_low_even_from_a_human_looking_domain():
    """19 of the corpus OTPs came from a human-looking domain; subject decides."""
    r = s(gmail(subject="Your authentication code", **{"from": "Arc <thilak.jk@abmrc.com>"}))
    assert r.classification == ImportanceClass.OTP
    assert r.score <= 0.35


# ── the high classes ───────────────────────────────────────────────────────


def test_founder_decision_is_the_ceiling():
    r = s(gmail(subject="Decision: Q3 Launch Timeline Updated",
                body_text="After reviewing progress, we've decided to postpone the Q3 launch."))
    assert r.classification == ImportanceClass.FOUNDER_DECISION
    assert r.score == 1.0


def test_commitment_from_a_human():
    r = s(gmail(body_text="Thanks. I'll send the pricing deck by Friday."))
    assert r.classification == ImportanceClass.COMMITMENT


def test_newsletter_commitment_language_is_not_a_commitment():
    """'we'll tackle them live' in a digest is not a promise to this org."""
    r = s(gmail(subject="Daily digest", label_ids=["CATEGORY_PROMOTIONS"],
                **{"from": "PH <hello@digest.producthunt.com>", "list_id": "<ph.digest>"},
                body_text="Come ready with your challenges, we'll tackle them live."))
    assert r.classification != ImportanceClass.COMMITMENT
    assert r.score <= 0.25


def test_human_reply_beats_human_other():
    reply = s(gmail(in_reply_to="<abc@acme.com>"))
    plain = s(gmail())
    assert reply.classification == ImportanceClass.HUMAN_REPLY
    assert reply.score > plain.score


def test_request_language_only_counts_from_a_human():
    human = s(gmail(body_text="Could you review the Q3 forecast?"))
    assert human.classification == ImportanceClass.ACTION_REQUEST
    promo = s(gmail(body_text="Could you review our new offer?",
                    label_ids=["CATEGORY_PROMOTIONS"]))
    assert promo.classification == ImportanceClass.MARKETING


# ── modifiers ──────────────────────────────────────────────────────────────


def test_gmail_important_label_lifts_score():
    base = s(gmail(subject="Payment reminder"))
    lifted = s(gmail(subject="Payment reminder", label_ids=["IMPORTANT"]))
    assert lifted.score > base.score


def test_trash_is_a_nudge_not_a_burial():
    """A genuine compliance notice was found sitting in TRASH — it must not go to 0."""
    trashed = s(gmail(subject="Payment reminder", label_ids=["TRASH"]))
    assert 0.0 < trashed.score < s(gmail(subject="Payment reminder")).score


def test_score_is_clamped_to_unit_range():
    r = s(gmail(subject="Decision: x", body_text="We've decided to ship.",
                label_ids=["IMPORTANT"], attachments=[{"n": 1}]))
    assert 0.0 <= r.score <= 1.0


# ── contract: pure, versioned, replayable ──────────────────────────────────


def test_scoring_is_deterministic():
    p = gmail(subject="Accepted: sync @ Mon", body_text="ok")
    assert s(p) == s(p)


def test_scoring_never_raises_on_malformed_payloads():
    for bad in ({}, {"from": None}, {"label_ids": "not-a-list"},
                {"subject": 12345}, {"body_text": None}):
        r = score_signal(SignalSource.GMAIL, bad)
        assert 0.0 <= r.score <= 1.0


def test_sealed_body_is_never_scored_as_content():
    """If decryption was skipped, ciphertext must not be mistaken for prose."""
    r = s(gmail(body_text="fenc:1:gAAAAABqchCaucAiqUJ9jIx2V7A9cQ"))
    assert r.classification not in (ImportanceClass.COMMITMENT,
                                    ImportanceClass.FOUNDER_DECISION)


def test_score_payload_decrypts_before_scoring():
    """Unsealed payloads pass through unchanged (no _content_fields => no-op)."""
    p = gmail(body_text="We've decided to postpone the launch.")
    assert score_payload(SignalSource.GMAIL, p).classification == ImportanceClass.FOUNDER_DECISION


def test_every_class_has_a_base_score():
    from navis.signals.importance import BASE_SCORES

    declared = {v for k, v in vars(ImportanceClass).items() if not k.startswith("_")}
    assert declared == set(BASE_SCORES), "a class without a score would KeyError at ingestion"


def test_version_is_recorded_on_every_result():
    assert s(gmail()).version == IMPORTANCE_VERSION


def test_importance_id_is_deterministic_and_version_scoped():
    sid = uuid.uuid4()
    assert compute_importance_id(sid, 1) == compute_importance_id(sid, 1)
    assert compute_importance_id(sid, 1) != compute_importance_id(sid, 2)
    assert compute_importance_id(uuid.uuid4(), 1) != compute_importance_id(sid, 1)


# ── non-email sources ──────────────────────────────────────────────────────


@pytest.mark.parametrize("source,expected", [
    (SignalSource.CALENDAR, ImportanceClass.MEETING),
    (SignalSource.MEET, ImportanceClass.MEETING),
    (SignalSource.DRIVE, ImportanceClass.DOCUMENT),
    (SignalSource.DOCS, ImportanceClass.DOCUMENT),
])
def test_non_email_sources_carry_inherent_class(source, expected):
    assert score_signal(source, {"summary": "Investor sync"}).classification == expected


def test_unmodelled_source_sits_at_neutral():
    """No validated cascade for Slack yet — don't invent a ranking."""
    r = score_signal(SignalSource.SLACK, {"text": "hello"})
    assert r.score == pytest.approx(0.5)
