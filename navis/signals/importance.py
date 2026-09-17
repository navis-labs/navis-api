"""Deterministic signal importance — how much does this message matter to a founder?

A real founder inbox is ~90% newsletters, receipts, marketing and notifications.
The instinct is to exclude that mail, but measurement showed exclusion is wrong:
in the reference corpus every calendar RSVP and every invoice is *automated*, so
dropping automated mail deletes the meetings. "Automated" and "important" are
orthogonal axes — 22 signals in that corpus are BOTH.

So nothing is excluded. Every signal is scored, once, and the score is recorded
beside the log. What consumes it is a separate decision (Phase 2); this module
only exposes a stable number.

CONTRACT
  * Pure — a function of the payload alone. No clock, no model, no randomness,
    no database. The same signal always yields the same score, so a replay
    reproduces the corpus exactly.
  * Versioned — `IMPORTANCE_VERSION` changes whenever the policy changes, and
    scores are stored per version, so an old and a new policy coexist and a
    re-score is a migration you can compare against, not a destructive edit.
  * Non-destructive — writes to its own table; `signal_events` stays immutable
    and no memory is filtered, reordered, or deleted.

The class boundaries and base scores below were validated against the reference
corpus before being written down: the ranking put founder decision, meetings,
human replies and invoices at the top, and Groww digests / Facebook codes /
survey blasts at the bottom.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from email.utils import parseaddr
from typing import Any

from navis.signals.base import SignalSource
from navis.signals.content import decrypt_content_fields
from navis.signals.text_rules import (
    is_commitment_sentence,
    is_decision_sentence,
    sentences,
)

# Bump when the policy below changes. Scores are stored per version so a new
# policy can be computed and compared BEFORE anything starts consuming it.
IMPORTANCE_VERSION = 1

# How much a signal matters, 0.0 (noise) .. 1.0 (a founder decision).
NEUTRAL = 0.50


class ImportanceClass:
    """The vocabulary. Stable strings — they are persisted and queried."""

    FOUNDER_DECISION = "founder_decision"
    COMMITMENT = "commitment"
    MEETING = "meeting"
    HUMAN_REPLY = "human_reply"
    ACTION_REQUEST = "action_request"
    DEV = "dev"
    INVOICE = "invoice"
    HUMAN_OTHER = "human_other"
    DOCUMENT = "document"
    AUTOMATED_OTHER = "automated_other"
    NEWSLETTER = "newsletter"
    OTP = "otp"
    SOCIAL = "social"
    MARKETING = "marketing"
    SPAM = "spam"


# Base score per class. Ordered high → low for readability; the cascade below
# decides which one applies.
BASE_SCORES: dict[str, float] = {
    ImportanceClass.FOUNDER_DECISION: 1.00,
    ImportanceClass.COMMITMENT: 0.90,
    ImportanceClass.MEETING: 0.85,
    ImportanceClass.HUMAN_REPLY: 0.80,
    ImportanceClass.ACTION_REQUEST: 0.75,
    ImportanceClass.DEV: 0.65,
    ImportanceClass.INVOICE: 0.60,
    ImportanceClass.HUMAN_OTHER: 0.55,
    ImportanceClass.DOCUMENT: 0.50,
    ImportanceClass.AUTOMATED_OTHER: 0.25,
    ImportanceClass.NEWSLETTER: 0.20,
    ImportanceClass.OTP: 0.15,
    ImportanceClass.SOCIAL: 0.10,
    ImportanceClass.MARKETING: 0.10,
    ImportanceClass.SPAM: 0.00,
}

# Evidence that is independent of the class cascade.
#   IMPORTANT is Gmail's own classifier — measured to be orthogonal to our
#   automated/human split, which is exactly why it earns a say.
#   TRASH is a weak negative: the founder discarded it, but the corpus showed a
#   genuine compliance notice sitting in TRASH, so it nudges rather than buries.
BONUS_IMPORTANT = 0.15
PENALTY_TRASH = -0.10
BONUS_ATTACHMENT = 0.05

_CAL_SUBJECT_RE = re.compile(
    r"^(Accepted|Declined|Tentative|Invitation|Updated invitation|Canceled event|Cancelled event):",
    re.IGNORECASE,
)
_PAYMENT_RE = re.compile(
    r"\b(invoice|receipt|repayment|billing|amount due|payment (due|failed|reminder|successful)"
    r"|subscription (renew|renewal)|transaction)\b",
    re.IGNORECASE,
)
_DEV_RE = re.compile(
    r"\b(pull request|PR #\d|merged|commit|issue #\d|build (failed|passed)|CI|deploy(ed|ment)?"
    r"|code review|branch)\b",
    re.IGNORECASE,
)
_OTP_RE = re.compile(
    r"\b(authentication code|verification code|one[- ]time (code|password)|OTP"
    r"|security code|sign-?in code|login code|2FA)\b",
    re.IGNORECASE,
)
# Request language ONLY counts from a human — marketing says "review our offer".
_REQUEST_RE = re.compile(
    r"\b(please (review|confirm|approve|send|share)|could you|can you|kindly"
    r"|need your (input|approval|sign)|sign off|let me know by|awaiting your)\b",
    re.IGNORECASE,
)

_BULK_PRECEDENCE = {"bulk", "list", "junk"}
_BULK_LABELS = {
    "CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS",
    "CATEGORY_UPDATES", "SPAM",
}
_NOREPLY_LOCALPART_RE = re.compile(
    r"^(?:no[-_.]?reply|do[-_.]?not[-_.]?reply|donotreply|mailer[-_.]?daemon|postmaster"
    r"|bounces?|notifications?|newsletters?|alerts?|automated|mailer)(?:[-+._]|$)",
    re.IGNORECASE,
)
_BODY_SCAN_CHARS = 4000  # bound the text we inspect; bodies reach 20KB+


@dataclass(frozen=True, slots=True)
class Importance:
    """One scored signal. `features` records WHY, so a score is auditable."""

    classification: str
    score: float
    version: int = IMPORTANCE_VERSION
    features: dict[str, Any] = field(default_factory=dict)


def _labels(payload: dict) -> set[str]:
    out: set[str] = set()
    for key in ("label_ids", "labels"):
        raw = payload.get(key)
        if isinstance(raw, list):
            out.update(str(v) for v in raw if isinstance(v, (str, int)))
    return out


def is_automated(payload: dict) -> bool:
    """Bulk/machine-sent rather than written by a person. NOT a proxy for unimportant."""
    if str(payload.get("list_id") or "").strip():
        return True  # List-Id => mailing list
    if str(payload.get("precedence") or "").strip().lower() in _BULK_PRECEDENCE:
        return True
    auto = str(payload.get("auto_submitted") or "").strip().lower()
    if auto and auto != "no":
        return True  # RFC 3834
    if _labels(payload) & _BULK_LABELS:
        return True
    _name, sender = parseaddr(str(payload.get("from", "")))
    local = sender.split("@", 1)[0] if "@" in sender else ""
    return bool(local and _NOREPLY_LOCALPART_RE.match(local))


def _body_text(payload: dict) -> str:
    """Plain-ish body for scanning. HTML de-noising is Memory's job; here we only
    need enough text for the sentence rules, so a cheap tag strip is sufficient."""
    body = payload.get("body_text") or payload.get("snippet") or ""
    if not isinstance(body, str) or not body:
        return ""
    if body.startswith("fenc:"):
        return ""  # still sealed — caller did not decrypt; never score ciphertext
    if "<" in body and ">" in body:
        body = re.sub(r"<[^>]+>", " ", body)
    return body[:_BODY_SCAN_CHARS]


def _classify_email(payload: dict) -> tuple[str, dict[str, Any]]:
    """Priority cascade — the FIRST matching class wins, strongest evidence first."""
    labels = _labels(payload)
    subject = str(payload.get("subject") or "")
    automated = is_automated(payload)
    body = _body_text(payload)
    facts: dict[str, Any] = {"automated": automated}

    if "SPAM" in labels:
        return ImportanceClass.SPAM, facts

    # Content-derived classes require a human author: a newsletter saying
    # "we'll tackle them live" is not a commitment to this organization.
    if not automated and body:
        lines = sentences(body)
        if any(is_decision_sentence(s) for s in lines):
            facts["matched"] = "decision_sentence"
            return ImportanceClass.FOUNDER_DECISION, facts
        if any(is_commitment_sentence(s) for s in lines):
            facts["matched"] = "commitment_sentence"
            return ImportanceClass.COMMITMENT, facts

    if _CAL_SUBJECT_RE.search(subject):
        facts["matched"] = "calendar_subject"
        return ImportanceClass.MEETING, facts
    if payload.get("in_reply_to") and not automated:
        facts["matched"] = "in_reply_to"
        return ImportanceClass.HUMAN_REPLY, facts
    if not automated and _REQUEST_RE.search(body):
        facts["matched"] = "request_language"
        return ImportanceClass.ACTION_REQUEST, facts
    if _DEV_RE.search(subject):
        facts["matched"] = "dev_subject"
        return ImportanceClass.DEV, facts
    if _PAYMENT_RE.search(subject):
        facts["matched"] = "payment_subject"
        return ImportanceClass.INVOICE, facts
    if _OTP_RE.search(subject):
        facts["matched"] = "otp_subject"
        return ImportanceClass.OTP, facts
    if "CATEGORY_SOCIAL" in labels:
        return ImportanceClass.SOCIAL, facts
    if "CATEGORY_PROMOTIONS" in labels:
        return ImportanceClass.MARKETING, facts
    if str(payload.get("list_id") or "").strip():
        return ImportanceClass.NEWSLETTER, facts
    if not automated:
        return ImportanceClass.HUMAN_OTHER, facts
    return ImportanceClass.AUTOMATED_OTHER, facts


# Non-email sources carry their own inherent meaning; a calendar event IS a
# meeting, a Drive file IS a document. Only Gmail needs the cascade.
_SOURCE_DEFAULTS = {
    SignalSource.CALENDAR: ImportanceClass.MEETING,
    SignalSource.MEET: ImportanceClass.MEETING,
    SignalSource.DRIVE: ImportanceClass.DOCUMENT,
    SignalSource.DOCS: ImportanceClass.DOCUMENT,
}


def score_signal(source: SignalSource, payload: dict[str, Any]) -> Importance:
    """Score one signal. Pure and total — never raises on a malformed payload.

    `payload` must already be decrypted (see `decrypt_content_fields`); a sealed
    body is treated as absent rather than scored as ciphertext.
    """
    payload = payload or {}
    if source is SignalSource.GMAIL:
        classification, facts = _classify_email(payload)
    elif source in _SOURCE_DEFAULTS:
        classification, facts = _SOURCE_DEFAULTS[source], {"automated": False}
    else:
        # Slack and anything future: no validated cascade yet, so sit at neutral
        # rather than inventing a ranking we have not measured.
        return Importance(ImportanceClass.HUMAN_OTHER, NEUTRAL, IMPORTANCE_VERSION,
                          {"source": source.value, "default": True})

    labels = _labels(payload)
    score = BASE_SCORES[classification]
    if "IMPORTANT" in labels:
        score += BONUS_IMPORTANT
        facts["gmail_important"] = True
    if "TRASH" in labels:
        score += PENALTY_TRASH
        facts["trashed"] = True
    if payload.get("attachments"):
        score += BONUS_ATTACHMENT
        facts["has_attachment"] = True

    facts["source"] = source.value
    return Importance(classification, max(0.0, min(1.0, round(score, 3))),
                      IMPORTANCE_VERSION, facts)


def score_payload(source: SignalSource, raw_payload: dict[str, Any]) -> Importance:
    """Convenience: decrypt sealed content, then score. Use on stored signals."""
    return score_signal(source, decrypt_content_fields(raw_payload or {}))
