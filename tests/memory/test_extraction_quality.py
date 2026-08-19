"""Extraction + embedding-text quality (P0 #2).

The audit found ~94% of the embedding corpus was opaque identifiers
("thread — 19f98866cbc10958", "person — a@b.com") and that the commitment regex
`\\bwe'?ll\\b` matched the ordinary word "well", so newsletter greetings became
organizational commitments. These tests pin both fixes.
"""
from __future__ import annotations

import pytest

from navis.memory.extractors.email import (
    CommitmentExtractor,
    DecisionExtractor,
    ThreadExtractor,
    classify_email,
    is_automated_email,
    is_commitment_sentence,
    is_decision_sentence,
    snippet_text,
    strip_quoted_reply,
)
from navis.ml.embeddings.service import memory_text
from tests.memory.fakes import gmail_view


class Rec:
    """Minimal stand-in for MemoryRecord — memory_text only reads these three."""

    def __init__(self, entity_type: str, payload: dict, dedup_key: str = "k"):
        self.entity_type = entity_type
        self.payload = payload
        self.dedup_key = dedup_key


# ── The "well" / "we'll" bug ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sentence",
    [
        "Everything was going well.",
        "I hope you are doing well.",
        "We hope this email finds you well.",
        "ROE tells us how well a company is using the shareholders money.",
        "It is a well-oiled machine with many parties involved.",
        "There's a Shared view on mobile as well, so every thread stays in sync.",
        "He is ill today and cannot join.",  # the twin defect: \bi'?ll\b matched "ill"
    ],
)
def test_ordinary_words_are_not_commitments(sentence):
    assert not is_commitment_sentence(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "I'll send the deck tomorrow.",
        "I’ll send the deck tomorrow.",  # typographic apostrophe
        "We'll get back to you by Friday.",
        "I will share the pricing model with you.",
        "We will follow up after the board meeting.",
        "Let me circle back on the contract terms.",
        "I'm going to send you the signed agreement.",
    ],
)
def test_real_commitments_still_extract(sentence):
    assert is_commitment_sentence(sentence)


def test_pleasantry_with_a_real_promise_is_kept():
    """Rejecting greetings must not swallow a promise in the same sentence."""
    assert is_commitment_sentence("Hope you are doing well, I'll send the deck today.")


# ── Decision tightening ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sentence",
    [
        # Bare "approved" used to match all of these newsletter/news sentences.
        "The government approved construction of a 14.52 km line in West Bengal.",
        "The Defence Acquisition Council approved proposals worth Rs 45,000 crore.",
        "a card scoped to one purchase, dead the second it clears",
        "the tool catching AI agents making product decisions nobody approved",
    ],
)
def test_news_prose_is_not_a_decision(sentence):
    assert not is_decision_sentence(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "We decided to go with the annual plan.",
        "We have agreed to postpone the launch to Q3.",
        "Decision: we are shipping on the 14th.",
        "We will go with Vendor B for payments.",
        "We approved the revised budget this morning.",
    ],
)
def test_real_decisions_still_extract(sentence):
    assert is_decision_sentence(sentence)


# ── Automated / bulk mail classification ───────────────────────────────────


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"from": "A <a@b.com>", "list_id": "<news.b.com>"}, True),
        ({"from": "A <a@b.com>", "precedence": "bulk"}, True),
        ({"from": "A <a@b.com>", "precedence": "Bulk"}, True),          # case-insensitive
        ({"from": "A <a@b.com>", "auto_submitted": "auto-generated"}, True),
        ({"from": "A <a@b.com>", "auto_submitted": "no"}, False),        # RFC 3834 "no"
        ({"from": "A <a@b.com>", "label_ids": ["CATEGORY_PROMOTIONS"]}, True),
        ({"from": "A <a@b.com>", "label_ids": ["CATEGORY_UPDATES"]}, True),
        ({"from": "A <a@b.com>", "labels": ["SPAM"]}, True),             # legacy key
        ({"from": "Digest <noreply@d.com>"}, True),                      # no-reply sender
        ({"from": "Alice <alice@acme.com>", "label_ids": ["CATEGORY_PERSONAL"]}, False),
        ({"from": "Alice <alice@acme.com>"}, False),
    ],
)
def test_is_automated_email(payload, expected):
    assert is_automated_email(payload) is expected


def test_classify_email_categories():
    assert classify_email({"from": "a@b.com", "label_ids": ["CATEGORY_PROMOTIONS"]}) == "promotional"
    assert classify_email({"from": "a@b.com", "label_ids": ["CATEGORY_UPDATES"]}) == "update"
    assert classify_email({"from": "a@b.com", "label_ids": ["CATEGORY_SOCIAL"]}) == "social"
    assert classify_email({"from": "noreply@b.com"}) == "automated"
    assert classify_email({"from": "alice@acme.com"}) == "personal"


def _email(**over):
    """A human email carrying one genuine commitment."""
    payload = {
        "from": "Alice <alice@acme.com>",
        "to": "founder@navis.com",
        "subject": "Q3 pricing",
        "thread_id": "t-1",
        "body_text": "Hi, I hope you are doing well. I'll send the pricing deck by Friday.",
        "label_ids": ["CATEGORY_PERSONAL"],
    }
    payload.update(over)
    return gmail_view(payload=payload)


def test_newsletter_produces_no_commitment_or_decision():
    """Suppression is per-CLAIM: the email is still remembered, just not as a promise."""
    view = gmail_view(payload={
        "from": "Product Hunt <hello@digest.producthunt.com>",
        "to": "founder@navis.com", "subject": "Daily digest", "thread_id": "t-2",
        "label_ids": ["CATEGORY_PROMOTIONS"],
        "body_text": "Come ready with your challenges, we'll tackle them live. "
                     "We decided to feature five winners.",
    })
    assert CommitmentExtractor().extract(view) == []
    assert DecisionExtractor().extract(view) == []
    # ...but the thread fact is still produced, classified as promotional.
    thread = ThreadExtractor().extract(view)[0]
    assert thread.payload["category"] == "promotional"
    assert thread.payload["is_automated"] is True


def test_human_email_still_yields_its_commitment():
    drafts = CommitmentExtractor().extract(_email())
    assert len(drafts) == 1
    assert "pricing deck" in drafts[0].payload["text"]
    assert drafts[0].payload["speaker_email"] == "alice@acme.com"


def test_quoted_reply_history_is_not_re_extracted():
    """A reply must not re-claim the original sender's promise as its own."""
    body = (
        "Sounds good, thanks.\n"
        "\n"
        "On Mon, 4 Aug 2026, Bob <bob@acme.com> wrote:\n"
        "> I'll send the signed contract tomorrow.\n"
    )
    assert "signed contract" not in strip_quoted_reply(body)
    drafts = CommitmentExtractor().extract(_email(body_text=body))
    assert drafts == []


def test_sealed_body_is_never_extracted():
    """If the decryption seam was bypassed, ciphertext must not reach memory."""
    view = gmail_view(payload={
        "from": "Alice <alice@acme.com>", "to": "f@n.com", "subject": "s",
        "thread_id": "t-3", "body_text": "fenc:1:gAAAAABqchCaucAiqUJ9jIx2V7A9cQ",
        "label_ids": ["CATEGORY_PERSONAL"],
    })
    assert CommitmentExtractor().extract(view) == []
    assert ThreadExtractor().extract(view)[0].payload["snippet"] is None


def test_snippet_drops_urls_and_invisible_padding():
    raw = "Deals ‌​ inside﻿ see https://track.example.com/a?b=1 now"
    assert snippet_text(raw, 200) == "Deals inside see now"


# ── Embedding source text ──────────────────────────────────────────────────


def test_thread_text_uses_subject_and_participants_not_the_id():
    text = memory_text(Rec("thread", {
        "thread_id": "19f98866cbc10958",
        "subject": "Q3 pricing",
        "participants": ["alice@acme.com", "founder@navis.com"],
        "snippet": "Sending the deck Friday",
        "category": "personal",
    }, dedup_key="19f98866cbc10958"))
    assert "19f98866cbc10958" not in text          # the whole point
    assert "Q3 pricing" in text
    assert "alice@acme.com" in text
    assert "Sending the deck Friday" in text


def test_thread_text_survives_records_without_the_additive_fields():
    """Existing rows predate participants/snippet — subject alone must still work."""
    text = memory_text(Rec("thread", {"thread_id": "abc123", "subject": "Board update"},
                           dedup_key="abc123"))
    assert "Board update" in text
    assert "abc123" not in text


def test_person_text_carries_name_email_role_and_domain():
    text = memory_text(Rec("person", {"name": "Alice Smith", "email": "alice@acme.com",
                                      "role": "from"}, dedup_key="alice@acme.com"))
    assert "Alice Smith" in text and "alice@acme.com" in text
    assert "sender" in text and "acme.com" in text


def test_relationship_text_describes_the_relationship():
    inbound = memory_text(Rec("relationship", {"person_email": "alice@acme.com",
                                               "direction": "inbound", "via": "email"}))
    outbound = memory_text(Rec("relationship", {"person_email": "bob@acme.com",
                                                "direction": "outbound", "via": "email"}))
    assert inbound.startswith("Received email from alice@acme.com")
    assert outbound.startswith("Sent email to bob@acme.com")
    assert "inbound:" not in inbound  # no raw dedup-key form


def test_company_text_includes_a_readable_name():
    text = memory_text(Rec("company", {"domain": "slice.bank.in", "name": None},
                           dedup_key="slice.bank.in"))
    assert "slice" in text and "slice.bank.in" in text


def test_commitment_text_carries_speaker_and_subject():
    text = memory_text(Rec("commitment", {
        "text": "I'll send the deck by Friday.", "speaker_email": "alice@acme.com",
        "subject": "Q3 pricing"}))
    assert "alice@acme.com" in text and "Q3 pricing" in text and "deck" in text


def test_meeting_and_document_text():
    meeting = memory_text(Rec("meeting", {
        "title": "Investor sync", "starts_at": "2026-08-06T10:00:00+05:30",
        "attendees": ["a@x.com", "b@y.com"]}))
    assert "Investor sync" in meeting and "a@x.com" in meeting and "2026-08-06" in meeting
    doc = memory_text(Rec("document", {"title": "Roadmap 2026", "owners": ["a@x.com"]}))
    assert "Roadmap 2026" in doc and "a@x.com" in doc


def test_long_attendee_lists_are_bounded():
    text = memory_text(Rec("meeting", {
        "title": "All hands", "attendees": [f"p{i}@x.com" for i in range(20)]}))
    assert "and 15 others" in text
    assert len(text) < 400


def test_no_semantic_content_falls_back_visibly():
    text = memory_text(Rec("thread", {"thread_id": "zzz"}, dedup_key="zzz"))
    assert text == "thread zzz"


def test_memory_text_is_deterministic():
    payload = {"subject": "Q3 pricing", "participants": ["a@x.com"], "snippet": "hi"}
    assert memory_text(Rec("thread", payload)) == memory_text(Rec("thread", payload))


@pytest.mark.parametrize("fragment", ["*Decision:*", "Decision:", "we'll", "I'll."])
def test_contentless_fragments_are_rejected(fragment):
    """A marker with no statement behind it is not a fact worth remembering."""
    assert not is_decision_sentence(fragment)
    assert not is_commitment_sentence(fragment)


# ── Message direction (the basis of "who is waiting on you?") ──────────
#
# `direction` is what lets Reasoning answer "did the user reply?" as a FACT
# rather than a judgement. Gmail applies the SENT label itself, the connector
# captures sent mail alongside received, and State's latest-wins fold turns
# per-message direction into "who spoke last in this thread" for free.


def test_thread_records_inbound_direction_and_sender():
    view = gmail_view(payload={
        "from": "John Carter <john@partner.com>",
        "to": "founder@navis.com",
        "subject": "Contract review",
        "thread_id": "t-77",
        "sender_email": "john@partner.com",
        "sender_name": "John Carter",
        "label_ids": ["INBOX", "CATEGORY_PERSONAL"],
    })
    thread = ThreadExtractor().extract(view)[0]
    assert thread.payload["direction"] == "inbound"
    assert thread.payload["from_email"] == "john@partner.com"
    assert thread.payload["from_name"] == "John Carter"


def test_thread_records_outbound_direction_from_the_SENT_label():
    """Gmail's own label — not a header comparison, which would need to know
    every alias the founder sends from."""
    view = gmail_view(payload={
        "from": "Founder <founder@navis.com>",
        "to": "john@partner.com",
        "subject": "Re: Contract review",
        "thread_id": "t-77",
        "sender_email": "founder@navis.com",
        "label_ids": ["SENT"],
    })
    thread = ThreadExtractor().extract(view)[0]
    assert thread.payload["direction"] == "outbound"


def test_direction_defaults_to_inbound_when_labels_are_missing():
    """A message with no labels is treated as inbound. That is the SAFE default:
    the worst case is nudging about a thread the founder already handled, versus
    silently dropping a customer who is genuinely waiting."""
    view = gmail_view(payload={
        "from": "John <john@partner.com>",
        "to": "founder@navis.com",
        "subject": "Hi",
        "thread_id": "t-78",
        "sender_email": "john@partner.com",
    })
    assert ThreadExtractor().extract(view)[0].payload["direction"] == "inbound"


def test_sender_email_is_normalised_to_lowercase():
    """Reasoning compares senders against no-reply markers and dedupes people by
    address; mixed case would defeat both."""
    view = gmail_view(payload={
        "from": "John <John@Partner.COM>",
        "to": "founder@navis.com",
        "subject": "Hi",
        "thread_id": "t-79",
        "sender_email": "John@Partner.COM",
    })
    assert ThreadExtractor().extract(view)[0].payload["from_email"] == "john@partner.com"


# ── Commitment deadlines + binding language (Phase 2A) ─────────────────


def test_a_dated_promise_records_the_deadline_it_stated():
    """The deadline is resolved against the MESSAGE date, not against now — a
    promise "by Friday" made three weeks ago means THAT Friday."""
    from datetime import datetime, timezone

    view = gmail_view(
        payload={
            "from": "Chrissy <chrissy@partner.com>",
            "to": "founder@navis.com",
            "subject": "Product deck",
            "thread_id": "t-90",
            "sender_email": "chrissy@partner.com",
            "sender_name": "Chrissy Lane",
            "body_text": "I'll send you the product deck by Friday.",
        },
        event_timestamp=datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc),  # Monday
    )
    (draft,) = CommitmentExtractor().extract(view)
    assert draft.payload["due_at"].startswith("2026-08-07")  # that week's Friday
    assert draft.payload["due_text"] == "Friday"
    assert draft.payload["speaker_email"] == "chrissy@partner.com"
    assert draft.payload["thread_id"] == "t-90"


def test_an_undated_promise_records_NO_deadline():
    """Null means "no date given". Nothing downstream may substitute a guess:
    a fabricated deadline makes Navis assert something false about a person."""
    view = _email(body_text="I'll send you the product deck.")
    (draft,) = CommitmentExtractor().extract(view)
    assert draft.payload["due_at"] is None
    assert draft.payload["due_text"] is None


@pytest.mark.parametrize(
    "body",
    [
        "Maybe I'll send the deck on Friday if I get time.",
        "We should send the deck to the investors.",
        "Can you send me the deck by Friday?",
        "I'll try to send it by Friday.",
    ],
)
def test_non_binding_language_records_no_commitment(body):
    assert CommitmentExtractor().extract(_email(body_text=body)) == []


# ── Sticky thread markers ──────────────────────────────────────────────


def test_a_delivery_email_stamps_the_thread_with_fulfilment():
    view = gmail_view(payload={
        "from": "Chrissy <chrissy@partner.com>",
        "to": "founder@navis.com", "subject": "Product deck", "thread_id": "t-91",
        "sender_email": "chrissy@partner.com",
        "body_text": "Attached is the product deck, sorry for the delay.",
    })
    payload = ThreadExtractor().extract(view)[0].payload
    assert payload["fulfilment_at"]
    assert payload["fulfilment_from"] == "chrissy@partner.com"


def test_a_decision_email_stamps_the_thread_with_resolution():
    view = gmail_view(payload={
        "from": "Co <cofounder@navis.com>",
        "to": "founder@navis.com", "subject": "Release", "thread_id": "t-92",
        "sender_email": "cofounder@navis.com",
        "body_text": "We have decided to proceed with the release on Monday.",
    })
    payload = ThreadExtractor().extract(view)[0].payload
    assert payload["resolution_at"]
    assert payload["resolution_from"] == "cofounder@navis.com"


def test_an_ordinary_message_OMITS_the_markers_rather_than_nulling_them():
    """Omission is what makes the markers sticky: State's fold skips nulls, so
    writing None would be harmless, but writing nothing is the contract."""
    payload = ThreadExtractor().extract(_email(body_text="Sounds good, thanks."))[0].payload
    assert "fulfilment_at" not in payload
    assert "resolution_at" not in payload


# ── Decision requests ──────────────────────────────────────────────────


def test_an_inbound_ask_becomes_a_decision_REQUEST():
    from navis.memory.extractors.email import DecisionRequestExtractor

    view = gmail_view(payload={
        "from": "Mia <mia@team.com>",
        "to": "founder@navis.com", "subject": "Hiring budget", "thread_id": "t-93",
        "sender_email": "mia@team.com", "sender_name": "Mia Chen",
        "body_text": "Can you approve the new hiring budget for Q4?",
        "label_ids": ["INBOX", "CATEGORY_PERSONAL"],
    })
    (draft,) = DecisionRequestExtractor().extract(view)
    assert draft.payload["kind"] == "request"
    assert draft.payload["asked_by"] == "mia@team.com"
    assert draft.payload["thread_id"] == "t-93"


def test_a_recorded_decision_is_marked_as_a_RECORD_not_a_request():
    from navis.memory.extractors.email import DecisionRequestExtractor

    view = _email(body_text="We have decided to raise a Series A this quarter.")
    assert DecisionRequestExtractor().extract(view) == []
    (draft,) = DecisionExtractor().extract(view)
    assert draft.payload["kind"] == "record"


def test_an_ask_the_ORG_SENT_is_not_a_decision_it_owes():
    """Outbound: the org is WAITING on that answer, not owing it."""
    from navis.memory.extractors.email import DecisionRequestExtractor

    view = gmail_view(payload={
        "from": "Founder <founder@navis.com>",
        "to": "mia@team.com", "subject": "Hiring budget", "thread_id": "t-94",
        "sender_email": "founder@navis.com",
        "body_text": "Can you approve the new hiring budget?",
        "label_ids": ["SENT"],
    })
    assert DecisionRequestExtractor().extract(view) == []


@pytest.mark.parametrize(
    "body",
    [
        "Here's the decision we made about the launch.",
        "FYI, the launch is Monday next week.",
        "Let's discuss this sometime next week.",
    ],
)
def test_informational_email_records_no_decision_request(body):
    from navis.memory.extractors.email import DecisionRequestExtractor

    assert DecisionRequestExtractor().extract(_email(body_text=body)) == []


# ── P0-1: the sender classification reaches PERSON and RELATIONSHIP ────
#
# The classifier was already correct and already ran on every message, but its
# output was attached ONLY to the THREAD fact. RELATIONSHIP carried `direction`
# and nothing else, so a newsletter blast and a human counterparty entered State
# indistinguishable — which is why an engine reading threads could filter noise
# and one reading relationships structurally could not. These pin the fix:
# the classification is PRESERVED onto the address-scoped entities, scoped to
# the SENDER, and nothing is suppressed.

import uuid  # noqa: E402

from navis.memory.envelope import EntityType  # noqa: E402
from navis.memory.extractors.email import (  # noqa: E402
    CompanyExtractor,
    PersonExtractor,
    RelationshipExtractor,
)
from navis.memory.service import build_memory_records  # noqa: E402
from navis.state.v2.builders import MemoryRecordView, build_projection  # noqa: E402
from tests.memory.fakes import EVENT, INTEG, ORG  # noqa: E402

ORG_B = uuid.UUID("99999999-9999-9999-9999-999999999999")

_NEWSLETTER = {
    "from": "Open Scout <openscout+on-the-side@substack.com>",
    "to": "founder@navis.com",
    "subject": "This week in AI",
    "thread_id": "t-nl",
    "sender_email": "openscout+on-the-side@substack.com",
    "label_ids": ["CATEGORY_PROMOTIONS"],
    "body_text": "Five links you should read this week.",
}
_HUMAN = {
    "from": "Alice <alice@acme.com>",
    "to": "founder@navis.com",
    "subject": "Q3 pricing",
    "thread_id": "t-h",
    "sender_email": "alice@acme.com",
    "label_ids": ["INBOX", "CATEGORY_PERSONAL"],
    "body_text": "Can we talk Thursday about pricing?",
}


def _draft(drafts, dedup_key):
    return next(d for d in drafts if d.dedup_key == dedup_key)


# 1. Human sender


def test_a_human_sender_is_recorded_as_not_automated():
    view = gmail_view(payload=_HUMAN)

    person = _draft(PersonExtractor().extract(view), "alice@acme.com")
    assert person.payload["is_automated"] is False
    assert person.payload["category"] == "personal"

    rel = _draft(RelationshipExtractor().extract(view), "inbound:alice@acme.com")
    assert rel.payload["is_automated"] is False
    assert rel.payload["category"] == "personal"


# 2. Automated / newsletter sender


def test_a_newsletter_sender_is_flagged_but_still_fully_remembered():
    """Preserve classification, do NOT delete evidence: the address remains a
    fact, it is simply legible as a service blast rather than a human party."""
    view = gmail_view(payload=_NEWSLETTER)
    people = PersonExtractor().extract(view)
    rels = RelationshipExtractor().extract(view)

    # Nothing suppressed — every address on the message still becomes a fact.
    assert {d.dedup_key for d in people} == {
        "openscout+on-the-side@substack.com", "founder@navis.com"
    }
    assert len(rels) == 2

    sender = _draft(people, "openscout+on-the-side@substack.com")
    assert sender.payload["is_automated"] is True
    assert sender.payload["category"] == "promotional"
    assert sender.payload["email"] == "openscout+on-the-side@substack.com"  # still there

    rel = _draft(rels, "inbound:openscout+on-the-side@substack.com")
    assert rel.payload["is_automated"] is True
    assert rel.payload["category"] == "promotional"


def test_a_recipient_never_inherits_the_senders_classification():
    """The founder cc'd on a promotional blast is not automated; the blast is.
    Every input the classifier reads describes how the FROM address mails, so a
    recipient's own entity must not pick the flag up under latest-wins folding."""
    view = gmail_view(payload=_NEWSLETTER)

    founder = _draft(PersonExtractor().extract(view), "founder@navis.com")
    assert "is_automated" not in founder.payload
    assert "category" not in founder.payload

    rel = _draft(RelationshipExtractor().extract(view), "outbound:founder@navis.com")
    assert "is_automated" not in rel.payload
    assert "category" not in rel.payload


@pytest.mark.parametrize(
    "extra",
    [
        {"label_ids": ["CATEGORY_PROMOTIONS"]},          # Gmail's own tab
        {"label_ids": ["CATEGORY_UPDATES"]},             # receipts / OTPs
        {"list_id": "<openscout.substack.com>"},         # RFC List-Id
        {"precedence": "bulk"},                          # RFC precedence
        {"auto_submitted": "auto-generated"},            # RFC 3834
    ],
)
def test_every_automation_marker_the_classifier_knows_reaches_the_relationship(extra):
    """One classifier, not two: whatever makes a THREAD automated must make the
    sender's RELATIONSHIP automated by exactly the same rule."""
    payload = {**_HUMAN, "label_ids": [], **extra}
    view = gmail_view(payload=payload)

    thread = ThreadExtractor().extract(view)[0]
    rel = _draft(RelationshipExtractor().extract(view), "inbound:alice@acme.com")

    assert thread.payload["is_automated"] is True
    assert rel.payload["is_automated"] is thread.payload["is_automated"]
    assert rel.payload["category"] == thread.payload["category"]


def test_company_deliberately_carries_no_classification():
    """A COMPANY is a DOMAIN, and one domain hosts both alice@acme.com and
    no-reply@acme.com. A latest-wins flag there would track whichever mailed
    last — so the field is omitted rather than added to satisfy symmetry."""
    companies = CompanyExtractor().extract(gmail_view(payload=_NEWSLETTER))
    assert companies
    for c in companies:
        assert "is_automated" not in c.payload
        assert "category" not in c.payload


# 3. Existing records without the fields


def _rel_record(payload: dict, *, tag: str, org_id=ORG) -> MemoryRecordView:
    from datetime import datetime, timezone

    ns = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    return MemoryRecordView(
        memory_id=uuid.uuid5(ns, f"mem:{org_id}:{tag}"),
        org_id=org_id,
        entity_type=EntityType.RELATIONSHIP.value,
        entity_id=uuid.uuid5(ns, f"ent:{org_id}:relationship:alice@acme.com"),
        event_timestamp=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
        payload=payload,
        source_signal_event_id=uuid.uuid5(ns, f"sig:{org_id}:{tag}"),
        schema_version=1,
        confidence=0.9,
    )


def test_records_written_before_the_fields_existed_still_fold():
    """Additive keys, no schema bump, no migration: a v1 payload loads and folds
    exactly as before and reads back as UNKNOWN — never as "this is a human"."""
    state = build_projection(
        ORG,
        [_rel_record({"person_email": "alice@acme.com", "direction": "inbound"}, tag="v1")],
    )
    (r,) = state.relationships()
    assert r.person_email == "alice@acme.com"
    assert r.directions == ["inbound"]
    assert r.is_automated is None      # unknown, NOT False
    assert r.category is None


def test_an_old_record_cannot_clobber_the_new_classification():
    """The v1/v2 coexistence argument, executed. Both facts of one message share
    an event_timestamp, so the fold falls back to memory_id order — but v1 simply
    LACKS the keys, and latest-NON-NULL-wins cannot be clobbered by an absent
    key. Deterministic whichever way the records arrive."""
    v1 = _rel_record({"person_email": "alice@acme.com", "direction": "inbound"}, tag="v1")
    v2 = _rel_record(
        {"person_email": "alice@acme.com", "direction": "inbound",
         "is_automated": True, "category": "promotional"},
        tag="v2",
    )
    for order in ([v1, v2], [v2, v1]):
        (r,) = build_projection(ORG, order).relationships()
        assert r.is_automated is True
        assert r.category == "promotional"


# 4. Provenance


def test_classified_facts_keep_their_full_signal_lineage():
    records = build_memory_records(gmail_view(payload=_NEWSLETTER))
    classified = [r for r in records if (r.payload or {}).get("is_automated") is not None]

    # person + relationship for the sender, and the thread fact.
    assert {r.entity_type for r in classified} == {
        EntityType.PERSON, EntityType.RELATIONSHIP, EntityType.THREAD
    }
    for r in classified:
        assert r.org_id == ORG
        assert r.source_signal_event_id == EVENT
        assert r.source_integration_id == INTEG
        assert r.extractor_name and r.extractor_version


# 5. Organization isolation


def test_classification_is_scoped_to_one_organization():
    """The same message ingested for two orgs yields disjoint memory_ids AND
    disjoint entity_ids, so one org's classification can never land on another
    org's entity."""
    a = build_memory_records(gmail_view(payload=_NEWSLETTER))
    b = build_memory_records(gmail_view(payload=_NEWSLETTER, org_id=ORG_B))

    def classified_entities(records):
        return {r.entity_id for r in records if (r.payload or {}).get("is_automated") is not None}

    assert classified_entities(a) and classified_entities(b)
    assert classified_entities(a).isdisjoint(classified_entities(b))
    assert {r.memory_id for r in a}.isdisjoint({r.memory_id for r in b})
    assert all(r.org_id == ORG for r in a)
    assert all(r.org_id == ORG_B for r in b)


# 6. THREAD behaviour is untouched


def test_thread_classification_and_identity_are_unchanged():
    """THREAD was already correct. P0-1 must not move its extractor name, its
    version, or a single payload key."""
    assert ThreadExtractor.name == "email_thread_v2"
    assert ThreadExtractor.version == "1.2.0"

    payload = ThreadExtractor().extract(gmail_view(payload=_NEWSLETTER))[0].payload
    assert payload["category"] == "promotional"
    assert payload["is_automated"] is True
    assert payload["thread_id"] == "t-nl"
    assert payload["direction"] == "inbound"


# 7. WaitingReplyEngine behaviour is untouched


def test_waiting_on_reply_is_unaffected_by_relationship_classification():
    """The engine reads THREADs. Adding fields to RELATIONSHIP must not couple
    it: a human thread still fires even when the same address also carries an
    automated relationship fact, and an automated thread still does not."""
    from datetime import timedelta

    from navis.reasoning.v2 import run_reasoning_v2
    from navis.reasoning.v2.contract import SignalType
    from tests.reasoning.test_waiting_on_reply import NOW, ORG as W_ORG, _msg

    automated_rel = MemoryRecordView(
        memory_id=uuid.uuid5(uuid.NAMESPACE_DNS, "rel-auto"),
        org_id=W_ORG,
        entity_type=EntityType.RELATIONSHIP.value,
        entity_id=uuid.uuid5(uuid.NAMESPACE_DNS, "ent-rel-auto"),
        event_timestamp=NOW - timedelta(days=4),
        payload={"person_email": "john@partner.com", "direction": "inbound",
                 "is_automated": True, "category": "promotional"},
        source_signal_event_id=uuid.uuid5(uuid.NAMESPACE_DNS, "sig-rel-auto"),
        schema_version=1,
        confidence=0.9,
    )
    state = build_projection(
        W_ORG,
        [_msg("t-1", days_ago=4, direction="inbound"), automated_rel],
        as_of=NOW,
    )
    waiting = [
        s for s in run_reasoning_v2(state)
        if s.signal_type is SignalType.WAITING_ON_USER_REPLY
    ]
    assert len(waiting) == 1
    assert "John Carter" in waiting[0].title
