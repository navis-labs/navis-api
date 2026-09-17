"""EmailOpsAgent — "send email to X" becomes an approvable send, not a shrug.

"Send email to mulayamabmrc@gmail.com" used to reach the memory-QA path and be
answered with "Navis doesn't have enough memory" — a non-sequitur, since the
user issued a command that needs no memory at all.

The load-bearing constraint: the agent must NEVER invent a message body. Asking
"what should it say?" is correct; guessing at what a founder wants to tell an
investor is not.
"""
from __future__ import annotations

import uuid


from navis.reasoning.email_ops import (
    EmailClarification,
    EmailOpsAgent,
    EmailOpsContext,
    EmailProposal,
)

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")


def _ctx(utterance: str, **pending) -> EmailOpsContext:
    return EmailOpsContext(org_id=ORG, utterance=utterance, **pending)


async def _propose(utterance: str, **pending):
    # db=None: the identity-graph lookup is skipped, so these tests exercise the
    # deterministic extraction path only.
    return await EmailOpsAgent(db=None).propose(_ctx(utterance, **pending))


# ── The reported utterance ─────────────────────────────────────────────


async def test_bare_send_request_asks_for_the_body_and_keeps_the_recipient():
    result = await _propose("Send email to mulayamabmrc@gmail.com")
    assert isinstance(result, EmailClarification)
    assert result.field == "body"
    # The recipient is carried so the user's next message alone completes it.
    assert result.resolved["to"] == "mulayamabmrc@gmail.com"
    # And it is emphatically NOT the memory fallback.
    assert "memory" not in result.question.lower()


async def test_the_followup_answer_completes_the_send():
    first = await _propose("Send email to mulayamabmrc@gmail.com")
    second = await EmailOpsAgent(db=None).propose(
        EmailOpsContext(
            org_id=ORG,
            utterance="saying the deck is attached and I'll follow up Monday",
            pending_to=first.resolved["to"],
            pending_subject=first.resolved["subject"],
        )
    )
    assert isinstance(second, EmailProposal)
    assert second.to == "mulayamabmrc@gmail.com"
    assert "deck is attached" in second.body


# ── Never fabricate ────────────────────────────────────────────────────


async def test_the_agent_never_invents_a_body():
    result = await _propose("email priya@navis.ai about the term sheet")
    assert isinstance(result, EmailClarification)
    assert result.field == "body"
    # The topic IS captured (it was stated) and becomes the subject.
    assert result.resolved["subject"] == "the term sheet"


async def test_a_stated_message_is_used_verbatim():
    result = await _propose(
        "send an email to arjun@x.co saying we'll ship Friday and I'll send metrics"
    )
    assert isinstance(result, EmailProposal)
    assert result.body == "we'll ship Friday and I'll send metrics"


# ── Recipient resolution ───────────────────────────────────────────────


async def test_missing_recipient_asks():
    result = await _propose("send an email saying we'll ship Friday")
    assert isinstance(result, EmailClarification)
    assert result.field == "recipient"


async def test_an_unresolvable_name_asks_for_an_address():
    result = await _propose("send a note to Priya saying thanks")
    assert isinstance(result, EmailClarification)
    assert result.field == "recipient"
    assert "Priya" in result.question


async def test_a_literal_address_wins_over_a_pending_one():
    result = await _propose(
        "actually send it to new@x.com saying thanks",
        pending_to="old@x.com",
    )
    assert isinstance(result, EmailProposal)
    assert result.to == "new@x.com"


# ── Subject derivation ─────────────────────────────────────────────────


async def test_a_derived_subject_is_flagged_as_an_assumption():
    result = await _propose(
        "email arjun@x.co saying the metrics are ready for review"
    )
    assert isinstance(result, EmailProposal)
    assert result.subject == "The metrics are ready for review"
    assert any("subject" in a for a in result.assumptions)


async def test_a_stated_topic_becomes_the_subject_with_no_assumption():
    result = await _propose(
        "email arjun@x.co about Q3 metrics saying the numbers are final"
    )
    assert isinstance(result, EmailProposal)
    # "about ..." is matched first, so the subject is what the user said.
    assert "Q3 metrics" in result.subject
    assert result.assumptions == []


async def test_a_long_body_produces_a_bounded_subject():
    body = "we should revisit the pricing model because " + "detail " * 40
    result = await _propose(f"email arjun@x.co saying {body}")
    assert isinstance(result, EmailProposal)
    assert len(result.subject) <= 61


# ── Execution payload contract ─────────────────────────────────────────


async def test_payload_matches_the_gmail_provider_contract():
    result = await _propose("email arjun@x.co saying the deck is ready")
    payload = result.to_execution_payload()
    # GmailProvider.validate requires 'to' and 'subject'.
    assert payload["to"] == ["arjun@x.co"]
    assert payload["subject"]
    assert payload["body"] == "the deck is ready"
