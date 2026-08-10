"""Grounded draft generation for the three proactive candidate kinds.

The load-bearing property is NEGATIVE: what the generator must never write.
These emails go to investors, customers and colleagues under the founder's own
name, so a draft that invents "yes, Monday works" does not produce a bad email —
it produces a lie the founder sends.

The generator is deterministic templates precisely so that property is provable
rather than merely observed on the examples someone happened to try.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from navis.execution.drafts.generator import (
    COMMITMENT_FOLLOW_UP,
    DECISION_OWED,
    WAITING_REPLY,
    InsufficientContext,
    describe_commitment,
    generate,
)
from navis.execution.drafts.thread_context import ThreadContext, ThreadMessage

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)


def _msg(
    *,
    mid="m-1",
    days_ago=4.0,
    direction="inbound",
    from_email="john@partner.com",
    from_name="John Carter",
    subject="Product Release",
    snippet="Hi, can you confirm whether we're still targeting Monday?",
) -> ThreadMessage:
    return ThreadMessage(
        message_id=mid,
        rfc_message_id=f"<{mid}@mail.gmail.com>",
        sent_at=NOW - timedelta(days=days_ago),
        direction=direction,
        from_email=from_email,
        from_name=from_name,
        subject=subject,
        snippet=snippet,
    )


def _thread(*messages, thread_id="thread-1", subject="Product Release") -> ThreadContext:
    return ThreadContext(
        thread_id=thread_id, subject=subject, messages=messages or (_msg(),)
    )


# ── Phrases that would be fabrication, in ANY draft ────────────────────
#
# Kept as one list and asserted against every generated body: a future template
# change that starts answering on the founder's behalf fails here, whichever
# kind it was added to.
# AFFIRMATIVE forms only. An earlier version listed a bare "confirmed", which
# flagged the correct draft "I haven't confirmed a decision here yet" — the very
# sentence that avoids fabricating. The claim is what matters, not the word.
_FABRICATION_MARKERS = (
    "yes,", "yes we", "no,", "we're still targeting", "we are still targeting",
    "monday works", "i've confirmed", "i have confirmed", "we've confirmed",
    "as we discussed", "as agreed", "i've attached", "please find attached",
    "i have decided", "we've decided", "let's go with", "i'll send it by",
)


def _assert_no_fabrication(body: str) -> None:
    lowered = body.lower()
    for marker in _FABRICATION_MARKERS:
        assert marker not in lowered, f"draft fabricated: {marker!r} in {body!r}"


# ══════════════════════════════════════════════════════════════════════
# Waiting on reply
# ══════════════════════════════════════════════════════════════════════


def test_waiting_reply_acknowledges_without_answering():
    """THE most important assertion in this file.

    John asked "are we still targeting Monday?". Navis does not know, and must
    not say. It acknowledges and says the founder will come back — with no
    timeframe attached, and no apology offered on their behalf.
    """
    d = generate(
        candidate_kind=WAITING_REPLY,
        payload={"person_email": "john@partner.com", "waiting_days": 4},
        thread=_thread(),
        sender_name="Alex",
    )
    _assert_no_fabrication(d.body)
    assert "Hi John," in d.body
    assert "come back to you" in d.body
    assert "shortly" not in d.body.lower()
    assert "apolog" not in d.body.lower()
    assert d.body.rstrip().endswith("Alex")


def test_the_subject_is_a_reply_to_the_real_thread():
    d = generate(
        candidate_kind=WAITING_REPLY,
        payload={"person_email": "john@partner.com"},
        thread=_thread(),
        sender_name=None,
    )
    assert d.subject == "Re: Product Release"


def test_an_existing_re_prefix_is_not_doubled():
    d = generate(
        candidate_kind=WAITING_REPLY,
        payload={"person_email": "john@partner.com"},
        thread=_thread(subject="Re: Product Release"),
        sender_name=None,
    )
    assert d.subject == "Re: Product Release"


def test_the_recipient_and_reply_anchor_come_from_the_thread():
    d = generate(
        candidate_kind=WAITING_REPLY,
        payload={"person_email": "john@partner.com"},
        thread=_thread(),
        sender_name=None,
    )
    assert d.recipients == ("john@partner.com",)
    assert d.thread_id == "thread-1"
    assert d.in_reply_to_message_id == "<m-1@mail.gmail.com>"


def test_the_draft_answers_the_RIGHT_person_in_a_busy_thread():
    """Three people in the thread; the follow-up must anchor on the one the
    candidate names, not on whoever spoke last."""
    d = generate(
        candidate_kind=WAITING_REPLY,
        payload={"person_email": "john@partner.com"},
        thread=_thread(
            _msg(mid="m-1", days_ago=6, from_email="john@partner.com", from_name="John Carter"),
            _msg(mid="m-2", days_ago=2, from_email="mia@team.com", from_name="Mia Chen"),
        ),
        sender_name=None,
    )
    assert d.recipients == ("john@partner.com",)
    assert d.in_reply_to_message_id == "<m-1@mail.gmail.com>"
    assert "Hi John," in d.body


# ══════════════════════════════════════════════════════════════════════
# Commitment follow-up
# ══════════════════════════════════════════════════════════════════════


def test_commitment_follow_up_quotes_their_promise_and_their_deadline():
    d = generate(
        candidate_kind=COMMITMENT_FOLLOW_UP,
        payload={
            "person_email": "chrissy@partner.com",
            "commitment": "I'll send you the product deck by Friday.",
            "due_text": "Friday",
            "overdue_days": 2,
        },
        thread=_thread(
            _msg(from_email="chrissy@partner.com", from_name="Chrissy Lane",
                 subject="Product deck"),
            subject="Product deck",
        ),
        sender_name="Alex",
    )
    _assert_no_fabrication(d.body)
    assert "Hi Chrissy," in d.body
    assert "the product deck" in d.body
    # Attributed to HER, and unambiguous — "on the product deck by Friday"
    # parses just as readily as "follow up BY Friday".
    assert "you mentioned you'd send it by Friday" in d.body
    assert "where this stands" in d.body


def test_no_urgency_or_excuse_is_manufactured():
    d = generate(
        candidate_kind=COMMITMENT_FOLLOW_UP,
        payload={
            "person_email": "chrissy@partner.com",
            "commitment": "I'll send you the product deck by Friday.",
            "due_text": "Friday",
            "overdue_days": 14,
        },
        thread=_thread(_msg(from_email="chrissy@partner.com", from_name="Chrissy Lane")),
        sender_name=None,
    )
    lowered = d.body.lower()
    for pushy in ("urgent", "asap", "immediately", "still waiting", "chasing", "overdue"):
        assert pushy not in lowered


def test_an_unparseable_promise_is_quoted_verbatim_not_reworded():
    """The fallback that makes paraphrasing safe: if the transform does not
    produce a clean noun phrase, their own sentence is used unaltered."""
    d = generate(
        candidate_kind=COMMITMENT_FOLLOW_UP,
        payload={
            "person_email": "chrissy@partner.com",
            "commitment": "The numbers will follow shortly and I will circle back",
        },
        thread=_thread(_msg(from_email="chrissy@partner.com", from_name="Chrissy")),
        sender_name=None,
    )
    assert "you mentioned:" in d.body
    assert "The numbers will follow shortly and I will circle back" in d.body


def test_a_deadline_is_not_repeated_in_the_verbatim_form():
    """Their sentence already contains the deadline; appending it again reads as
    "…'I'll send it by Friday' by Friday."."""
    d = generate(
        candidate_kind=COMMITMENT_FOLLOW_UP,
        payload={
            "person_email": "c@x.com",
            "commitment": "The report goes out by Friday, promise",
            "due_text": "Friday",
        },
        thread=_thread(_msg(from_email="c@x.com", from_name="Chris")),
        sender_name=None,
    )
    assert d.body.lower().count("friday") == 1


def test_a_commitment_with_no_text_refuses_rather_than_inventing_one():
    with pytest.raises(InsufficientContext, match="inventing"):
        generate(
            candidate_kind=COMMITMENT_FOLLOW_UP,
            payload={"person_email": "c@x.com", "commitment": ""},
            thread=_thread(_msg(from_email="c@x.com")),
            sender_name=None,
        )


@pytest.mark.parametrize(
    "sentence,expected,paraphrased",
    [
        ("I'll send you the product deck by Friday.", "the product deck", True),
        ("We'll deliver the Q3 numbers by end of day.", "the Q3 numbers", True),
        ("I'll share the revised contract next week.", "the revised contract", True),
        # No promise prefix to strip => quoted verbatim rather than mangled.
        ("I'll get back to you tomorrow.", "I'll get back to you tomorrow", False),
    ],
)
def test_commitment_phrasing(sentence, expected, paraphrased):
    assert describe_commitment(sentence) == (expected, paraphrased)


# ══════════════════════════════════════════════════════════════════════
# Decision owed
# ══════════════════════════════════════════════════════════════════════


def test_decision_draft_never_makes_the_decision():
    """The sharpest edge in the feature. "Monday or Tuesday?" has two plausible
    answers and Navis knows neither; picking one manufactures the founder's
    decision and broadcasts it to their team as fact."""
    d = generate(
        candidate_kind=DECISION_OWED,
        payload={
            "person_email": "john@partner.com",
            "decision_question": "Should we launch on Monday or Tuesday?",
            "age_days": 4,
        },
        thread=_thread(),
        sender_name="Alex",
    )
    _assert_no_fabrication(d.body)
    assert "monday" not in d.body.lower()
    assert "tuesday" not in d.body.lower()
    assert "come back to you with an answer" in d.body
    # And it does NOT claim the founder has not decided — Navis knows only that
    # no decision is recorded in THIS THREAD.
    assert "haven't" not in d.body.lower()
    assert "shortly" not in d.body.lower()


def test_the_reason_states_the_question_without_answering_it():
    d = generate(
        candidate_kind=DECISION_OWED,
        payload={
            "person_email": "john@partner.com",
            "decision_question": "Should we launch on Monday or Tuesday?",
            "age_days": 4,
        },
        thread=_thread(),
        sender_name=None,
    )
    assert "Should we launch on Monday or Tuesday?" in d.reason
    assert "4 days ago" in d.reason


# ══════════════════════════════════════════════════════════════════════
# Refusals, names, evidence
# ══════════════════════════════════════════════════════════════════════


def test_drafting_refuses_without_the_source_thread():
    """Item 4: a draft written from the candidate title alone cannot know the
    subject, the recipient or which message it answers."""
    with pytest.raises(InsufficientContext, match="thread"):
        generate(
            candidate_kind=WAITING_REPLY,
            payload={"person_email": "john@partner.com"},
            thread=None,
            sender_name=None,
        )


def test_drafting_refuses_an_unknown_candidate_kind():
    with pytest.raises(InsufficientContext):
        generate(
            candidate_kind="schedule_meeting",
            payload={"person_email": "a@b.com"},
            thread=_thread(),
            sender_name=None,
        )


@pytest.mark.parametrize(
    "from_name", [None, "", "Support Team", "The Notifications", "noreply", "j.smith92"]
)
def test_an_unusable_name_degrades_to_a_bare_greeting(from_name):
    """"Hi Support Team," and "Hi J.smith92," read as machine-written. A mangled
    name is worse than none."""
    d = generate(
        candidate_kind=WAITING_REPLY,
        payload={"person_email": "x@y.com"},
        thread=_thread(_msg(from_email="x@y.com", from_name=from_name)),
        sender_name=None,
    )
    assert d.body.startswith("Hi,")


def test_an_unknown_sender_name_produces_a_bare_sign_off():
    """No invented signature — the founder's own mail client completes it."""
    d = generate(
        candidate_kind=WAITING_REPLY,
        payload={"person_email": "john@partner.com"},
        thread=_thread(),
        sender_name=None,
    )
    assert d.body.rstrip().endswith("Best,")


def test_evidence_carries_facts_and_no_reasoning():
    d = generate(
        candidate_kind=COMMITMENT_FOLLOW_UP,
        payload={
            "person_email": "chrissy@partner.com",
            "commitment": "I'll send you the product deck by Friday.",
            "due_text": "Friday",
            "due_at": "2026-08-07T23:59:59+00:00",
            "overdue_days": 2,
        },
        thread=_thread(_msg(from_email="chrissy@partner.com", from_name="Chrissy Lane")),
        sender_name=None,
    )
    e = d.evidence
    assert e["thread_id"] == "thread-1"
    assert e["source_message_id"] == "m-1"
    assert e["source_from"] == "chrissy@partner.com"
    assert e["commitment"].startswith("I'll send you the product deck")
    assert e["due_text"] == "Friday"
    assert e["overdue_days"] == 2
    # Facts only — nothing describing HOW the conclusion was reached.
    for key in e:
        assert "reason" not in key
        assert "because" not in key


def test_generation_is_deterministic():
    args = dict(
        candidate_kind=WAITING_REPLY,
        payload={"person_email": "john@partner.com", "waiting_days": 4},
        thread=_thread(),
        sender_name="Alex",
    )
    assert generate(**args) == generate(**args)


# ══════════════════════════════════════════════════════════════════════
# No unnecessary promises (Phase 2B hardening)
# ══════════════════════════════════════════════════════════════════════
#
# Three classes of sentence that read as harmless and are not. Each was in a
# shipped template and each is now forbidden:
#
#   * an APOLOGY is a concession. In a negotiation or a dispute it has real
#     cost, and whether to make one is the founder's call, not a template's.
#   * a TIMEFRAME is a promise the founder never made. Navis's own commitment
#     detector would read an inbound "I'll get back to you shortly" as binding —
#     manufacturing one for the user is the same error pointed the other way.
#   * a CLAIM ABOUT THE FOUNDER'S STATE ("I haven't decided yet") is
#     unverifiable. Navis knows only that no decision is recorded in this
#     thread; the founder may have decided in a meeting or on a call.

_APOLOGY_MARKERS = ("apolog", "sorry", "my bad", "slow reply", "delay on my")
_TIMEFRAME_MARKERS = (
    "shortly", "by end of day", "by eod", "tomorrow", "today", "this week",
    "within the hour", "in a few", "first thing",
)
_STATE_CLAIM_MARKERS = (
    "i haven't decided", "i have not decided", "i haven't confirmed",
    "i have not confirmed", "we haven't decided", "i've been meaning",
    "i've been busy", "i missed",
)


def _all_kinds_bodies():
    """One body per candidate kind, generated the way production does."""
    thread = _thread(
        _msg(from_email="p@x.com", from_name="Pat Rivers", subject="Product deck")
    )
    payloads = {
        WAITING_REPLY: {"person_email": "p@x.com", "waiting_days": 9},
        COMMITMENT_FOLLOW_UP: {
            "person_email": "p@x.com",
            "commitment": "I'll send you the product deck by Friday.",
            "due_text": "Friday",
            "overdue_days": 12,
        },
        DECISION_OWED: {
            "person_email": "p@x.com",
            "decision_question": "Should we launch Monday or Tuesday?",
            "age_days": 9,
        },
    }
    for kind, payload in payloads.items():
        for sender in ("Alex", None):
            yield kind, generate(
                candidate_kind=kind, payload=payload, thread=thread, sender_name=sender
            ).body


@pytest.mark.parametrize("marker", _APOLOGY_MARKERS)
def test_no_draft_apologises_on_the_founders_behalf(marker):
    for kind, body in _all_kinds_bodies():
        assert marker not in body.lower(), f"{kind} apologised: {body!r}"


@pytest.mark.parametrize("marker", _TIMEFRAME_MARKERS)
def test_no_draft_commits_the_founder_to_a_timeframe(marker):
    for kind, body in _all_kinds_bodies():
        assert marker not in body.lower(), f"{kind} promised a timeframe: {body!r}"


@pytest.mark.parametrize("marker", _STATE_CLAIM_MARKERS)
def test_no_draft_asserts_the_founders_internal_state(marker):
    for kind, body in _all_kinds_bodies():
        assert marker not in body.lower(), f"{kind} claimed a state: {body!r}"


def test_the_one_retained_promise_is_timeframe_free():
    """An acknowledgement that does not say "I'll come back to you" says
    nothing at all, so that promise stays — but bare, with no deadline attached
    that the founder would then be on the hook for."""
    thread = _thread()
    body = generate(
        candidate_kind=WAITING_REPLY,
        payload={"person_email": "john@partner.com"},
        thread=thread,
        sender_name=None,
    ).body
    assert "come back to you" in body.lower()
    for marker in _TIMEFRAME_MARKERS:
        assert marker not in body.lower()


def test_the_commitment_deadline_is_attributed_to_them():
    """"you mentioned you'd send it by Friday" — their words, not Navis
    asserting a deadline at them. It also removes a genuine ambiguity: "following
    up on the deck by Friday" parses as "follow up BY Friday" just as easily."""
    d = generate(
        candidate_kind=COMMITMENT_FOLLOW_UP,
        payload={
            "person_email": "c@x.com",
            "commitment": "I'll send you the product deck by Friday.",
            "due_text": "Friday",
        },
        thread=_thread(_msg(from_email="c@x.com", from_name="Chrissy")),
        sender_name=None,
    )
    assert "you mentioned you'd send it by Friday" in d.body
    assert "on the product deck by Friday" not in d.body


def test_the_follow_up_ask_suits_a_commitment_that_is_not_a_send():
    """"had a chance to share it" was wrong for "I'll call the vendor" — which
    takes the verbatim path precisely because it is not a send."""
    d = generate(
        candidate_kind=COMMITMENT_FOLLOW_UP,
        payload={"person_email": "c@x.com", "commitment": "I'll call the vendor by Friday"},
        thread=_thread(_msg(from_email="c@x.com", from_name="Chrissy")),
        sender_name=None,
    )
    assert "share it" not in d.body
    assert "where this stands" in d.body
