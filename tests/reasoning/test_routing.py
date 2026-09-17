"""Chat routing — every reported "wrong answer" utterance is pinned here.

The bug class these guard against is silent: a missing rule doesn't crash, it
just sends an operational request into the memory-QA path, which answers
"Navis doesn't have enough memory to answer this yet". Nothing fails, the user
is simply told something false. So the ONLY defence is asserting the route for
the exact phrasings that failed.
"""
from __future__ import annotations

import pytest

from navis.reasoning.routing import (
    ChatRoute,
    classify,
    is_cancel_meeting_intent,
    is_reschedule_intent,
    is_send_email_intent,
    normalise_utterance,
)


# ── The reported failures ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "utterance,expected",
    [
        # All six were answered with the memory fallback in production.
        ("Reschedule the meeting to 5–6 PM", ChatRoute.RESCHEDULE_MEETING),
        ("Cancel the meeting", ChatRoute.CANCEL_MEETING),
        ("Send email to mulayamabmrc@gmail.com", ChatRoute.SEND_EMAIL),
        ("What meetings do I have this week?", ChatRoute.CALENDAR_QUERY),
        # These two stay in workspace QA by design — but the QA path now
        # degrades through Calendar/Gmail instead of claiming empty memory.
        # See tests/reasoning/test_workspace_context.py.
        ("What should I do first?", ChatRoute.WORKSPACE_QA),
        ("Who am I keeping waiting?", ChatRoute.WORKSPACE_QA),
    ],
)
def test_reported_failures_route_correctly(utterance, expected):
    assert classify(utterance) is expected


def test_en_dash_is_normalised():
    """"5–6 PM" (EN DASH) is what a phone keyboard and a pasted invite produce.
    Every range parser in the codebase matches ASCII '-' only."""
    assert normalise_utterance("Reschedule to 5–6 PM") == "Reschedule to 5-6 PM"
    assert normalise_utterance("a—b") == "a-b"
    assert normalise_utterance("  spaced   out  ") == "spaced out"


def test_reschedule_is_not_shadowed_by_the_schedule_verb():
    r"""`\bschedule\b` does NOT match inside "reschedule" — the boundary falls
    between two word characters. That single fact is why the reschedule
    utterance reached neither the scheduling agent nor a reschedule handler."""
    import re

    assert re.search(r"\bschedule\b", "Reschedule the meeting") is None
    assert classify("Reschedule the meeting") is ChatRoute.RESCHEDULE_MEETING


# ── Ordering ───────────────────────────────────────────────────────────


def test_mutation_wins_over_creation():
    """"reschedule the meeting" contains a meeting noun, so a naive rule order
    reads it as a fresh booking and asks "who with?"."""
    assert classify("reschedule the meeting with Rahul") is ChatRoute.RESCHEDULE_MEETING
    assert classify("cancel my 1:1 with Rahul") is ChatRoute.CANCEL_MEETING
    assert classify("schedule a meeting with Rahul") is ChatRoute.SCHEDULE_MEETING


def test_pending_scheduling_owns_bare_followups():
    assert (
        classify("tomorrow at 3 PM", has_pending_scheduling=True)
        is ChatRoute.SCHEDULE_MEETING
    )
    # Without a pending flow the same fragment is just a question.
    assert classify("tomorrow at 3 PM") is ChatRoute.WORKSPACE_QA


def test_pending_scheduling_does_not_hijack_a_fresh_question():
    assert (
        classify("what meetings do I have today?", has_pending_scheduling=True)
        is ChatRoute.CALENDAR_QUERY
    )


# ── False positives the rules must NOT produce ─────────────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "Was the meeting cancelled?",       # a question, not a command
        "did I cancel the sync?",
        "cancel my subscription",           # not a meeting noun
        "why was the call cancelled",
    ],
)
def test_cancel_false_positives(utterance):
    assert not is_cancel_meeting_intent(utterance)
    assert classify(utterance) is not ChatRoute.CANCEL_MEETING


@pytest.mark.parametrize(
    "utterance",
    [
        "when was the meeting moved to?",
        "why did we postpone the launch",    # no meeting noun, question form
    ],
)
def test_reschedule_false_positives(utterance):
    assert not is_reschedule_intent(utterance)


@pytest.mark.parametrize(
    "utterance",
    [
        "what emails came in today?",   # a query, not a send
        "draft the roadmap",            # no recipient
        "summarise my inbox",
        "which emails need a reply?",
    ],
)
def test_email_false_positives(utterance):
    assert not is_send_email_intent(utterance)
    assert classify(utterance) is not ChatRoute.SEND_EMAIL


@pytest.mark.parametrize(
    "utterance",
    [
        "Send email to mulayamabmrc@gmail.com",
        "email priya@navis.ai about the term sheet",
        "reply to Priya with a note about pricing",
        "send a message to arjun@x.co saying we'll ship Friday",
    ],
)
def test_email_true_positives(utterance):
    assert is_send_email_intent(utterance)
    assert classify(utterance) is ChatRoute.SEND_EMAIL


def test_empty_input_is_safe():
    assert classify("") is ChatRoute.WORKSPACE_QA
    assert classify("   ") is ChatRoute.WORKSPACE_QA


# ── EMAIL_SEARCH: retrieving mail is not sending it (TASK 7.5.4) ───────────
#
# "Find the email from Ajeet" was reaching WORKSPACE_QA and being answered with
# reasoning prose about the organization while the message sat in signal_events.


@pytest.mark.parametrize(
    "utterance",
    [
        "Find the email from Ajeet.",
        "Show me the latest email from Ajeet.",
        "What did Ajeet email me about?",
        "Find emails mentioning Navis.",
        "Search my inbox for the invoice",
        "Any emails from the investor?",
    ],
)
def test_explicit_mail_retrieval_routes_to_email_search(utterance):
    assert classify(utterance) is ChatRoute.EMAIL_SEARCH


@pytest.mark.parametrize(
    "utterance",
    [
        # Sending ALWAYS wins the tie — these name mail too.
        "Send the email to ajeet@example.com",
        "Send an email to Ajeet about the deck",
        "Draft an email to Ajeet",
        "Reply to Ajeet's email and tell him we're delayed",
    ],
)
def test_sending_still_wins_over_retrieval(utterance):
    assert classify(utterance) is ChatRoute.SEND_EMAIL


@pytest.mark.parametrize(
    "utterance",
    [
        # Organizational understanding has no mail noun and must stay on the
        # Identity -> Memory -> State -> Context -> Reasoning -> Intelligence path.
        "Why does Ajeet matter to my organization?",
        "Who is Ajeet?",
        "What is happening with Ajeet?",
        "What changed with Ajeet?",
        "What should I do first?",
        # "message" is how a founder refers to Slack too — routing it to Gmail
        # would answer from the wrong system on a Slack-only organization.
        "Show me the message from Ajeet",
    ],
)
def test_organizational_questions_never_route_to_email_search(utterance):
    assert classify(utterance) is not ChatRoute.EMAIL_SEARCH


def test_meeting_verbs_are_unaffected_by_the_new_rule():
    assert classify("Schedule a meeting with Ajeet tomorrow") is ChatRoute.SCHEDULE_MEETING
    assert classify("Cancel the meeting") is ChatRoute.CANCEL_MEETING
    assert classify("Reschedule the meeting to 5-6 PM") is ChatRoute.RESCHEDULE_MEETING
    assert classify("What meetings do I have this week?") is ChatRoute.CALENDAR_QUERY
