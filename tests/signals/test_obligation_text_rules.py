"""Linguistic rules behind the two Phase 2A detectors.

Every example the product spec listed as SHOULD / SHOULD NOT surface is pinned
here, because these predicates are the only thing standing between "Navis
noticed something real" and "Navis chased a colleague over a sentence they never
meant as a promise".
"""
from __future__ import annotations

import pytest

from navis.signals.text_rules import (
    body_records_decision,
    body_shows_fulfilment,
    is_binding_commitment,
    is_decision_request_sentence,
    is_fulfilment_sentence,
)


# ── Commitments ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sentence",
    [
        "I'll send the deck by Friday.",
        "I'll get back to you tomorrow.",
        "We'll deliver the report next week.",
        "I will send over the signed contract on Monday.",
        "We're going to ship the fix by end of day.",
    ],
)
def test_binding_promises_are_detected(sentence):
    assert is_binding_commitment(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Maybe I'll send it Friday.",                # hedged
        "I might send the deck over tomorrow.",      # hedged
        "I'll try to send it by Friday.",            # hedged
        "Hopefully I'll get it to you by Friday.",   # hedged
        "We should send the deck.",                  # suggestion
        "Let's discuss the deck this week.",         # suggestion
        "Can you send me the deck?",                 # request AT the reader
        "Could you send the report by Monday?",      # request AT the reader
        "Please send the deck by Friday.",           # request AT the reader
        "Will you send it by Friday?",               # question
        "Thanks, that all sounds good to me.",       # no promise language
        "I hope you are doing well.",                # pleasantry
    ],
)
def test_non_binding_language_is_rejected(sentence):
    assert not is_binding_commitment(sentence)


def test_a_sentence_that_both_asks_and_promises_counts():
    """"Can you confirm? I'll send it Friday" is genuinely a promise too."""
    assert is_binding_commitment("Can you confirm the address? I'll send it by Friday.")


# ── Decision requests ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sentence",
    [
        "Should we launch on Monday or Tuesday?",
        "Can you approve the new hiring budget?",
        "Please confirm whether we should proceed.",
        "We need your approval on the vendor contract.",
        "Which option do you want us to go with?",
        "Are we going ahead with the December launch?",
        "Let me know if we should proceed with the release.",
    ],
)
def test_decision_requests_are_detected(sentence):
    assert is_decision_request_sentence(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Here's the decision we made about pricing.",   # a record, not a request
        "We have decided to go with option B.",         # a record
        "FYI, the launch is Monday next week.",         # informational
        "Let's discuss this sometime next week.",       # no decision asked
        "How are you doing this week?",                 # ordinary question
        "Did you see the article I sent you?",          # ordinary question
        "No action is needed on your side.",            # explicitly no ask
    ],
)
def test_non_requests_are_rejected(sentence):
    assert not is_decision_request_sentence(sentence)


def test_a_record_wins_over_a_request_when_a_sentence_reads_as_both():
    assert not is_decision_request_sentence(
        "We have decided we should go with option A."
    )


# ── Fulfilment / resolution ────────────────────────────────────────────


@pytest.mark.parametrize(
    "sentence",
    [
        "Attached is the product deck.",
        "Please find the report enclosed.",
        "Here's the deck you asked for.",
        "I've sent the contract over.",
        "Sharing the numbers now.",
        "As promised, the summary is below.",
    ],
)
def test_delivery_claims_are_detected(sentence):
    assert is_fulfilment_sentence(sentence)


@pytest.mark.parametrize(
    "sentence",
    ["Thanks for the reminder.", "I'll send it tomorrow.", "Sorry for the delay."],
)
def test_non_deliveries_are_not_mistaken_for_delivery(sentence):
    assert not is_fulfilment_sentence(sentence)


def test_body_level_helpers_scan_every_sentence():
    body = "Sorry for the delay on this. Attached is the deck. Let me know."
    assert body_shows_fulfilment(body)
    assert not body_records_decision(body)

    body2 = "Thanks all. We have decided to proceed with option B."
    assert body_records_decision(body2)
    assert not body_shows_fulfilment(body2)
