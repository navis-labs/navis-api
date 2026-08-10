"""Due-date extraction from commitment language.

Two properties matter more than coverage breadth:

  1. A sentence with NO deadline returns None. Inventing one makes Navis assert
     something false about a third party ("Chrissy missed Friday"), which is the
     most damaging error this feature can make.
  2. Relative dates resolve against the MESSAGE, not against now — otherwise
     every historical deadline silently slides into the future and every genuine
     follow-up is suppressed.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from navis.signals.due_dates import extract_due_at

# Monday 2026-08-03, 10:00 UTC.
MON = datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)


def _due(sentence: str, *, sent_at: datetime = MON) -> str | None:
    r = extract_due_at(sentence, sent_at=sent_at)
    return r[0].date().isoformat() if r else None


@pytest.mark.parametrize(
    "sentence,expected",
    [
        # Relative to the message's own week.
        ("I'll send the deck by Friday.", "2026-08-07"),
        ("I'll send it Friday.", "2026-08-07"),
        ("I'll get back to you tomorrow.", "2026-08-04"),
        ("I'll have it done by end of day.", "2026-08-03"),
        ("I'll finish it today.", "2026-08-03"),
        ("We'll deliver the report next week.", "2026-08-14"),
        ("I'll send it this week.", "2026-08-07"),
        ("I'll send it by end of week.", "2026-08-07"),
        ("I'll send it next Monday.", "2026-08-10"),
        ("I'll send it in 3 days.", "2026-08-06"),
        ("I'll send it in 2 weeks.", "2026-08-17"),
        # Absolute.
        ("I'll send it by 2026-08-20.", "2026-08-20"),
        ("I'll send it by 15 August.", "2026-08-15"),
        ("I'll send it by August 15th.", "2026-08-15"),
        ("I'll send it on the 20th of August.", "2026-08-20"),
    ],
)
def test_recognised_deadlines(sentence, expected):
    assert _due(sentence) == expected


@pytest.mark.parametrize(
    "sentence",
    [
        "I'll send you the product deck.",
        "I'll get back to you shortly.",
        "We'll take a look and revert.",
        "I'll send the notes from Monday's call.",   # backward-looking
        "I'll follow up after Tuesday.",             # backward-looking
        "I'll send it by 31 February.",              # impossible date
    ],
)
def test_no_deadline_means_no_deadline(sentence):
    """None is the honest answer, and by far the most common one."""
    assert _due(sentence) is None


def test_a_deadline_is_the_END_of_the_named_day():
    """"by Friday" is met any time that Friday. Treating it as 00:00 would mark
    a promise overdue while the person still had the whole day to keep it."""
    due, _ = extract_due_at("I'll send it by Friday.", sent_at=MON)
    assert (due.hour, due.minute) == (23, 59)


def test_relative_dates_anchor_on_the_MESSAGE_not_on_now():
    """The load-bearing property. The same sentence sent in different weeks
    means different Fridays."""
    week_later = datetime(2026, 8, 10, 10, 0, tzinfo=timezone.utc)
    assert _due("I'll send it by Friday.") == "2026-08-07"
    assert _due("I'll send it by Friday.", sent_at=week_later) == "2026-08-14"


def test_a_weekday_naming_the_send_day_rolls_forward_a_week():
    """"I'll send it Monday", sent on a Monday, does not mean today."""
    assert _due("I'll send it Monday.") == "2026-08-10"


def test_next_weekday_always_skips_a_full_week():
    friday = datetime(2026, 8, 7, 10, 0, tzinfo=timezone.utc)
    assert _due("I'll send it next Friday.", sent_at=friday) == "2026-08-14"


def test_a_bare_month_day_already_past_rolls_to_next_year():
    assert _due("I'll send it by 1 March.") == "2027-03-01"


def test_the_matched_phrase_is_returned_for_display():
    """The card quotes the promiser's own words ("due Friday"), not a
    reformatted date Navis chose."""
    _, phrase = extract_due_at("I'll send the deck by Friday.", sent_at=MON)
    assert phrase.lower() == "friday"


def test_extraction_is_pure():
    a = extract_due_at("I'll send it by Friday.", sent_at=MON)
    b = extract_due_at("I'll send it by Friday.", sent_at=MON)
    assert a == b


def test_empty_input_is_safe():
    assert extract_due_at("", sent_at=MON) is None
