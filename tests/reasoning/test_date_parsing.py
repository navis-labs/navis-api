"""Relative / implicit date parsing.

The production bug: "9 August", asked in 2026, resolved to 2023-08-09. Nothing
in the deterministic extractor matched a bare day-month, so the utterance fell
through to the LLM — which had no notion of "today" in its prompt and answered
with a year from its training distribution.

Two independent defences are pinned here, because either alone is fragile:
  1. the deterministic extractor now parses the shape and derives the year from
     the caller's clock;
  2. anything the MODEL returns is sanitised against that same clock.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from navis.reasoning.scheduling.extract import (
    _dated_system_prompt,
    _deterministic_prefill,
    _extract_date,
    _sanitise_llm_date,
    extract_constraints,
)

TODAY = date(2026, 8, 7)  # a Friday


# ── The reported bug ───────────────────────────────────────────────────


def test_bare_day_month_uses_the_current_year_not_a_model_guess():
    assert _extract_date("9 August", today=TODAY) == "2026-08-09"
    assert _extract_date("meet on 9 August", today=TODAY) == "2026-08-09"


def test_model_authored_past_year_is_corrected():
    """Defence in depth: even if the LLM answers 2023-08-09, we never act on it."""
    assert _sanitise_llm_date("2023-08-09", today=TODAY) == "2026-08-09"


def test_llm_prompt_carries_a_concrete_today():
    """A model with no clock invents one. The anchor is what stops it."""
    prompt = _dated_system_prompt(TODAY)
    assert "2026-08-07" in prompt
    assert "Friday" in prompt
    assert "never a past year" in prompt


# ── Shapes ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("9 August", "2026-08-09"),
        ("9th August", "2026-08-09"),
        ("9 Aug", "2026-08-09"),
        ("August 9", "2026-08-09"),
        ("Aug 9th", "2026-08-09"),
        ("on the 9th of August", "2026-08-09"),
        ("23 December", "2026-12-23"),
        ("9/8", "2026-08-09"),          # day-first
        ("9/8/2027", "2027-08-09"),     # explicit year honoured
        ("2026-08-09", "2026-08-09"),   # explicit ISO honoured
        # Symbolic constraints keep their symbolic form — the availability
        # stage resolves those against the same clock.
        ("tomorrow", "tomorrow"),
        ("today", "today"),
        ("next week", "next_week"),
        ("friday", "friday"),
    ],
)
def test_recognised_date_shapes(utterance, expected):
    assert _extract_date(utterance, today=TODAY) == expected


def test_a_day_already_past_rolls_to_next_year():
    """"1 March" asked in August means next March, not eight months ago."""
    assert _extract_date("1 March", today=TODAY) == "2027-03-01"


def test_impossible_dates_are_rejected_not_rolled():
    assert _extract_date("31 February", today=TODAY) is None
    assert _extract_date("45 August", today=TODAY) is None


@pytest.mark.parametrize(
    "utterance",
    [
        "a 1.5 hour sync",     # would parse as 1 May if '.' were a separator
        "at 3.30 pm",          # would parse as 3 March
        "book a 30 min call",
    ],
)
def test_durations_and_times_are_not_mistaken_for_dates(utterance):
    assert _extract_date(utterance, today=TODAY) is None


# ── Timezone anchoring ─────────────────────────────────────────────────


async def test_extraction_anchors_on_the_supplied_clock():
    """`now` is threaded from the agent (in the founder's LOCAL timezone) so
    "today" means the same thing to the parser and to the availability stage."""
    now = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    constraints = await extract_constraints("meet on 9 August at 3 PM", now=now)
    assert constraints.date_constraint == "2026-08-09"
    assert constraints.time_window == "15:00"


async def test_extraction_defaults_to_wall_clock_when_now_is_omitted():
    """Existing callers that pass no clock must keep working."""
    constraints = await extract_constraints("tomorrow at 10am")
    assert constraints.date_constraint == "tomorrow"
    assert constraints.time_window == "10:00"


def test_prefill_shape_is_stable():
    prefill = _deterministic_prefill("meet with Rahul on 9 August at 3 PM", today=TODAY)
    assert prefill["date_constraint"] == "2026-08-09"
    assert prefill["time_window"] == "15:00"
    assert prefill["attendee_hint"] == "Rahul"
