"""CalendarQueryAgent — intent detection, time-scope, and rendering.

Covers:
  * Positive intent on real user utterances
  * Negative guards (scheduling utterances, unrelated questions)
  * Time-scope parsing (today / tomorrow / this_week / next_week / next_meeting)
  * ResolvedWindow math (local-day alignment, UTC conversion)
  * Handler renders "no meetings" empty state
  * Handler renders event list with correct times
  * Handler surfaces Google 403 / no-integration as friendly prose
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


from navis.reasoning.calendar_query import (
    CalendarQueryAgent,
    CalendarQueryContext,
    is_calendar_query_intent,
    parse_time_scope,
    TimeScope,
)
from navis.reasoning.calendar_query.intent import resolve_window
from navis.signals.google.resolve import CalendarNotConnected

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
# Fixed test clock — Tuesday 2026-08-04 15:00 UTC.
NOW = datetime(2026, 8, 4, 15, 0, tzinfo=timezone.utc)


# ── Fakes ──────────────────────────────────────────────────────────────


class _FakeCalendarClient:
    """Just enough surface for CalendarQueryAgent (list_events_expanded)."""

    def __init__(self, events: list[dict[str, Any]] | Exception | None = None) -> None:
        self.events = events
        self.calls: list[dict[str, Any]] = []

    async def list_events_expanded(
        self, *, time_min: str, time_max: str, max_results: int = 50
    ) -> dict[str, Any]:
        self.calls.append(
            {"time_min": time_min, "time_max": time_max, "max_results": max_results}
        )
        if isinstance(self.events, Exception):
            raise self.events
        return {"items": self.events or []}


class _FakeTransport:
    async def aclose(self) -> None:
        pass


def _factory(client: _FakeCalendarClient | None, *, error: Exception | None = None):
    async def _f(db, org_id):
        if error is not None:
            raise error
        if client is None:
            # The real factory raises this specific subclass for "no Google
            # grant at all"; a bare RuntimeError means "connected, but the
            # grant can't do what we need". The two get different copy, so the
            # fake has to model the distinction.
            raise CalendarNotConnected("no active Google integration")
        return client, _FakeTransport()

    return _f


def _agent(client: _FakeCalendarClient | None) -> CalendarQueryAgent:
    return CalendarQueryAgent(db=None, calendar_client_factory=_factory(client))


def _ctx(utterance: str, time_zone: str = "UTC") -> CalendarQueryContext:
    return CalendarQueryContext(
        org_id=ORG, utterance=utterance, time_zone=time_zone, now=NOW
    )


def _event(
    *,
    title: str,
    start: str,
    end: str,
    attendees: list[str] | None = None,
    location: str | None = None,
    hangout: str | None = None,
) -> dict[str, Any]:
    return {
        "id": f"e-{title.replace(' ', '_')}",
        "summary": title,
        "start": {"dateTime": start, "timeZone": "UTC"},
        "end": {"dateTime": end, "timeZone": "UTC"},
        "attendees": [{"email": e} for e in (attendees or [])],
        "location": location,
        "hangoutLink": hangout,
    }


# ── Intent detection ───────────────────────────────────────────────────


def test_intent_fires_on_calendar_query_utterances():
    for u in [
        "what meetings do I have today",
        "list my meetings today",
        "show me today's meetings",
        "how many meetings today",
        "prepare me for my next meeting",
        "any meetings this week?",
        "tell me about this week any meeting i have?",
        "what's on my calendar tomorrow",
        "when is my next call",
        "is there any meeting today",
        "do i have meetings today",
        "list events for tomorrow",
    ]:
        assert is_calendar_query_intent(u), f"should detect: {u!r}"


def test_intent_rejects_non_calendar_and_scheduling_utterances():
    for u in [
        "fix a meeting with rahul tomorrow at 3pm",  # scheduling create
        "schedule a call with him",                  # scheduling create
        "what's for lunch today",                    # unrelated
        "who won yesterday",                         # unrelated
        "call the api",                              # dev-ops, not calendar
        "meeting notes",                             # meeting NOUN only, no query verb
    ]:
        assert not is_calendar_query_intent(u), f"should NOT detect: {u!r}"


# ── Time-scope parsing ─────────────────────────────────────────────────


def test_time_scope_parsing():
    assert parse_time_scope("meetings today") is TimeScope.TODAY
    assert parse_time_scope("meetings tomorrow") is TimeScope.TOMORROW
    assert parse_time_scope("meetings tommrow") is TimeScope.TOMORROW  # typo
    assert parse_time_scope("meetings this week") is TimeScope.THIS_WEEK
    assert parse_time_scope("meetings next week") is TimeScope.NEXT_WEEK
    assert parse_time_scope("my next meeting") is TimeScope.NEXT_MEETING
    assert parse_time_scope("upcoming meeting") is TimeScope.NEXT_MEETING
    # No scope keyword → default ANY (agent treats as "today")
    assert parse_time_scope("list my meetings") is TimeScope.ANY


def test_resolve_window_today_is_local_day_aligned():
    # User in Asia/Kolkata (UTC+05:30). NOW = 2026-08-04 15:00 UTC =
    # 2026-08-04 20:30 IST. "Today" should span the LOCAL day, not UTC.
    win = resolve_window(TimeScope.TODAY, now=NOW, time_zone="Asia/Kolkata")
    # start_utc should be 2026-08-03 18:30 UTC (= 2026-08-04 00:00 IST)
    assert win.time_min_utc.isoformat() == "2026-08-03T18:30:00+00:00"
    # end_utc should be 2026-08-04 18:29:59 UTC (= 2026-08-04 23:59:59 IST)
    assert win.time_max_utc.date().isoformat() == "2026-08-04"


def test_resolve_window_tomorrow_shifts_local_date():
    win = resolve_window(TimeScope.TOMORROW, now=NOW, time_zone="UTC")
    assert win.time_min_utc.isoformat() == "2026-08-05T00:00:00+00:00"


def test_resolve_window_next_meeting_starts_from_now():
    win = resolve_window(TimeScope.NEXT_MEETING, now=NOW, time_zone="UTC")
    assert win.time_min_utc == NOW
    assert win.max_events == 1


# ── Handler behavior ───────────────────────────────────────────────────


async def test_no_integration_returns_friendly_prose():
    agent = _agent(None)
    result = await agent.answer(_ctx("what meetings do I have today"))
    assert result.events == []
    assert "connect" in result.text.lower()


async def test_empty_calendar_today():
    agent = _agent(_FakeCalendarClient([]))
    result = await agent.answer(_ctx("what meetings do I have today"))
    assert result.events == []
    assert "no meetings" in result.text.lower()
    assert "today" in result.text.lower()


async def test_empty_calendar_tomorrow():
    agent = _agent(_FakeCalendarClient([]))
    result = await agent.answer(_ctx("meetings tomorrow"))
    assert "no meetings" in result.text.lower()
    assert "tomorrow" in result.text.lower()


async def test_empty_next_meeting():
    agent = _agent(_FakeCalendarClient([]))
    result = await agent.answer(_ctx("when is my next meeting"))
    assert "no upcoming meetings" in result.text.lower()


async def test_single_meeting_today_rendered():
    events = [_event(
        title="Investor sync",
        start="2026-08-04T16:00:00+00:00",
        end="2026-08-04T16:30:00+00:00",
        attendees=["alice@x.com", "bob@x.com"],
    )]
    agent = _agent(_FakeCalendarClient(events))
    result = await agent.answer(_ctx("what meetings do I have today"))
    assert len(result.events) == 1
    assert "Investor sync" in result.text
    assert "you have 1 meeting" in result.text.lower()


async def test_next_meeting_returns_first_only():
    events = [
        _event(
            title="Investor sync",
            start="2026-08-04T16:00:00+00:00",
            end="2026-08-04T16:30:00+00:00",
            attendees=["alice@x.com"],
        ),
    ]
    agent = _agent(_FakeCalendarClient(events))
    result = await agent.answer(_ctx("prepare me for my next meeting"))
    assert "Investor sync" in result.text
    assert "your next meeting" in result.text.lower()
    assert "alice@x.com" in result.text


async def test_multiple_meetings_grouped_by_day_this_week():
    events = [
        _event(title="Mon standup", start="2026-08-04T10:00:00+00:00", end="2026-08-04T10:30:00+00:00"),
        _event(title="Wed review", start="2026-08-06T14:00:00+00:00", end="2026-08-06T15:00:00+00:00"),
    ]
    agent = _agent(_FakeCalendarClient(events))
    result = await agent.answer(_ctx("meetings this week"))
    assert "Mon standup" in result.text
    assert "Wed review" in result.text
    assert "2 meetings" in result.text.lower()


async def test_google_403_surfaces_reconnect_message():
    class _Boom(Exception):
        pass

    err = _Boom(
        "google api 403: insufficientPermissions ACCESS_TOKEN_SCOPE_INSUFFICIENT"
    )
    agent = _agent(_FakeCalendarClient(err))
    result = await agent.answer(_ctx("what meetings do I have today"))
    assert result.events == []
    assert "reconnect" in result.text.lower()


async def test_google_generic_error_surfaces_retry_message():
    err = RuntimeError("connection refused")
    agent = _agent(_FakeCalendarClient(err))
    result = await agent.answer(_ctx("what meetings do I have today"))
    assert result.events == []
    # No reconnect mention — this is a transient error.
    assert "try again" in result.text.lower()


async def test_time_zone_renders_in_users_local_time():
    events = [_event(
        title="Coffee chat",
        start="2026-08-04T09:30:00+00:00",  # 3:00 PM IST
        end="2026-08-04T10:00:00+00:00",   # 3:30 PM IST
    )]
    agent = _agent(_FakeCalendarClient(events))
    result = await agent.answer(_ctx("meetings today", time_zone="Asia/Kolkata"))
    # 09:30 UTC in Kolkata is 15:00 == 3:00 PM
    assert "3:00 PM" in result.text
    assert "3:30 PM" in result.text


# ── Connection-state copy (issue: "isn't connected", then reads meetings) ──
#
# The two failure modes need DIFFERENT words. Telling someone to "connect"
# something they already connected is the fastest way to lose them, and the
# reported symptom — chat says not-connected while the dashboard lists the same
# org's meetings — was exactly this distinction being collapsed.


async def test_no_google_grant_says_connect():
    agent = CalendarQueryAgent(db=None, calendar_client_factory=_factory(None))
    result = await agent.answer(_ctx("what meetings do I have today?"))
    assert "isn't connected" in result.text
    assert result.events == []


async def test_insufficient_scope_says_reconnect_not_connect():
    agent = CalendarQueryAgent(
        db=None,
        calendar_client_factory=_factory(
            None, error=RuntimeError("Calendar is connected read-only")
        ),
    )
    result = await agent.answer(_ctx("what meetings do I have today?"))
    assert "reconnect" in result.text.lower()
    assert "isn't connected" not in result.text
