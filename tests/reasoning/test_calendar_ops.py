"""CalendarOpsAgent — reschedule / cancel an existing meeting.

These verbs had NO handler at all: "Reschedule the meeting to 5-6 PM" and
"Cancel the meeting" fell through to RAG-over-memory and were answered with
"Navis doesn't have enough memory to answer this yet".

The agent's job is targeting (WHICH meeting) plus slot resolution (to WHEN).
It must never guess between two plausible meetings, and never perform the
mutation itself — that is the execution worker's job, behind approval.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


from navis.reasoning.calendar_ops import (
    CalendarOpsAgent,
    CalendarOpsClarification,
    CalendarOpsContext,
    MeetingMutationProposal,
)
from navis.signals.google.resolve import CalendarNotConnected

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
# Fixed clock — Friday 2026-08-07 09:00 UTC.
NOW = datetime(2026, 8, 7, 9, 0, tzinfo=timezone.utc)


# ── Fakes ──────────────────────────────────────────────────────────────


class _FakeCalendarClient:
    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        self.events = events or []
        self.calls: list[dict[str, Any]] = []

    async def list_events_expanded(
        self, *, time_min: str, time_max: str, max_results: int = 50
    ) -> dict[str, Any]:
        self.calls.append({"time_min": time_min, "time_max": time_max})
        return {"items": self.events}


class _FakeTransport:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def _agent(
    client: _FakeCalendarClient | None,
    *,
    error: Exception | None = None,
    transport: _FakeTransport | None = None,
) -> CalendarOpsAgent:
    async def _factory(db, org_id):
        if error is not None:
            raise error
        return client, transport or _FakeTransport()

    return CalendarOpsAgent(db=None, calendar_client_factory=_factory)


def _ctx(
    utterance: str,
    operation: str = "reschedule",
    *,
    time_zone: str = "UTC",
    target_event_id: str | None = None,
) -> CalendarOpsContext:
    return CalendarOpsContext(
        org_id=ORG,
        utterance=utterance,
        time_zone=time_zone,
        now=NOW,
        operation=operation,
        target_event_id=target_event_id,
    )


def _event(
    *, eid: str, title: str, start: str, end: str, attendees: list[str] | None = None
) -> dict[str, Any]:
    return {
        "id": eid,
        "summary": title,
        "start": {"dateTime": start, "timeZone": "UTC"},
        "end": {"dateTime": end, "timeZone": "UTC"},
        "attendees": [{"email": e} for e in (attendees or [])],
    }


_INVESTOR = _event(
    eid="evt-investor",
    title="Investor sync",
    start="2026-08-07T14:00:00+00:00",
    end="2026-08-07T14:30:00+00:00",
    attendees=["arjun@fund.com"],
)
_STANDUP = _event(
    eid="evt-standup",
    title="Team standup",
    start="2026-08-10T09:00:00+00:00",
    end="2026-08-10T09:15:00+00:00",
    attendees=["team@navis.ai"],
)


# ── Targeting ──────────────────────────────────────────────────────────


async def test_single_upcoming_meeting_resolves_without_asking():
    agent = _agent(_FakeCalendarClient([_INVESTOR]))
    result = await agent.propose(_ctx("cancel the meeting", "cancel"))
    assert isinstance(result, MeetingMutationProposal)
    assert result.operation == "cancel"
    assert result.target.event_id == "evt-investor"


async def test_multiple_candidates_ask_instead_of_guessing():
    agent = _agent(_FakeCalendarClient([_INVESTOR, _STANDUP]))
    result = await agent.propose(_ctx("cancel the meeting", "cancel"))
    assert isinstance(result, CalendarOpsClarification)
    assert result.field == "event"
    assert {o["event_id"] for o in result.options} == {"evt-investor", "evt-standup"}


async def test_a_name_hint_narrows_to_one():
    agent = _agent(_FakeCalendarClient([_INVESTOR, _STANDUP]))
    result = await agent.propose(_ctx("cancel the standup", "cancel"))
    assert isinstance(result, MeetingMutationProposal)
    assert result.target.event_id == "evt-standup"


async def test_an_attendee_hint_narrows_to_one():
    agent = _agent(_FakeCalendarClient([_INVESTOR, _STANDUP]))
    result = await agent.propose(_ctx("cancel the meeting with arjun@fund.com", "cancel"))
    assert isinstance(result, MeetingMutationProposal)
    assert result.target.event_id == "evt-investor"


async def test_a_previously_picked_event_is_honoured():
    """The multi-turn path: the user answered "which one?" last turn."""
    agent = _agent(_FakeCalendarClient([_INVESTOR, _STANDUP]))
    result = await agent.propose(
        _ctx("cancel it", "cancel", target_event_id="evt-standup")
    )
    assert isinstance(result, MeetingMutationProposal)
    assert result.target.event_id == "evt-standup"


async def test_a_stale_picked_event_says_so():
    agent = _agent(_FakeCalendarClient([_INVESTOR]))
    result = await agent.propose(
        _ctx("cancel it", "cancel", target_event_id="evt-gone")
    )
    assert isinstance(result, CalendarOpsClarification)
    assert result.field == "none"
    assert "no longer" in result.question


async def test_empty_calendar_says_nothing_to_change():
    agent = _agent(_FakeCalendarClient([]))
    result = await agent.propose(_ctx("cancel the meeting", "cancel"))
    assert isinstance(result, CalendarOpsClarification)
    assert result.field == "none"
    assert "no upcoming meetings" in result.question


async def test_past_events_are_never_offered():
    past = _event(
        eid="evt-past",
        title="Yesterday review",
        start="2026-08-06T10:00:00+00:00",
        end="2026-08-06T10:30:00+00:00",
    )
    agent = _agent(_FakeCalendarClient([past, _INVESTOR]))
    result = await agent.propose(_ctx("cancel the meeting", "cancel"))
    assert isinstance(result, MeetingMutationProposal)
    assert result.target.event_id == "evt-investor"


# ── Reschedule slot resolution ─────────────────────────────────────────


async def test_the_reported_utterance_produces_a_concrete_new_slot():
    """"Reschedule the meeting to 5–6 PM" — including the EN DASH, which is
    what the user actually typed and what no ASCII range regex matched."""
    agent = _agent(_FakeCalendarClient([_INVESTOR]))
    result = await agent.propose(_ctx("Reschedule the meeting to 5–6 PM"))
    assert isinstance(result, MeetingMutationProposal)
    assert result.operation == "reschedule"
    assert result.new_start.isoformat() == "2026-08-07T17:00:00+00:00"
    assert result.new_end.isoformat() == "2026-08-07T18:00:00+00:00"


async def test_a_range_defines_the_new_duration():
    agent = _agent(_FakeCalendarClient([_INVESTOR]))
    result = await agent.propose(_ctx("move it to 5-6 PM"))
    payload = result.to_execution_payload()
    assert payload["duration_minutes"] == 60  # not the original 30


async def test_a_single_time_keeps_the_existing_duration():
    agent = _agent(_FakeCalendarClient([_INVESTOR]))
    result = await agent.propose(_ctx("move it to 4 PM"))
    assert result.new_start.isoformat() == "2026-08-07T16:00:00+00:00"
    assert result.to_execution_payload()["duration_minutes"] == 30
    assert any("30-min" in a for a in result.assumptions)


async def test_a_time_with_no_day_keeps_the_existing_day_and_says_so():
    agent = _agent(_FakeCalendarClient([_INVESTOR]))
    result = await agent.propose(_ctx("push it to 4 PM"))
    assert result.new_start.date().isoformat() == "2026-08-07"
    assert any("Kept it on" in a for a in result.assumptions)


async def test_a_day_with_no_time_keeps_the_existing_time():
    agent = _agent(_FakeCalendarClient([_INVESTOR]))
    result = await agent.propose(_ctx("move it to Monday"))
    assert isinstance(result, MeetingMutationProposal)
    assert result.new_start.isoformat().startswith("2026-08-10T14:00")


async def test_no_new_time_at_all_asks():
    agent = _agent(_FakeCalendarClient([_INVESTOR]))
    result = await agent.propose(_ctx("reschedule the meeting"))
    assert isinstance(result, CalendarOpsClarification)
    assert result.field == "time"
    assert result.target_event_id == "evt-investor"


async def test_a_time_already_past_is_refused():
    agent = _agent(_FakeCalendarClient([_INVESTOR]))
    result = await agent.propose(_ctx("move it to 7 AM"))  # NOW is 09:00
    assert isinstance(result, CalendarOpsClarification)
    assert result.field == "time"
    assert "already passed" in result.question


# ── Execution payload contract ─────────────────────────────────────────


async def test_reschedule_payload_matches_the_sender_contract():
    agent = _agent(_FakeCalendarClient([_INVESTOR]))
    result = await agent.propose(_ctx("move it to 5-6 PM"))
    payload = result.to_execution_payload()
    # RescheduleMeetingProvider.validate requires both of these.
    assert payload["event_id"] == "evt-investor"
    assert payload["start"] == "2026-08-07T17:00:00+00:00"
    assert payload["end"] == "2026-08-07T18:00:00+00:00"
    # Lineage for the audit row.
    assert payload["previous_start"] == "2026-08-07T14:00:00+00:00"


async def test_cancel_payload_matches_the_sender_contract():
    agent = _agent(_FakeCalendarClient([_INVESTOR]))
    result = await agent.propose(_ctx("cancel the meeting", "cancel"))
    payload = result.to_execution_payload()
    assert payload["event_id"] == "evt-investor"
    assert "start" not in payload  # nothing is being moved
    assert payload["attendees"] == ["arjun@fund.com"]


# ── Connection state ───────────────────────────────────────────────────


async def test_no_google_grant_says_connect():
    agent = _agent(None, error=CalendarNotConnected("none"))
    result = await agent.propose(_ctx("cancel the meeting", "cancel"))
    assert isinstance(result, CalendarOpsClarification)
    assert result.field == "scope"
    assert "isn't connected" in result.question


async def test_insufficient_scope_says_reconnect():
    agent = _agent(None, error=RuntimeError("read-only grant"))
    result = await agent.propose(_ctx("cancel the meeting", "cancel"))
    assert result.field == "scope"
    assert "reconnect" in result.question.lower()
    assert "isn't connected" not in result.question


async def test_the_transport_is_always_closed():
    """A leaked transport leaks a pooled connection on every failed lookup."""
    transport = _FakeTransport()
    agent = _agent(_FakeCalendarClient([_INVESTOR]), transport=transport)
    await agent.propose(_ctx("cancel the meeting", "cancel"))
    assert transport.closed
