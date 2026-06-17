"""Meet derived from Calendar: created / completed, conference filter, versioning."""
from __future__ import annotations

from datetime import datetime, timezone

from navis.signals.base import SignalEvent, SignalEventType, SignalSource
from navis.signals.meet.deriver import derive_meeting_events
from tests.signals.fakes import make_ctx

NOW = datetime(2026, 6, 17, tzinfo=timezone.utc)


def _cal_event(ctx, *, external_id, etype, end, conference=True):
    payload = {
        "summary": "Sync",
        "start": {"dateTime": "2026-06-01T10:00:00Z"},
        "end": {"dateTime": end},
    }
    if conference:
        payload["hangout_link"] = "https://meet.google.com/abc"
    return SignalEvent.create(
        ctx,
        source=SignalSource.CALENDAR,
        event_type=etype,
        external_id=external_id,
        external_version="100",
        payload=payload,
    )


async def test_future_meeting_is_created_past_is_completed():
    ctx = make_ctx()
    future = _cal_event(
        ctx, external_id="e1", etype=SignalEventType.CALENDAR_EVENT_CREATED, end="2026-12-01T11:00:00Z"
    )
    past = _cal_event(
        ctx, external_id="e2", etype=SignalEventType.CALENDAR_EVENT_CREATED, end="2026-01-01T11:00:00Z"
    )
    events = derive_meeting_events([future, past], now=NOW)

    by_id = {e.external_id: e for e in events}
    assert by_id["e1"].event_type == SignalEventType.MEETING_CREATED
    assert by_id["e2"].event_type == SignalEventType.MEETING_COMPLETED
    assert all(e.source == SignalSource.MEET for e in events)


def test_events_without_conference_are_ignored():
    ctx = make_ctx()
    no_conf = _cal_event(
        ctx,
        external_id="e3",
        etype=SignalEventType.CALENDAR_EVENT_CREATED,
        end="2026-12-01T11:00:00Z",
        conference=False,
    )
    assert derive_meeting_events([no_conf], now=NOW) == []


def test_cancelled_calendar_events_produce_no_meeting():
    ctx = make_ctx()
    cancelled = _cal_event(
        ctx, external_id="e4", etype=SignalEventType.CALENDAR_EVENT_CANCELLED, end="2026-12-01T11:00:00Z"
    )
    assert derive_meeting_events([cancelled], now=NOW) == []


def test_meeting_version_is_distinct_per_phase():
    ctx = make_ctx()
    ev = _cal_event(
        ctx, external_id="e5", etype=SignalEventType.CALENDAR_EVENT_CREATED, end="2026-12-01T11:00:00Z"
    )
    meeting = derive_meeting_events([ev], now=NOW)[0]
    # Distinct version so MEETING_CREATED never dedupes against the calendar event.
    assert meeting.external_version == "100:meeting.created"
