"""Meet derived from Calendar: scheduling facts only (created/updated), conference filter,
versioning, determinism (no time-derived completion), conference identity, privacy, lineage."""
from __future__ import annotations

from datetime import datetime, timezone

from navis.signals.base import SignalEvent, SignalEventType, SignalSource
from navis.signals.meet.deriver import derive_meeting_events
from tests.signals.fakes import make_ctx

NOW = datetime(2026, 6, 17, tzinfo=timezone.utc)


def _cal_event(ctx, *, external_id, etype, end, conference=True, conf_id="cid", metadata=None):
    payload = {
        "summary": "Sync",
        "start": {"dateTime": "2026-06-01T10:00:00Z"},
        "end": {"dateTime": end},
    }
    if conference:
        payload["hangout_link"] = "https://meet.google.com/abc"
        payload["conference_data"] = {"conference_id": conf_id, "solution": "hangoutsMeet"}
    return SignalEvent.create(
        ctx,
        source=SignalSource.CALENDAR,
        event_type=etype,
        external_id=external_id,
        external_version="100",
        payload=payload,
        metadata=metadata or {},
    )


async def test_calendar_meetings_are_scheduling_facts_never_time_derived_completed():
    # A past end-time must NOT fabricate MEETING_COMPLETED — completion is participation
    # truth (Meet API), not derivable from the calendar schedule.
    ctx = make_ctx()
    future = _cal_event(
        ctx, external_id="e1", etype=SignalEventType.CALENDAR_EVENT_CREATED, end="2026-12-01T11:00:00Z"
    )
    past = _cal_event(
        ctx, external_id="e2", etype=SignalEventType.CALENDAR_EVENT_CREATED, end="2026-01-01T11:00:00Z"
    )
    updated = _cal_event(
        ctx, external_id="e3", etype=SignalEventType.CALENDAR_EVENT_UPDATED, end="2026-01-01T11:00:00Z"
    )
    events = derive_meeting_events([future, past, updated], now=NOW)

    by_id = {e.external_id: e for e in events}
    assert by_id["e1"].event_type == SignalEventType.MEETING_CREATED
    assert by_id["e2"].event_type == SignalEventType.MEETING_CREATED  # past, but still just "created"
    assert by_id["e3"].event_type == SignalEventType.MEETING_UPDATED
    assert all(e.event_type != SignalEventType.MEETING_COMPLETED for e in events)
    assert all(e.source == SignalSource.MEET for e in events)


async def test_derivation_is_deterministic_in_now():
    # Same input + different `now` => identical event_ids (no hidden time dependence).
    ctx = make_ctx()
    ev = _cal_event(
        ctx, external_id="e1", etype=SignalEventType.CALENDAR_EVENT_CREATED, end="2026-01-01T11:00:00Z"
    )
    a = derive_meeting_events([ev], now=datetime(2025, 1, 1, tzinfo=timezone.utc))[0]
    b = derive_meeting_events([ev], now=datetime(2030, 1, 1, tzinfo=timezone.utc))[0]
    assert a.event_id == b.event_id
    assert a.event_type == b.event_type == SignalEventType.MEETING_CREATED


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


def test_conference_identity_surfaced_and_join_url_dropped():
    ctx = make_ctx()
    ev = _cal_event(
        ctx, external_id="e6", etype=SignalEventType.CALENDAR_EVENT_CREATED,
        end="2026-12-01T11:00:00Z", conf_id="conf-xyz",
    )
    m = derive_meeting_events([ev], now=NOW)[0]
    # Stable join key surfaced for future Meet-API signals.
    assert m.payload["conference_id"] == "conf-xyz"
    assert m.metadata["conference_id"] == "conf-xyz"
    # Privacy: the join/access URL must NOT persist.
    assert "hangout_link" not in m.payload
    assert "https://meet.google.com" not in str(m.payload)
    assert m.payload["has_conference"] is True


def test_capture_lineage_propagated_from_calendar():
    ctx = make_ctx()
    ev = _cal_event(
        ctx, external_id="e7", etype=SignalEventType.CALENDAR_EVENT_CREATED,
        end="2026-12-01T11:00:00Z",
        metadata={"calendar_id": "primary", "capture_mode": "full"},
    )
    m = derive_meeting_events([ev], now=NOW)[0]
    assert m.metadata["derived_from"] == "calendar"
    assert m.metadata["attendance_tracked"] is False
    assert m.metadata["calendar_id"] == "primary"
    assert m.metadata["capture_mode"] == "full"
