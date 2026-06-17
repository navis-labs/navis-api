"""Calendar connector: created/updated/cancelled mapping, syncToken, 410 recovery."""
from __future__ import annotations

from navis.signals.base import SignalEventType
from navis.signals.calendar.connector import CalendarConnector
from tests.signals.fakes import FakeCalendarClient, make_ctx


def _created():
    return {
        "id": "e1",
        "status": "confirmed",
        "etag": '"100"',
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "start": {"dateTime": "2026-02-01T10:00:00Z"},
        "end": {"dateTime": "2026-02-01T11:00:00Z"},
    }


def _updated():
    return {
        "id": "e2",
        "status": "confirmed",
        "etag": '"205"',
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-05T00:00:00Z",
        "start": {"dateTime": "2026-02-02T10:00:00Z"},
        "end": {"dateTime": "2026-02-02T11:00:00Z"},
    }


def _cancelled():
    return {"id": "e3", "status": "cancelled", "etag": '"9"'}


async def test_maps_created_updated_cancelled_and_versions_by_etag():
    pages = [{"items": [_created(), _updated(), _cancelled()], "nextSyncToken": "SYNC-2"}]
    res = await CalendarConnector().full_sync(make_ctx(), FakeCalendarClient(pages=pages))

    by_id = {e.external_id: e for e in res.events}
    assert by_id["e1"].event_type == SignalEventType.CALENDAR_EVENT_CREATED
    assert by_id["e2"].event_type == SignalEventType.CALENDAR_EVENT_UPDATED
    assert by_id["e3"].event_type == SignalEventType.CALENDAR_EVENT_CANCELLED
    # version-aware via etag (quotes stripped)
    assert by_id["e2"].external_version == "205"
    assert res.cursor == {"sync_token": "SYNC-2"}


async def test_incremental_410_requests_full_resync():
    client = FakeCalendarClient(raises_410=True)
    res = await CalendarConnector().incremental_sync(make_ctx(), client, {"sync_token": "OLD"})
    assert res.full_resync_required is True


async def test_recurring_and_conference_flags_in_metadata():
    recurring = {
        "id": "r1",
        "status": "confirmed",
        "etag": '"1"',
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "recurrence": ["RRULE:FREQ=WEEKLY"],
        "hangoutLink": "https://meet.google.com/abc",
        "start": {"dateTime": "2026-02-02T10:00:00Z"},
        "end": {"dateTime": "2026-02-02T11:00:00Z"},
    }
    res = await CalendarConnector().full_sync(
        make_ctx(), FakeCalendarClient(pages=[{"items": [recurring], "nextSyncToken": "S"}])
    )
    ev = res.events[0]
    assert ev.metadata["is_recurring_master"] is True
    assert ev.metadata["has_conference"] is True
