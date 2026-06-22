"""Calendar connector: created/updated/cancelled mapping, syncToken, 410 recovery,
multi-page drain, page-cap safety, payload minimization, and lineage."""
from __future__ import annotations

import pytest

from navis.signals.base import SignalEventType
from navis.signals.calendar.connector import CalendarConnector, CalendarSyncIncomplete
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


# ---- P0: multi-page drain + page-cap safety (never silently drop reality) ----

async def test_multipage_drain_captures_all_pages_and_gets_synctoken():
    pages = [
        {"items": [_created()], "nextPageToken": "1"},     # intermediate page (NO syncToken)
        {"items": [_updated()], "nextSyncToken": "FINAL"},  # final page yields the token
    ]
    res = await CalendarConnector().full_sync(make_ctx(), FakeCalendarClient(pages=pages))
    assert {e.external_id for e in res.events} == {"e1", "e2"}   # BOTH pages captured
    assert res.cursor == {"sync_token": "FINAL"}                 # token from the FINAL page


async def test_page_cap_fails_loud_not_silent(monkeypatch):
    from navis.platform.config import settings

    monkeypatch.setattr(settings, "calendar_max_sync_pages", 1)
    pages = [
        {"items": [_created()], "nextPageToken": "1"},   # more pages remain past the cap
        {"items": [_updated()], "nextSyncToken": "S"},
    ]
    with pytest.raises(CalendarSyncIncomplete):  # loud + retryable, never a stalled cursor
        await CalendarConnector().full_sync(make_ctx(), FakeCalendarClient(pages=pages))


# ---- Payload minimization + lineage ----

def _rich_raw():
    return {
        "id": "x", "status": "confirmed", "etag": '"1"',
        "created": "2026-01-01T00:00:00Z", "updated": "2026-01-01T00:00:00Z",
        "start": {"dateTime": "2026-02-01T10:00:00Z"}, "end": {"dateTime": "2026-02-01T11:00:00Z"},
        "attendees": [
            {"email": "a@x.com", "displayName": "Alice Smith", "responseStatus": "accepted",
             "comment": "private note", "additionalGuests": 2},
        ],
        "organizer": {"email": "o@x.com", "displayName": "Org Person", "self": True},
        "recurringEventId": "series-1",
        "originalStartTime": {"dateTime": "2026-02-01T09:00:00Z"},
        "conferenceData": {
            "conferenceId": "cid",
            "conferenceSolution": {"key": {"type": "hangoutsMeet"}},
            "entryPoints": [
                {"entryPointType": "video", "uri": "https://meet.google.com/abc"},
                {"entryPointType": "phone", "uri": "tel:+1-555-1234", "pin": "9999"},
            ],
        },
    }


def test_payload_minimized_drops_pii_keeps_identity_and_lineage():
    ev = CalendarConnector._to_event(make_ctx(), _rich_raw(), calendar_id="primary", capture_mode="full")
    att = ev.payload["attendees"][0]
    assert att == {"email": "a@x.com", "response_status": "accepted", "optional": False, "resource": False}
    assert "displayName" not in att and "comment" not in att and "additionalGuests" not in att  # PII dropped
    assert ev.payload["organizer"] == {"email": "o@x.com"}            # organizer minimized to email
    assert ev.payload["attendee_count"] == 1
    conf = ev.payload["conference_data"]
    assert conf["conference_id"] == "cid" and conf["has_video"] and conf["has_phone"]
    assert "entryPoints" not in conf and "9999" not in str(conf) and "tel:" not in str(conf)  # phone/PIN dropped
    assert ev.payload["original_start_time"] == {"dateTime": "2026-02-01T09:00:00Z"}  # moved-occurrence lineage
    assert ev.metadata["calendar_id"] == "primary" and ev.metadata["capture_mode"] == "full"  # lineage
    assert ev.metadata["is_recurring_instance"] is True


def test_downstream_email_contract_survives_minimization():
    # memory/state resolve people by attendees[].email — minimization MUST preserve it.
    ev = CalendarConnector._to_event(make_ctx(), _rich_raw(), calendar_id="cal-2", capture_mode="incremental")
    assert [a.get("email") for a in ev.payload["attendees"]] == ["a@x.com"]
    assert bool(ev.payload.get("conference_data") or ev.payload.get("hangout_link")) is True  # has_conference contract
