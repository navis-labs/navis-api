"""Event-type routing (Issue A fix) + pointer lineage (Issue E fix) — unit, no DB."""
from __future__ import annotations

from navis.platform.events import EventType
from navis.signals.base import SignalEvent, SignalEventType, SignalSource
from navis.signals.service import (
    SignalIngestionService,
    bus_event_for_type_value,
    event_type_for,
)
from tests.signals.fakes import make_ctx


def _ev(source, event_type, **md):
    return SignalEvent.create(
        make_ctx(), source=source, event_type=event_type,
        external_id="x1", external_version="1", metadata=md or None,
    )


# ---- Issue A: route by event_type, not source ----

def test_calendar_events_route_to_distinct_event_types():
    assert event_type_for(_ev(SignalSource.CALENDAR, SignalEventType.CALENDAR_EVENT_CREATED)) == EventType.CALENDAR_EVENT_CREATED
    assert event_type_for(_ev(SignalSource.CALENDAR, SignalEventType.CALENDAR_EVENT_UPDATED)) == EventType.CALENDAR_EVENT_UPDATED
    assert event_type_for(_ev(SignalSource.CALENDAR, SignalEventType.CALENDAR_EVENT_CANCELLED)) == EventType.CALENDAR_EVENT_CANCELLED


def test_email_routes_and_unknown_falls_back():
    assert event_type_for(_ev(SignalSource.GMAIL, SignalEventType.EMAIL_RECEIVED)) == EventType.GMAIL_RECEIVED
    assert event_type_for(_ev(SignalSource.MEET, SignalEventType.MEETING_COMPLETED)) == EventType.MEETING_COMPLETED
    # Unmapped event_type -> generic SIGNAL_INGESTED (drive/docs/meeting-created etc.)
    assert event_type_for(_ev(SignalSource.DRIVE, SignalEventType.DRIVE_FILE_CREATED)) == EventType.SIGNAL_INGESTED
    assert bus_event_for_type_value("not.a.real.type") == EventType.SIGNAL_INGESTED


# ---- Issue E: pointer carries lineage, never the body ----

def test_pointer_payload_carries_lineage_no_body():
    e = _ev(SignalSource.GMAIL, SignalEventType.EMAIL_RECEIVED)
    p = SignalIngestionService._pointer_payload(e)
    assert p["event_id"] == str(e.event_id)
    assert p["schema_version"] == e.schema_version
    assert p["captured_at"]                       # capture-time lineage present
    assert "payload" not in p and "metadata" not in p  # ids/lineage only, no body


def test_pointer_payload_surfaces_optional_lineage_from_metadata():
    e = _ev(SignalSource.GMAIL, SignalEventType.EMAIL_RECEIVED,
            connector_version="gmail-2.3", sync_run_id="run-abc")
    p = SignalIngestionService._pointer_payload(e)
    assert p["connector_version"] == "gmail-2.3"
    assert p["sync_run_id"] == "run-abc"
