"""MeetDeriver — turn Calendar signals into Meet signals.

A Meet meeting is a Calendar event that carries conferenceData / a hangoutLink.
We derive MEETING_CREATED / MEETING_UPDATED / MEETING_COMPLETED from the Calendar
events the Calendar connector already captured — no separate Google call.

Attendance (who actually joined, duration) is NOT derivable from Calendar and is
explicitly out of scope for phase 1. The `attendance_provider` extension point is
where the Google Meet API (conferenceRecords/participants) or the Admin SDK
Reports API would plug in later, without changing this contract.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalEventType,
    SignalSource,
)


class AttendanceProvider(Protocol):  # pragma: no cover - extension point only
    """Future: resolve real attendance for a finished meeting."""

    async def attendance_for(self, external_id: str) -> dict[str, Any]: ...


def _has_conference(payload: dict[str, Any]) -> bool:
    return bool(payload.get("conference_data") or payload.get("hangout_link"))


def _meeting_end(payload: dict[str, Any]) -> datetime | None:
    end = (payload.get("end") or {}).get("dateTime")
    if not end:
        return None
    try:
        dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def derive_meeting_events(
    calendar_events: list[SignalEvent], *, now: datetime | None = None
) -> list[SignalEvent]:
    now = now or datetime.now(timezone.utc)
    out: list[SignalEvent] = []
    for ev in calendar_events:
        if ev.source != SignalSource.CALENDAR:
            continue
        if ev.event_type == SignalEventType.CALENDAR_EVENT_CANCELLED:
            continue
        if not _has_conference(ev.payload):
            continue

        ctx = SignalContext(ev.org_id, ev.user_id, ev.integration_id)
        end = _meeting_end(ev.payload)
        completed = end is not None and end < now

        if completed:
            meeting_type = SignalEventType.MEETING_COMPLETED
        elif ev.event_type == SignalEventType.CALENDAR_EVENT_CREATED:
            meeting_type = SignalEventType.MEETING_CREATED
        else:
            meeting_type = SignalEventType.MEETING_UPDATED

        # Distinct version per phase so created/updated/completed are separate facts.
        version = f"{ev.external_version}:{meeting_type.value}"
        out.append(
            SignalEvent.create(
                ctx,
                source=SignalSource.MEET,
                event_type=meeting_type,
                external_id=ev.external_id,
                external_version=version,
                timestamp=ev.timestamp,
                payload={
                    "summary": ev.payload.get("summary"),
                    "start": ev.payload.get("start"),
                    "end": ev.payload.get("end"),
                    "attendees": ev.payload.get("attendees", []),
                    "conference_data": ev.payload.get("conference_data"),
                    "hangout_link": ev.payload.get("hangout_link"),
                },
                metadata={"derived_from": "calendar", "attendance_tracked": False},
            )
        )
    return out


class MeetDeriver:
    """Thin OO wrapper. Holds the (future) attendance extension point."""

    def __init__(self, attendance_provider: AttendanceProvider | None = None) -> None:
        self.attendance_provider = attendance_provider

    def derive(
        self, calendar_events: list[SignalEvent], *, now: datetime | None = None
    ) -> list[SignalEvent]:
        return derive_meeting_events(calendar_events, now=now)
