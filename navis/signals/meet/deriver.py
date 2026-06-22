"""MeetDeriver — turn Calendar signals into Meet signals.

A Meet meeting is a Calendar event that carries conferenceData / a hangoutLink. We derive
the SCHEDULING facts (MEETING_CREATED / MEETING_UPDATED) from the Calendar events the
Calendar connector already captured — no separate Google call.

We do NOT derive MEETING_COMPLETED. Calendar is *scheduling* truth; completion is
*participation* truth (did anyone actually join?), which is ONLY knowable from the Google
Meet API (conferenceRecords/participants). Inferring "completed" from "scheduled end has
passed" (a) fabricated reality — a no-show meeting is not completed — and (b) was
non-deterministic: a re-sync/replay AFTER the end time flipped the same calendar event from
CREATED to COMPLETED, violating determinism + replay. MEETING_COMPLETED is reserved for the
future `attendance_provider` (Meet API), which joins by conference_id and APPENDS a new fact
— no rewrite/migration of historical signals.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalEventType,
    SignalSource,
)


class AttendanceProvider(Protocol):  # pragma: no cover - extension point only
    """Future: resolve real attendance for a finished meeting (Meet conferenceRecords)."""

    async def attendance_for(self, conference_id: str) -> dict[str, Any]: ...


def _has_conference(payload: dict[str, Any]) -> bool:
    return bool(payload.get("conference_data") or payload.get("hangout_link"))


def _conference_id(payload: dict[str, Any]) -> str | None:
    conf = payload.get("conference_data")
    return conf.get("conference_id") if isinstance(conf, dict) else None


def derive_meeting_events(
    calendar_events: list[SignalEvent], *, now: datetime | None = None
) -> list[SignalEvent]:
    """Derive Meet scheduling signals from Calendar signals. Pure + deterministic: the
    `now` parameter is accepted for backward compatibility but UNUSED (completion is no
    longer time-derived), so the output depends only on the input events."""
    out: list[SignalEvent] = []
    for ev in calendar_events:
        if ev.source != SignalSource.CALENDAR:
            continue
        if ev.event_type == SignalEventType.CALENDAR_EVENT_CANCELLED:
            continue
        if not _has_conference(ev.payload):
            continue

        ctx = SignalContext(ev.org_id, ev.user_id, ev.integration_id)
        # Scheduling facts only — deterministic in the calendar event_type, never in `now`.
        meeting_type = (
            SignalEventType.MEETING_CREATED
            if ev.event_type == SignalEventType.CALENDAR_EVENT_CREATED
            else SignalEventType.MEETING_UPDATED
        )
        conference_id = _conference_id(ev.payload)
        # Distinct version per phase so created/updated are separate facts and never dedupe
        # against the source calendar event.
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
                    # Stable conference identity — the join key future Meet-API signals
                    # (participants/recordings/transcripts) use to attach to THIS meeting.
                    "conference_id": conference_id,
                    "conference_data": ev.payload.get("conference_data"),  # minimized upstream
                    # hangout_link DROPPED — a join/access URL must not persist forever.
                    "has_conference": True,
                },
                metadata={
                    "derived_from": "calendar",
                    "attendance_tracked": False,
                    "conference_id": conference_id,
                    # Propagate capture lineage from the source calendar event.
                    "calendar_id": ev.metadata.get("calendar_id"),
                    "capture_mode": ev.metadata.get("capture_mode"),
                },
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
