"""Deterministic extractors over Calendar SignalEvents.

A meeting is versioned by the signal's external_version (etag), so each edit to a
meeting is a NEW immutable MeetingMemory record — the meeting's append-only history.
Attendees become PersonMemory facts.
"""
from __future__ import annotations

from navis.memory.envelope import EntityType
from navis.memory.extractors.base import Extractor, MemoryDraft, SignalView
from navis.memory.schemas import current_version
from navis.signals.base import SignalSource

_CAL = {SignalSource.CALENDAR}


class MeetingExtractor(Extractor):
    name = "calendar_meeting"
    version = "1.0.0"
    sources = _CAL

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        p = signal.payload
        attendees = [a.get("email") for a in (p.get("attendees") or []) if a.get("email")]
        return [
            MemoryDraft(
                entity_type=EntityType.MEETING,
                entity_natural_key=signal.external_id,  # the calendar event id
                dedup_key=signal.external_version,        # one record per version
                payload={
                    "title": p.get("summary"),
                    "starts_at": (p.get("start") or {}).get("dateTime"),
                    "ends_at": (p.get("end") or {}).get("dateTime"),
                    "attendees": attendees,
                    "status": p.get("status"),
                    "has_conference": bool(p.get("conference_data") or p.get("hangout_link")),
                },
                confidence=0.98,
                schema_version=current_version(EntityType.MEETING),
            )
        ]


class CalendarAttendeeExtractor(Extractor):
    name = "calendar_attendee"
    version = "1.0.0"
    sources = _CAL

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        ver = current_version(EntityType.PERSON)
        drafts = []
        seen: set[str] = set()
        for a in signal.payload.get("attendees") or []:
            email = (a.get("email") or "").strip().lower()
            if not email or email in seen:
                continue
            seen.add(email)
            drafts.append(
                MemoryDraft(
                    entity_type=EntityType.PERSON,
                    entity_natural_key=email,
                    dedup_key=f"attendee:{email}",
                    payload={"email": email, "name": a.get("displayName"), "role": "attendee"},
                    confidence=0.9,
                    schema_version=ver,
                )
            )
        return drafts
