"""Deterministic extractor over Meet SignalEvents (derived from Calendar).

A Meet signal carries the stable `conference_id` — the join key that future participant /
recording / transcript signals will use to attach to this meeting. We emit a MEETING record
keyed on the same calendar event id the Calendar extractor uses (entity_natural_key =
external_id), so Meet facts and Calendar facts converge on ONE meeting entity over time.
Attendees become PersonMemory facts. No completion/participation inference (that is reality
only the Meet API can confirm — out of scope for Memory).
"""
from __future__ import annotations

from navis.memory.envelope import EntityType
from navis.memory.extractors.base import Extractor, MemoryDraft, SignalView
from navis.memory.schemas import current_version
from navis.memory.confidence import HIGH, STRUCTURAL
from navis.signals.base import SignalSource


class MeetMeetingExtractor(Extractor):
    name = "meet_meeting"
    version = "1.0.0"
    sources = {SignalSource.MEET}

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        p = signal.payload
        attendees = [a.get("email") for a in (p.get("attendees") or []) if a.get("email")]
        drafts: list[MemoryDraft] = [MemoryDraft(
            entity_type=EntityType.MEETING,
            entity_natural_key=signal.external_id,       # same id space as Calendar meetings
            dedup_key=f"meet:{signal.external_version}",  # distinct from the Calendar record
            payload={
                "title": p.get("summary"),
                "starts_at": (p.get("start") or {}).get("dateTime"),
                "ends_at": (p.get("end") or {}).get("dateTime"),
                "attendees": attendees,
                "status": None,
                "has_conference": True,
                "conference_id": p.get("conference_id"),  # the stable join key
            },
            confidence=STRUCTURAL,
            schema_version=current_version(EntityType.MEETING),
        )]
        ver = current_version(EntityType.PERSON)
        seen: set[str] = set()
        for a in p.get("attendees") or []:
            email = (a.get("email") or "").strip().lower()
            if email and email not in seen:
                seen.add(email)
                drafts.append(MemoryDraft(
                    entity_type=EntityType.PERSON, entity_natural_key=email,
                    dedup_key=f"meet_attendee:{email}",
                    payload={"email": email, "name": None, "role": "attendee", "source": "meet"},
                    confidence=HIGH, schema_version=ver,
                ))
        return drafts
