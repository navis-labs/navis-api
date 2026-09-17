"""Extractor correctness + determinism across Gmail/Calendar/Drive."""
from __future__ import annotations

from navis.memory.envelope import EntityType
from navis.memory.extractors import run_extractors
from tests.memory.fakes import calendar_view, drive_view, gmail_view


def _types(drafts):
    return {d.entity_type for d in drafts}


def test_gmail_extracts_people_company_thread_relationship_commitment_decision():
    drafts = run_extractors(gmail_view())
    t = _types(drafts)
    assert EntityType.PERSON in t
    assert EntityType.COMPANY in t  # acme.com / vc.com are non-free domains
    assert EntityType.THREAD in t
    assert EntityType.RELATIONSHIP in t
    assert EntityType.COMMITMENT in t  # "I'll send you ... by Friday"
    assert EntityType.DECISION in t   # "We have decided to raise"


def test_company_excludes_free_domains():
    drafts = run_extractors(gmail_view(payload={"from": "x <x@gmail.com>", "to": "y <y@acme.com>"}))
    companies = {d.payload["domain"] for d in drafts if d.entity_type == EntityType.COMPANY}
    assert "gmail.com" not in companies
    assert "acme.com" in companies


def test_calendar_extracts_meeting_and_attendees():
    drafts = run_extractors(calendar_view())
    t = _types(drafts)
    assert EntityType.MEETING in t
    assert EntityType.PERSON in t  # attendees
    meeting = next(d for d in drafts if d.entity_type == EntityType.MEETING)
    assert meeting.payload["has_conference"] is True
    assert meeting.dedup_key == "etag-7"  # versioned by etag


def test_drive_folder_does_not_become_a_project():
    """A Drive folder is a storage container, not an organizational project.
    "Receipts" / "Untitled folder" must never surface as a confirmed Project —
    and since the Drive path was PROJECT's only producer, inventing one here is
    the whole failure mode. The Drive SignalEvent itself is unaffected."""
    drafts = run_extractors(drive_view())

    assert [d for d in drafts if d.entity_type == EntityType.PROJECT] == []


def test_project_entity_type_remains_supported_for_real_sources():
    """07D removed a bad PRODUCER, not the Project model: the entity type, its
    payload schema and State's fold all stay, so a genuine project signal can
    populate PROJECT without rebuilding any of this."""
    from navis.memory.schemas import current_version
    from navis.memory.schemas.payloads import PAYLOAD_FIELDS_V1

    assert current_version(EntityType.PROJECT) == 1
    assert "folder_id" in PAYLOAD_FIELDS_V1[EntityType.PROJECT]


def test_extractors_are_deterministic():
    a = run_extractors(gmail_view())
    b = run_extractors(gmail_view())
    assert [(d.entity_type, d.dedup_key, tuple(sorted(d.payload.items()))) for d in a] == [
        (d.entity_type, d.dedup_key, tuple(sorted(d.payload.items()))) for d in b
    ]
