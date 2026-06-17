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


def test_drive_extracts_project_from_folder():
    drafts = run_extractors(drive_view())
    projects = [d for d in drafts if d.entity_type == EntityType.PROJECT]
    assert projects and projects[0].payload["folder_id"] == "folder-42"


def test_extractors_are_deterministic():
    a = run_extractors(gmail_view())
    b = run_extractors(gmail_view())
    assert [(d.entity_type, d.dedup_key, tuple(sorted(d.payload.items()))) for d in a] == [
        (d.entity_type, d.dedup_key, tuple(sorted(d.payload.items()))) for d in b
    ]
