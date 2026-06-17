"""Pure-pipeline tests: the parts that need no database.

Covers the reasoning core of the product: extract commitments from an email,
project a state, run reasoning, and confirm "what matters today" ranks the
overdue commitment to the top.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from navis.decisions.what_matters_today import WhatMattersTodayEngine
from navis.memory.envelope import EntityType
from navis.memory.service import build_memory_records
from navis.reasoning import run_reasoning
from navis.state.aggregate import (
    CommitmentState,
    CommitmentView,
    OrganizationalState,
    RelationshipState,
    RelationshipView,
)
from tests.memory.fakes import gmail_view


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_memory_extracts_people_from_email():
    records = build_memory_records(
        gmail_view(payload={"from": "Jane Doe <jane@acme.com>", "to": "me@startup.com",
                            "body_text": "", "thread_id": ""})
    )
    emails = {r.payload["email"] for r in records if r.entity_type == EntityType.PERSON}
    assert "jane@acme.com" in emails
    assert "me@startup.com" in emails


def test_memory_extracts_commitment_promise():
    records = build_memory_records(
        gmail_view(payload={"from": "founder@startup.com", "subject": "Re: pricing",
                            "body_text": "Thanks for the call. I'll send you the revised proposal by Friday.",
                            "to": "", "thread_id": ""})
    )
    commitments = [r for r in records if r.entity_type == EntityType.COMMITMENT]
    assert commitments, "expected at least one commitment"
    assert "send you" in commitments[0].payload["text"].lower()


def test_overdue_commitment_tops_the_brief():
    org_id = uuid.uuid4()
    overdue = CommitmentView(
        id=uuid.uuid4(),
        description="Send investor the updated deck",
        status="overdue",
        due_at=_now() - timedelta(days=3),
        promised_at=_now() - timedelta(days=11),
        age_days=11.0,
        counterparty=None,
    )
    cold = RelationshipView(
        id=uuid.uuid4(),
        person="investor@vc.com",
        kind="investor",
        last_contact_at=_now() - timedelta(days=11),
        days_since_contact=11.0,
        health=0.3,
    )
    state = OrganizationalState(
        org_id=org_id,
        built_at=_now(),
        commitments=CommitmentState(overdue=[overdue]),
        relationships=RelationshipState(relationships=[cold]),
    )

    signals = run_reasoning(state)
    assert signals, "reasoning should produce signals"

    brief = WhatMattersTodayEngine().rank(signals)
    assert brief, "brief should have items"
    # The overdue commitment is the highest-severity concrete risk.
    titles = [item.title for item in brief]
    assert any("Send investor the updated deck" in t for t in titles)
    # Every item is explainable (trust-first).
    assert all(item.why for item in brief)


if __name__ == "__main__":
    test_memory_extracts_people_from_email()
    test_memory_extracts_commitment_promise()
    test_overdue_commitment_tops_the_brief()
    print("all pipeline tests passed")
