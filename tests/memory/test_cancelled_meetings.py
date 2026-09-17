"""Cancelled meetings must not be citable as live.

Memory is append-only on purpose. Cancelling a meeting does NOT remove the
`status: confirmed` record written when it was booked, and that record carries
far more text (title, attendees, organiser) than the cancellation stub — so
vector search reliably ranks the stale version higher.

That is how the assistant could keep insisting a deleted meeting exists while
truthfully citing its own memory. These tests pin the read-side suppression:
history is preserved, but a cancelled meeting is not offered as current.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone


from navis.memory.meetings import (
    cancelled_meeting_entity_ids,
    cancelled_meeting_memory_ids,
)

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
NOW = datetime(2026, 8, 7, 9, 0, tzinfo=timezone.utc)

MEETING_A = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
MEETING_B = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000002")


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeDB:
    """Serves pre-sorted rows, mimicking the ORDER BY in the real queries.

    Deliberately does NOT re-sort: the production code relies on Postgres
    returning entity rows newest-first, so a fake that sorted for it would hide
    a broken ORDER BY.
    """

    def __init__(self, *batches):
        self._batches = list(batches)
        self.calls = 0

    async def execute(self, _stmt):
        self.calls += 1
        return _Result(self._batches.pop(0) if self._batches else [])


def _record(entity_id, status, *, at, memory_id=None):
    """(entity_id, payload, event_timestamp, memory_id) — the projection the
    entity query selects."""
    return (entity_id, {"status": status}, at, memory_id or uuid.uuid4())


# ── Latest-version wins ────────────────────────────────────────────────


async def test_a_meeting_whose_latest_record_is_cancelled_is_cancelled():
    db = _FakeDB([
        _record(MEETING_A, "cancelled", at=NOW),               # newest first
        _record(MEETING_A, "confirmed", at=NOW - timedelta(days=1)),
    ])
    assert await cancelled_meeting_entity_ids(db, ORG, [MEETING_A]) == {MEETING_A}


async def test_a_live_meeting_is_not_suppressed():
    db = _FakeDB([
        _record(MEETING_A, "confirmed", at=NOW),
        _record(MEETING_A, "confirmed", at=NOW - timedelta(days=1)),
    ])
    assert await cancelled_meeting_entity_ids(db, ORG, [MEETING_A]) == set()


async def test_a_re_created_meeting_reads_as_live_again():
    """Cancel then re-book. The newest record wins, so the meeting comes back —
    suppressing it forever would be its own staleness bug."""
    db = _FakeDB([
        _record(MEETING_A, "confirmed", at=NOW),                     # re-created
        _record(MEETING_A, "cancelled", at=NOW - timedelta(hours=2)),
        _record(MEETING_A, "confirmed", at=NOW - timedelta(days=1)),
    ])
    assert await cancelled_meeting_entity_ids(db, ORG, [MEETING_A]) == set()


async def test_only_the_cancelled_meeting_is_suppressed():
    db = _FakeDB([
        _record(MEETING_A, "cancelled", at=NOW),
        _record(MEETING_B, "confirmed", at=NOW),
    ])
    result = await cancelled_meeting_entity_ids(db, ORG, [MEETING_A, MEETING_B])
    assert result == {MEETING_A}


async def test_a_missing_status_is_not_treated_as_cancelled():
    """Absence of evidence is not evidence of cancellation — a record with no
    status must never remove a real meeting from an answer."""
    db = _FakeDB([(MEETING_A, {}, NOW, uuid.uuid4())])
    assert await cancelled_meeting_entity_ids(db, ORG, [MEETING_A]) == set()


async def test_status_casing_does_not_matter():
    db = _FakeDB([_record(MEETING_A, "CANCELLED", at=NOW)])
    assert await cancelled_meeting_entity_ids(db, ORG, [MEETING_A]) == {MEETING_A}


# ── memory_id -> entity join ───────────────────────────────────────────


async def test_memory_ids_for_a_cancelled_meeting_are_returned():
    """Retrieval holds memory ids, not entity ids. EVERY fact about the
    cancelled meeting must be suppressed — including the rich `confirmed` one
    that outranks the cancellation stub."""
    confirmed_fact = uuid.uuid4()
    cancelled_fact = uuid.uuid4()
    live_fact = uuid.uuid4()

    db = _FakeDB(
        # 1. memory_id -> entity_id
        [
            (confirmed_fact, MEETING_A),
            (cancelled_fact, MEETING_A),
            (live_fact, MEETING_B),
        ],
        # 2. entity records, newest first
        [
            _record(MEETING_A, "cancelled", at=NOW),
            _record(MEETING_A, "confirmed", at=NOW - timedelta(days=1)),
            _record(MEETING_B, "confirmed", at=NOW),
        ],
    )
    suppressed = await cancelled_meeting_memory_ids(
        db, ORG, [confirmed_fact, cancelled_fact, live_fact]
    )
    assert suppressed == {confirmed_fact, cancelled_fact}
    assert live_fact not in suppressed


async def test_non_meeting_facts_are_never_suppressed():
    """The query filters on entity_type=MEETING, so a person/commitment fact
    simply never appears — an empty first batch means "nothing to suppress"."""
    db = _FakeDB([])
    assert await cancelled_meeting_memory_ids(db, ORG, [uuid.uuid4()]) == set()


# ── Cheapness ──────────────────────────────────────────────────────────


async def test_empty_input_costs_no_queries():
    """This runs inside a chat turn; an empty candidate set must not round-trip."""
    db = _FakeDB()
    assert await cancelled_meeting_entity_ids(db, ORG, []) == set()
    assert await cancelled_meeting_memory_ids(db, ORG, []) == set()
    assert db.calls == 0


async def test_none_ids_are_ignored():
    db = _FakeDB()
    assert await cancelled_meeting_entity_ids(db, ORG, [None, None]) == set()
    assert db.calls == 0
