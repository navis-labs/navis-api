"""Graceful degradation — answer from Calendar/Gmail when Memory is empty.

The reported behaviour: "What should I do first?" and "Who am I keeping
waiting?" came back with "Navis doesn't have enough memory to answer this yet"
while the org's calendar and inbox were full of signal. Memory lags ingestion by
design (extractors run behind the sync); the raw signal tables do not.

These tests pin the three outcomes that matter:
  * signals present  -> real cited facts, no "not enough memory";
  * nothing synced but sources connected -> a message that NAMES them;
  * nothing connected -> a message that says to connect something.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


from navis.api.workspace_fallback import (
    build_workspace_context,
    connected_source_labels,
)
from navis.identity.models import IntegrationProvider

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
NOW = datetime(2026, 8, 7, 9, 0, tzinfo=timezone.utc)


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeDB:
    """Answers `scalars()` from a queue, in call order.

    `build_workspace_context` issues three queries, in this order:
      1. external ids of recently CANCELLED calendar events,
      2. upcoming calendar events,
      3. recent inbound Gmail.
    """

    def __init__(self, *batches):
        self._batches = list(batches)
        self.calls = 0

    async def scalars(self, _stmt):
        self.calls += 1
        rows = self._batches.pop(0) if self._batches else []
        return _Scalars(rows)


def _db(*, cancelled=(), calendar=(), gmail=()):
    """Named constructor — positional batches were unreadable once the
    cancellation query joined the sequence."""
    return _FakeDB(list(cancelled), list(calendar), list(gmail))


def _calendar_row(
    *, external_id: str, summary: str, at: datetime, ingested: datetime | None = None
):
    # `occurred_at` is the MEETING START; `created_at` is when we learned about
    # it. Version-picking must use the latter — see _calendar_facts.
    return SimpleNamespace(
        id=uuid.uuid4(),
        external_id=external_id,
        occurred_at=at,
        created_at=ingested or at,
        payload={
            "summary": summary,
            "organizer": {"email": "founder@acme.com"},
            "attendee_count": 3,
        },
    )


def _gmail_row(*, sender: str, subject: str, at: datetime):
    return SimpleNamespace(
        id=uuid.uuid4(),
        external_id=f"msg-{subject}",
        occurred_at=at,
        payload={
            "sender_email": sender,
            "subject": subject,
            "snippet": "Following up on the term sheet.",
        },
    )


# ── Facts are built from raw signals ───────────────────────────────────


async def test_calendar_and_gmail_signals_become_cited_facts():
    db = _db(
        calendar=[_calendar_row(
            external_id="evt-1", summary="Investor sync", at=NOW + timedelta(hours=5)
        )],
        gmail=[_gmail_row(sender="arjun@fund.com", subject="Term sheet", at=NOW - timedelta(days=1))],
    )
    ctx = await build_workspace_context(db, ORG, "What should I do first?", now=NOW)

    assert len(ctx.facts) == 2
    kinds = {f.kind for f in ctx.facts}
    assert kinds == {"calendar", "gmail"}
    # Real lineage, not a synthesised label — every fact points at a signal row.
    assert all(f.source.startswith("signal:") for f in ctx.facts)
    assert any("Investor sync" in f.text for f in ctx.facts)
    assert any("arjun@fund.com" in f.text for f in ctx.facts)


async def test_repeated_event_versions_collapse_to_the_NEWEST_fact():
    """signal_events stores one row per event VERSION (etag-driven).

    Every version of a meeting shares the same `occurred_at` (the start time),
    so ordering by it cannot tell versions apart at all — the old "first row
    wins" picked an arbitrary edit, which for an edited meeting meant citing a
    title or time the founder had already changed. Ingestion order is the only
    ordering that means "most recent truth"."""
    rows = [
        _calendar_row(
            external_id="evt-1", summary="Investor sync v1",
            at=NOW + timedelta(hours=5), ingested=NOW - timedelta(days=2),
        ),
        _calendar_row(
            external_id="evt-1", summary="Investor sync v2",
            at=NOW + timedelta(hours=5), ingested=NOW - timedelta(minutes=5),
        ),
        _calendar_row(
            external_id="evt-2", summary="Standup",
            at=NOW + timedelta(hours=8), ingested=NOW - timedelta(days=1),
        ),
    ]
    db = _db(calendar=rows)
    ctx = await build_workspace_context(db, ORG, "what's coming up?", now=NOW)

    assert len(ctx.facts) == 2
    assert any("v2" in f.text for f in ctx.facts), "the latest edit must win"
    assert not any("v1" in f.text for f in ctx.facts)
    # Chronological for the reader, regardless of ingestion order.
    assert [f.occurred_at for f in ctx.facts] == sorted(f.occurred_at for f in ctx.facts)


async def test_a_cancelled_meeting_is_never_cited():
    """THE read-after-write case. Cancelling a meeting does not delete its
    original CREATED signal row, and that row's occurred_at is the meeting's
    (still future) start — so it sails through the upcoming-events window
    untouched. Matching the cancellation to the EVENT is what removes it."""
    rows = [
        _calendar_row(
            external_id="evt-cancelled", summary="Investor sync",
            at=NOW + timedelta(days=3), ingested=NOW - timedelta(days=5),
        ),
        _calendar_row(
            external_id="evt-live", summary="Standup",
            at=NOW + timedelta(hours=2), ingested=NOW - timedelta(days=1),
        ),
    ]
    db = _db(cancelled=["evt-cancelled"], calendar=rows)
    ctx = await build_workspace_context(db, ORG, "what's my next meeting?", now=NOW)

    assert len(ctx.facts) == 1
    assert "Standup" in ctx.facts[0].text
    assert not any("Investor sync" in f.text for f in ctx.facts)


async def test_no_signals_yields_an_empty_context_not_an_invented_one():
    db = _db()
    ctx = await build_workspace_context(db, ORG, "what should I do first?", now=NOW)
    assert ctx.facts == []
    assert ctx.as_of == NOW


async def test_a_failing_source_does_not_take_the_whole_fallback_down():
    """A fallback that raises is worse than a thin one — the user would get an
    error instead of the half of the picture that IS available."""

    class _HalfBrokenDB:
        def __init__(self):
            self.calls = 0

        async def scalars(self, _stmt):
            self.calls += 1
            # Call 1 is the cancellation lookup, which aborts the whole
            # calendar branch. Gmail is then call 2 and must still succeed.
            if self.calls == 1:
                raise RuntimeError("calendar query exploded")
            return _Scalars(
                [_gmail_row(sender="a@b.com", subject="Hi", at=NOW - timedelta(hours=2))]
            )

    ctx = await build_workspace_context(_HalfBrokenDB(), ORG, "q", now=NOW)
    assert len(ctx.facts) == 1
    assert ctx.facts[0].kind == "gmail"


# ── The honest empty state ─────────────────────────────────────────────


async def test_connected_sources_are_named():
    """"Gmail is connected but hasn't synced" is actionable. "Not enough
    memory" is not — it names an internal concept the founder can't act on."""
    db = _FakeDB([IntegrationProvider.GMAIL, IntegrationProvider.GOOGLE_CALENDAR])
    assert await connected_source_labels(db, ORG) == ["Gmail", "Google Calendar"]


async def test_no_connected_sources_returns_empty():
    db = _FakeDB([])
    assert await connected_source_labels(db, ORG) == []


async def test_duplicate_provider_rows_are_deduped():
    db = _FakeDB([IntegrationProvider.GMAIL, IntegrationProvider.GMAIL])
    assert await connected_source_labels(db, ORG) == ["Gmail"]
