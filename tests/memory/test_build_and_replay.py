"""The core guarantees: deterministic build, replay-identical, idempotent append,
full provenance, and zero intelligence."""
from __future__ import annotations

from navis.memory.envelope import MemoryRecord
from navis.memory.service import MemoryIngestionService, build_memory_records
from tests.memory.fakes import (
    EVENT,
    INTEG,
    ORG,
    FakeMemoryRepo,
    calendar_view,
    drive_view,
    gmail_view,
)


def test_build_is_deterministic():
    a = build_memory_records(gmail_view())
    b = build_memory_records(gmail_view())
    assert [r.identity_tuple() for r in a] == [r.identity_tuple() for r in b]
    # Same memory_ids both times.
    assert {r.memory_id for r in a} == {r.memory_id for r in b}


def test_every_record_has_full_provenance():
    for view in (gmail_view(), calendar_view(), drive_view()):
        for r in build_memory_records(view):
            assert r.org_id == ORG and r.org_id is not None  # tenant mandatory
            assert r.source_signal_event_id == EVENT          # traces to signal
            assert r.source_integration_id == INTEG           # traces to integration
            assert r.extractor_name and r.extractor_version   # traces to extractor
            assert r.schema_version >= 1                      # versioned
            assert 0.0 <= r.confidence <= 1.0
            assert r.event_timestamp is not None
            assert r.created_at is None  # set by DB, not identity


_FORBIDDEN = {"health", "risk", "priority", "score", "scores", "recommendation",
              "urgency", "sentiment", "intelligence"}


def test_memory_contains_no_intelligence():
    for view in (gmail_view(), calendar_view(), drive_view()):
        for r in build_memory_records(view):
            keys = {k.lower() for k in r.payload}
            leaked = keys & _FORBIDDEN
            assert not leaked, f"{r.entity_type} leaked intelligence keys: {leaked}"


async def test_append_is_idempotent_under_replay():
    """Build → append twice (re-delivery / replay). Second pass writes nothing new."""
    repo = FakeMemoryRepo()
    records = build_memory_records(gmail_view())

    first = sum([await repo.append(r) for r in records])
    second = sum([await repo.append(r) for r in build_memory_records(gmail_view())])

    assert first == len(records) and first > 0
    assert second == 0  # idempotent: replay produced identical ids → all no-ops


async def test_ingestion_via_injected_signal_getter():
    """Reconnect path: service fetches a SignalView by event_id and appends facts."""
    repo = FakeMemoryRepo()
    view = gmail_view()

    async def getter(event_id):
        return view if event_id == view.event_id else None

    svc = MemoryIngestionService(db=None, memory_repo=repo, signal_getter=getter)
    written = await svc.ingest_signal(view.event_id)
    assert written > 0
    assert all(isinstance(r, MemoryRecord) for r in repo.records)
    # Re-ingest is idempotent.
    assert await svc.ingest_signal(view.event_id) == 0


# ── the replay-visibility trap ─────────────────────────────────────────


def test_new_payload_keys_require_a_new_extractor_name():
    """A version bump ALONE cannot surface new payload keys on replayed orgs.

    `memory_record_id` deliberately excludes `extractor_version`, so bumping only
    the version re-derives the SAME memory_id, hits ON CONFLICT DO NOTHING, and
    the new keys never reach the database. Measured on a live workspace before
    this was understood: 54/54 thread records skipped, 0 written, `direction`
    absent from every one.

    This test is the tripwire. If a future change adds a key to the thread
    payload, it has to move the NAME too — which is what the EVOLUTION LAW in
    memory.ids prescribes and what makes a replay actually converge.
    """
    records = {r.extractor_name: r for r in build_memory_records(gmail_view())}
    thread = records.get("email_thread_v2")
    assert thread is not None, (
        "thread extractor name changed — if the payload gained keys this is "
        "correct, but update this test; if it did NOT, reverting the name "
        "silently strands the new keys on every already-ingested org"
    )
    # The keys the rename exists to deliver.
    assert "direction" in thread.payload
    assert thread.payload["direction"] in {"inbound", "outbound"}


def test_renamed_extractor_does_not_fork_the_thread_entity():
    """The rename moves the RECORD id, never the ENTITY id.

    v1 and v2 records must fold onto the same thread entity, otherwise the org
    would grow a second, half-populated thread for every real one.
    """
    from navis.memory.ids import entity_id as entity_id_of

    thread = next(
        r for r in build_memory_records(gmail_view())
        if r.extractor_name == "email_thread_v2"
    )
    # entity_id is derived from (org, entity_type, natural_key) — no extractor
    # name — so a v1 record for the same thread lands on this exact entity.
    assert thread.entity_id == entity_id_of(
        ORG, thread.entity_type.value, str(thread.payload["thread_id"])
    )
