"""StateService: rebuild, get, incremental + watermark recovery (with fakes).

Also unit-tests the repository pieces that run without a DB: MemoryReader's
pagination loop (batching/keyset-advance/termination) and the SQL that
SnapshotRepository constructs (monotonic upsert guard + org-scoped delete). The
real Postgres semantics live in tests/integration/test_state_postgres.py.
"""
from __future__ import annotations

import types
import uuid
from datetime import datetime, timedelta, timezone

from navis.state.v2.builders import build_projection
from navis.state.v2.models import PROJECTION_VERSION, STATE_SCHEMA_VERSION, order_key
from navis.state.v2.repository import MemoryReader, SnapshotRepository
from navis.state.v2.service import StateService
from tests.state.fakes import ORG_A, FakeReader, FakeSnapshots, corpus


def _svc(records):
    reader = FakeReader(records)
    snaps = FakeSnapshots()
    return StateService(reader=reader, snapshots=snaps), reader, snaps


# --- repository: MemoryReader pagination loop (no DB) ------------------------

class _Result:
    def __init__(self, rows): self._rows = rows
    def all(self): return self._rows


class _StreamSession:
    """Returns successive canned pages for each scalars() call (keyset SQL is exercised
    by the Postgres integration test; here we test the loop's batching/termination)."""
    def __init__(self, pages): self.pages = list(pages); self.calls = 0
    async def scalars(self, _stmt):
        page = self.pages[self.calls] if self.calls < len(self.pages) else []
        self.calls += 1
        return _Result(page)


def _row(i):
    return types.SimpleNamespace(
        memory_id=uuid.uuid5(uuid.NAMESPACE_DNS, f"m{i}"),
        organization_id=ORG_A, entity_type="person",
        entity_id=uuid.uuid5(uuid.NAMESPACE_DNS, f"e{i}"),
        event_timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
        payload={"i": i}, source_signal_event_id=uuid.uuid5(uuid.NAMESPACE_DNS, f"s{i}"),
        schema_version=1, confidence=0.9,
        # The observation clock the reader projects onto MemoryRecordView.observed_at.
        created_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )


async def test_stream_paginates_in_order_and_terminates():
    # page1 == batch (continue), page2 < batch (stop). All rows, in order, no duplicates.
    reader = MemoryReader(_StreamSession([[_row(0), _row(1), _row(2)], [_row(3), _row(4)]]))  # type: ignore[arg-type]
    out = [v async for v in reader.stream(ORG_A, batch=3)]
    assert [v.payload["i"] for v in out] == [0, 1, 2, 3, 4]


async def test_stream_stops_on_empty_first_page():
    reader = MemoryReader(_StreamSession([[]]))  # type: ignore[arg-type]
    out = [v async for v in reader.stream(ORG_A, batch=3)]
    assert out == []


# --- repository: SnapshotRepository SQL construction (no DB) -----------------

class _RecSession:
    def __init__(self): self.stmts = []
    async def execute(self, stmt): self.stmts.append(stmt)


def _sql(stmt):
    from sqlalchemy.dialects import postgresql
    return str(stmt.compile(dialect=postgresql.dialect())).upper()


async def test_snapshot_save_emits_monotonic_conflict_guard():
    sess = _RecSession()
    await SnapshotRepository(sess).save(build_projection(ORG_A, corpus()))  # type: ignore[arg-type]
    sql = _sql(sess.stmts[0])
    assert "ON CONFLICT" in sql and "DO UPDATE" in sql
    assert "WHERE" in sql  # the stale-snapshot (monotonic watermark) guard


async def test_snapshot_delete_is_org_scoped():
    sess = _RecSession()
    repo = SnapshotRepository(sess)  # type: ignore[arg-type]
    await repo.delete(ORG_A)
    await repo.delete(ORG_A, projection_version=1)
    all_versions = _sql(sess.stmts[0])
    one_version = _sql(sess.stmts[1])
    assert all_versions.startswith("DELETE FROM STATE_SNAPSHOTS") and "ORGANIZATION_ID" in all_versions
    assert "PROJECTION_VERSION" not in all_versions          # None → drop all versions
    assert "PROJECTION_VERSION" in one_version                # value → scoped to one version


# --- snapshots.py: cache lifecycle (version-scoping, delete -> rebuild) ------

async def test_snapshot_lookup_is_version_scoped():
    _, _, snaps = _svc([])
    state = build_projection(ORG_A, corpus())
    await snaps.save(state)
    assert await snaps.latest(ORG_A, projection_version=state.projection_version) is not None
    assert await snaps.latest(ORG_A, projection_version=state.projection_version + 1) is None  # version mismatch


async def test_rebuild_after_delete_disposal():
    svc, _, snaps = _svc(corpus())
    await svc.rebuild_org(ORG_A)
    assert snaps.store                       # snapshot cached
    snaps.store.clear()                      # simulate SnapshotRepository.delete()/disposal
    state = await svc.get_state(ORG_A)        # must rebuild deterministically from Memory
    assert state.counts()["person"] >= 1
    assert snaps.store                        # re-cached after rebuild


async def test_rebuild_then_get_round_trips():
    svc, _, _ = _svc(corpus())
    rebuilt = await svc.rebuild_org(ORG_A)
    fetched = await svc.get_state(ORG_A)
    assert fetched.to_dict() == rebuilt.to_dict()  # snapshot serialization is lossless


async def test_get_state_rebuilds_when_no_snapshot():
    svc, _, _ = _svc(corpus())
    state = await svc.get_state(ORG_A)
    assert state.counts()["person"] >= 1


async def test_incremental_matches_full_via_service():
    records = sorted(corpus(), key=lambda r: order_key(r.event_timestamp, r.memory_id))
    half = len(records) // 2

    # Reader starts with the first half; build a snapshot.
    svc, reader, _ = _svc(records[:half])
    await svc.rebuild_org(ORG_A)

    # New memory arrives; incremental folds only the records after the watermark.
    reader.add(records[half:])
    incr = await svc.incremental_update(ORG_A)

    full = build_projection(ORG_A, records)
    assert incr.to_dict() == full.to_dict()


async def test_incremental_noop_when_no_new_records():
    svc, _, _ = _svc(corpus())
    first = await svc.rebuild_org(ORG_A)
    again = await svc.incremental_update(ORG_A)
    assert again.to_dict() == first.to_dict()  # nothing past the watermark → unchanged


# --- service hardening: streaming, corruption recovery, version drift, lock --

class _RawSnap:
    """A snapshot row with an arbitrary (possibly corrupt) full_projection."""
    def __init__(self, full_projection): self.full_projection = full_projection; self.organization_id = ORG_A


def _inject(snaps, full_projection):
    snaps.store[(ORG_A, PROJECTION_VERSION)] = _RawSnap(full_projection)


class _LockDenyDB:
    """A db whose advisory-lock attempt always fails — simulates a concurrent rebuild."""
    async def execute(self, _stmt, _params=None):
        return types.SimpleNamespace(scalar=lambda: False)


async def test_streaming_rebuild_equals_build_projection():
    # The streamed fold must be byte-identical to the batch build_projection.
    svc, _, _ = _svc(corpus())
    streamed = await svc.rebuild_org(ORG_A)
    assert streamed.to_dict() == build_projection(ORG_A, corpus()).to_dict()


async def test_get_state_disposes_and_rebuilds_on_corrupt_snapshot():
    svc, _, snaps = _svc(corpus())
    _inject(snaps, {"garbage": True})              # missing org_id → from_dict raises
    state = await svc.get_state(ORG_A)
    assert state.counts()["person"] >= 1            # rebuilt from Memory
    assert state.to_dict() == build_projection(ORG_A, corpus()).to_dict()  # deterministic
    snap = await snaps.latest(ORG_A, projection_version=PROJECTION_VERSION)
    assert snap is not None and snap.full_projection == state.to_dict()    # corrupt row replaced


async def test_get_state_rebuilds_on_schema_mismatch():
    svc, _, snaps = _svc(corpus())
    d = build_projection(ORG_A, corpus()).to_dict(); d["schema_version"] = 999
    _inject(snaps, d)
    state = await svc.get_state(ORG_A)
    assert state.schema_version == STATE_SCHEMA_VERSION   # rebuilt under current schema


async def test_snapshot_with_a_future_cursor_is_disposed_not_trusted():
    """A stored observation cursor past wall clock is impossible by construction —
    `created_at` is DB-assigned — so it means the row was written by something that
    did not mean observation time by it. Production held exactly such a row.

    Trusting it is unrecoverable rather than merely stale: the delta past a future
    cursor is always empty, so State would look "up to date", and the monotonic
    upsert guard would then refuse every corrected write for being older. So it is
    rejected on READ, where one rebuild fixes it permanently.
    """
    from navis.state.v2.service import _cache_clear

    _cache_clear()
    svc, _, snaps = _svc(corpus())
    poisoned = build_projection(ORG_A, corpus()).to_dict()
    future = datetime.now(timezone.utc) + timedelta(days=1)
    poisoned["watermark"]["last_timestamp"] = future.isoformat()
    poisoned["as_of"] = future.isoformat()
    _inject(snaps, poisoned)

    state = await svc.get_state(ORG_A)

    assert state.as_of is not None and state.as_of <= datetime.now(timezone.utc)
    assert state.to_dict() == build_projection(ORG_A, corpus()).to_dict()
    # The bad row is gone, so the rebuild's save lands in an empty slot rather
    # than losing a monotonic-guard comparison against a future timestamp.
    snap = await snaps.latest(ORG_A, projection_version=PROJECTION_VERSION)
    assert snap is not None and snap.full_projection == state.to_dict()


async def test_get_state_rebuilds_on_projection_version_mismatch():
    svc, _, snaps = _svc(corpus())
    d = build_projection(ORG_A, corpus()).to_dict(); d["projection_version"] = 999
    _inject(snaps, d)
    state = await svc.get_state(ORG_A)
    assert state.projection_version == PROJECTION_VERSION


async def test_replay_after_corruption_is_deterministic():
    fresh = build_projection(ORG_A, corpus())
    svc, _, snaps = _svc(corpus())
    _inject(snaps, {"corrupt": 1})
    recovered = await svc.get_state(ORG_A)
    assert recovered.to_dict() == fresh.to_dict()


async def test_incremental_equals_full_after_recovery():
    records = sorted(corpus(), key=lambda r: order_key(r.event_timestamp, r.memory_id))
    half = len(records) // 2
    svc, reader, snaps = _svc(records[:half])
    await svc.rebuild_org(ORG_A)
    _inject(snaps, {"corrupt": 1})          # cache corrupts
    reader.add(records[half:])              # and new memory arrives
    recovered = await svc.incremental_update(ORG_A)  # corrupt → dispose + full rebuild
    assert recovered.to_dict() == build_projection(ORG_A, records).to_dict()


async def test_a_forced_rebuild_never_serves_the_snapshot_it_rejected():
    """Lock contention must not undo a correctness decision.

    Every incremental bail-out rebuilds precisely BECAUSE the stored snapshot was
    found incomplete or out of order. If losing the advisory lock then served that
    same row back, "this projection is missing records" would be answered with the
    projection missing records — silent staleness reintroduced through the back
    door. A duplicated fold is the cheaper mistake.
    """
    from navis.state.v2.service import _cache_clear

    _cache_clear()
    snaps = FakeSnapshots()
    # The stored snapshot knows about ONE record; the org actually has a full corpus.
    stale = build_projection(ORG_A, corpus()[:1])
    await snaps.save(stale)
    svc = StateService(db=_LockDenyDB(), reader=FakeReader(corpus()), snapshots=snaps)  # type: ignore[arg-type]

    contended = await svc.rebuild_org(ORG_A)
    assert contended.to_dict() == stale.to_dict()          # unforced: dedupe wins

    forced = await svc.rebuild_org(ORG_A, force=True)
    assert forced.record_count == len(corpus())
    assert forced.to_dict() == build_projection(ORG_A, corpus()).to_dict()


async def test_an_incomplete_snapshot_is_rebuilt_even_while_another_fold_holds_the_lock():
    """The same guarantee, reached the way production reaches it: through
    `incremental_update` detecting a record count that does not add up."""
    from navis.state.v2.service import _cache_clear

    _cache_clear()
    snaps = FakeSnapshots()
    await snaps.save(build_projection(ORG_A, corpus()[:1]))   # under-counts the org
    svc = StateService(db=_LockDenyDB(), reader=FakeReader(corpus()), snapshots=snaps)  # type: ignore[arg-type]

    state = await svc.incremental_update(ORG_A)

    assert state.record_count == len(corpus())
    assert state.to_dict() == build_projection(ORG_A, corpus()).to_dict()


async def test_concurrent_rebuild_contention_serves_existing_snapshot():
    snaps = FakeSnapshots()
    cached = build_projection(ORG_A, corpus())
    await snaps.save(cached)
    # Reader is EMPTY: if the lock were ignored and it rebuilt, the result would be empty.
    svc = StateService(db=_LockDenyDB(), reader=FakeReader([]), snapshots=snaps)  # type: ignore[arg-type]
    out = await svc.rebuild_org(ORG_A)
    assert out.to_dict() == cached.to_dict()  # served the snapshot, did not rebuild from empty


# --- in-process warm cache: fast path bypasses deserialize + fold ------------

class _CountingSnaps(FakeSnapshots):
    """Snapshot store that records which endpoint was hit — used to prove the
    warm-cache fast path skips `latest()` (the JSONB payload fetch) and only
    calls the lightweight `watermark_only()`."""
    def __init__(self):
        super().__init__()
        self.latest_calls = 0
        self.watermark_only_calls = 0

    async def latest(self, org_id, *, projection_version):
        self.latest_calls += 1
        return await super().latest(org_id, projection_version=projection_version)

    async def watermark_only(self, org_id, *, projection_version):
        self.watermark_only_calls += 1
        return await super().watermark_only(org_id, projection_version=projection_version)


class _CountingReader(FakeReader):
    def __init__(self, records=None):
        super().__init__(records)
        self.count_calls = 0
        self.observed_since_calls = 0
        self.stream_calls = 0

    async def count(self, org_id) -> int:
        self.count_calls += 1
        return await super().count(org_id)

    async def observed_since(self, org_id, cursor, *, limit):
        self.observed_since_calls += 1
        return await super().observed_since(org_id, cursor, limit=limit)

    async def stream(self, org_id, *, batch=5000):
        self.stream_calls += 1
        async for r in super().stream(org_id, batch=batch):
            yield r


async def test_incremental_warm_cache_hits_skip_snapshot_payload_fetch():
    from navis.state.v2.service import _cache_clear

    _cache_clear()
    reader = _CountingReader(corpus())
    snaps = _CountingSnaps()
    svc = StateService(reader=reader, snapshots=snaps)  # type: ignore[arg-type]

    # First call — populates snapshot AND the in-process cache.
    first = await svc.rebuild_org(ORG_A)
    assert snaps.latest_calls == 0  # rebuild bypasses latest() by design

    # Reset counters and take the fast path.
    snaps.latest_calls = 0
    snaps.watermark_only_calls = 0
    reader.count_calls = 0
    reader.observed_since_calls = 0
    reader.stream_calls = 0

    again = await svc.incremental_update(ORG_A)
    assert again.to_dict() == first.to_dict()               # projection identity preserved
    assert snaps.latest_calls == 0                          # JSONB payload NOT fetched
    assert snaps.watermark_only_calls == 1                  # only the cheap watermark probe ran
    assert reader.count_calls == 1                          # completeness gate consulted
    assert reader.observed_since_calls == 0                 # no delta read
    assert reader.stream_calls == 0                         # no fold at all


async def test_incremental_falls_back_to_slow_path_when_new_records_arrive():
    from navis.state.v2.service import _cache_clear

    _cache_clear()
    records = sorted(corpus(), key=lambda r: order_key(r.event_timestamp, r.memory_id))
    half = len(records) // 2
    reader = _CountingReader(records[:half])
    snaps = _CountingSnaps()
    svc = StateService(reader=reader, snapshots=snaps)  # type: ignore[arg-type]

    await svc.rebuild_org(ORG_A)                    # warm cache
    reader.add(records[half:])                      # new memory lands

    snaps.latest_calls = 0
    snaps.watermark_only_calls = 0
    reader.count_calls = 0
    reader.observed_since_calls = 0
    reader.stream_calls = 0

    incr = await svc.incremental_update(ORG_A)
    # Correctness: incremental result == full rebuild from the whole corpus.
    assert incr.to_dict() == build_projection(ORG_A, records).to_dict()
    # Fast path checked, then bailed to the slow path (fold from snapshot).
    assert snaps.watermark_only_calls == 1
    assert reader.count_calls == 1
    assert snaps.latest_calls == 1                  # JSONB payload fetched for the fold
    assert reader.observed_since_calls == 1         # delta read in observation order
    assert reader.stream_calls == 0                 # incremental never re-streams the org


async def test_dispose_evicts_the_warm_cache():
    from navis.state.v2.service import _cache_clear, _STATE_CACHE

    _cache_clear()
    svc, _, snaps = _svc(corpus())
    await svc.rebuild_org(ORG_A)
    assert any(k[0] == ORG_A for k in _STATE_CACHE)     # populated by rebuild

    # Force a dispose via the corrupt-snapshot recovery path.
    _inject(snaps, {"corrupt": 1})
    await svc.get_state(ORG_A)                          # triggers dispose + rebuild
    # After dispose+rebuild there IS a fresh entry, but the OLD key (pre-dispose)
    # is gone. The invariant we assert is looser: no dangling entry with a
    # stale watermark_memory_id lingers. Since the rebuild produced identical
    # bytes, cheapest post-condition check is that only the "current" key exists.
    cur = build_projection(ORG_A, corpus())
    expected_key = (ORG_A, cur.projection_version, cur.watermark.last_memory_id)
    org_a_keys = [k for k in _STATE_CACHE if k[0] == ORG_A]
    assert org_a_keys == [expected_key]


# ── THE TWO CLOCKS ───────────────────────────────────────────────────────────
#
# Event time says when a business fact HAPPENED and drives the fold order.
# Observation time says when Navis LEARNED it and drives the resume cursor and
# `as_of`. Conflating them is what let one meeting booked for tomorrow become an
# organization's clock, push the cursor past wall clock, and strand every record
# observed afterwards — 25 of them in production, permanently unfoldable because
# an event-ordered cursor only ever moves forward.
#
# These tests fix the boundary between the two. They use REAL wall clock rather
# than the corpus's fixed BASE, because the invariant under test is literally
# "the projection's clock cannot be in the future".

_MEETING_ENTITY = "evt-future"


def _counter(key: str) -> float:
    """One counter out of the metrics registry, 0.0 when never incremented.

    Read rather than mocked: `state.rebuild.fallback{reason=...}` is the signal an
    operator actually watches, so a test that asserts on it is testing the thing
    production reads.
    """
    from navis.platform.monitoring.metrics import metrics

    return metrics.snapshot()["counters"].get(key, 0.0)


def _clock_rec(
    entity_type, natural_key, payload, *, happened_at, observed_at, tag,
    org_id=ORG_A,
):
    """One record with the two clocks set INDEPENDENTLY — the whole point."""
    from navis.state.v2.builders import MemoryRecordView

    return MemoryRecordView(
        memory_id=uuid.uuid5(uuid.NAMESPACE_DNS, f"clock:{org_id}:{tag}"),
        org_id=org_id,
        entity_type=entity_type.value,
        entity_id=uuid.uuid5(uuid.NAMESPACE_DNS, f"ent:{org_id}:{entity_type.value}:{natural_key}"),
        event_timestamp=happened_at,
        payload=payload,
        source_signal_event_id=uuid.uuid5(uuid.NAMESPACE_DNS, f"sig:{org_id}:{tag}"),
        schema_version=1,
        confidence=0.9,
        observed_at=observed_at,
    )


def _future_meeting(now, *, observed_at=None):
    """A calendar meeting starting tomorrow, learned about now.

    `event_timestamp` is the meeting's START — that is not a test contrivance,
    it is what `signals/calendar/connector.py::_to_event` deliberately stamps,
    and the reason this whole class of defect exists.
    """
    from navis.memory.envelope import EntityType

    starts = now + timedelta(days=1)
    return _clock_rec(
        EntityType.MEETING, _MEETING_ENTITY,
        {"title": "Board", "status": "confirmed", "starts_at": starts.isoformat(),
         "attendees": ["alice@acme.com"]},
        happened_at=starts,
        observed_at=observed_at or now,
        tag="meeting-future",
    )


def _email_thread(*, happened_at, observed_at, tag, subject="terms"):
    from navis.memory.envelope import EntityType

    return _clock_rec(
        EntityType.THREAD, "thread-9",
        {"thread_id": "thread-9", "subject": subject, "message_id": tag,
         "direction": "inbound"},
        happened_at=happened_at, observed_at=observed_at, tag=tag,
    )


async def test_future_meeting_does_not_become_the_organizations_clock():
    """TEST 1. A meeting that has not happened cannot move the projection's clock.

    The meeting is still fully present — `starts_at` is a future datetime and must
    stay one, because "what is on my calendar tomorrow" is a real question. What
    it may NOT do is answer "when did Navis last look", which is what `as_of` and
    the resume cursor mean.
    """
    from navis.state.v2.service import _cache_clear

    _cache_clear()
    now = datetime.now(timezone.utc)
    meeting = _future_meeting(now)
    svc, _, _ = _svc([meeting])

    state = await svc.rebuild_org(ORG_A)

    assert state.as_of is not None
    assert state.as_of <= now, "as_of ran past the observation horizon"
    assert state.watermark.last_timestamp is not None
    assert state.watermark.last_timestamp <= now, "cursor ran past the observation horizon"
    assert state.watermark.last_timestamp == meeting.observed_at

    # ...and the meeting is still a future meeting.
    (m,) = list(state.meetings())
    assert m.fields["starts_at"] == (now + timedelta(days=1)).isoformat()
    assert m.last_seen == (now + timedelta(days=1)).isoformat()  # event time, preserved
    # The event frontier DOES hold the future position — that is its job. It is
    # an ordering fact about the fold, never a clock anyone reads.
    assert state.fold_frontier.last_event_timestamp == now + timedelta(days=1)


async def test_email_observed_after_a_future_meeting_is_still_folded():
    """TEST 2. THE production regression, exactly.

    Under the old event-time cursor the meeting pushed the watermark to tomorrow,
    every later email sorted below it, `stream(after=watermark)` returned nothing,
    and the record was silently never folded. Twenty-five rows died this way.
    """
    from navis.state.v2.service import _cache_clear

    _cache_clear()
    now = datetime.now(timezone.utc)
    meeting = _future_meeting(now)
    svc, reader, _ = _svc([meeting])
    await svc.rebuild_org(ORG_A)

    # An email that happened an hour ago, observed a minute ago — i.e. AFTER the
    # meeting was learned about, but BEFORE the meeting will happen.
    email = _email_thread(
        happened_at=now - timedelta(hours=1),
        observed_at=now + timedelta(minutes=1),
        tag="m-late",
    )
    reader.add([email])

    state = await svc.incremental_update(ORG_A)

    assert state.record_count == 2, "the email was skipped — the P0 regression"
    assert [t.entity_id for t in state.threads()], "no thread entity was folded"
    assert state.as_of is not None and state.as_of <= now + timedelta(minutes=1)
    # Equivalence with a clean rebuild is the real assertion: whichever path ran,
    # the answer must be the one a from-scratch fold would give.
    assert state.to_dict() == (await svc.rebuild_org(ORG_A)).to_dict()


async def test_out_of_order_arrival_triggers_rebuild_and_is_never_skipped():
    """TEST 3. A record that HAPPENED before something already folded cannot be
    folded on top of it — that would clobber a newer fact and diverge from a full
    rebuild. It is detected and the whole projection is re-derived."""
    from navis.state.v2.service import _cache_clear

    _cache_clear()
    now = datetime.now(timezone.utc)
    reader = _CountingReader([
        _email_thread(happened_at=now - timedelta(hours=1),
                      observed_at=now - timedelta(hours=1), tag="m-new",
                      subject="newest"),
    ])
    snaps = FakeSnapshots()
    svc = StateService(reader=reader, snapshots=snaps)  # type: ignore[arg-type]
    await svc.rebuild_org(ORG_A)

    # A backfill delivers an OLDER message on the same thread, observed now.
    late = _email_thread(happened_at=now - timedelta(days=3),
                         observed_at=now, tag="m-old", subject="oldest")
    reader.add([late])

    reader.stream_calls = 0
    before = _counter("state.rebuild.fallback{reason=out_of_order}")
    state = await svc.incremental_update(ORG_A)

    assert reader.stream_calls == 1, "no rebuild ran — the fold was applied out of order"
    # ...and it rebuilt for THIS reason, not incidentally. Without the assertion a
    # rebuild triggered by an unrelated guard would let the test pass vacuously.
    assert _counter("state.rebuild.fallback{reason=out_of_order}") == before + 1
    assert state.record_count == 2, "the late record was skipped"
    # Correct answer, not merely a non-empty one: the NEWER message still wins the
    # thread's latest-non-null fields, which only a full re-fold in event order gives.
    (thread,) = list(state.threads())
    assert thread.subject == "newest"
    assert state.to_dict() == build_projection(
        ORG_A, reader._records
    ).to_dict()


async def test_replayed_records_leave_the_projection_identical():
    """TEST 4. At-least-once delivery is the norm, not the exception. Re-ingesting
    the same facts must converge, never accumulate."""
    from navis.state.v2.service import _cache_clear

    _cache_clear()
    svc, reader, _ = _svc(corpus())
    first = await svc.rebuild_org(ORG_A)

    reader.add(corpus())                       # the whole batch, redelivered
    replayed = await svc.incremental_update(ORG_A)
    assert replayed.to_dict() == first.to_dict()
    assert replayed.record_count == len(corpus())

    # And a second rebuild over the same rows is byte-identical too.
    assert (await svc.rebuild_org(ORG_A)).to_dict() == first.to_dict()


async def test_incremental_equals_rebuild_over_the_complete_projection():
    """TEST 5. The whole deterministic projection, not just record counts."""
    from navis.state.v2.service import _cache_clear

    _cache_clear()
    now = datetime.now(timezone.utc)
    early = list(corpus())
    svc, reader, _ = _svc(early)
    await svc.rebuild_org(ORG_A)

    # Records that both HAPPENED and were OBSERVED after everything so far.
    later = [
        _email_thread(happened_at=now - timedelta(minutes=30),
                      observed_at=now - timedelta(minutes=20), tag="m-a"),
        _future_meeting(now, observed_at=now - timedelta(minutes=10)),
    ]
    reader.add(later)

    incr = await svc.incremental_update(ORG_A)
    full = build_projection(ORG_A, early + later)

    assert incr.entities == full.entities
    assert incr.watermark.to_dict() == full.watermark.to_dict()
    assert incr.fold_frontier.to_dict() == full.fold_frontier.to_dict()
    assert incr.record_count == full.record_count == len(early) + len(later)
    assert incr.to_dict() == full.to_dict()


async def test_adversarial_arrival_order_converges_on_the_clean_rebuild():
    """TEST 6. Deliberately hostile ordering: a future meeting first, then older
    facts, then facts between them — delivered in several batches whose
    observation order is scrambled against their event order."""
    from navis.state.v2.service import _cache_clear
    from navis.memory.envelope import EntityType

    _cache_clear()
    now = datetime.now(timezone.utc)

    batches = [
        [_future_meeting(now, observed_at=now - timedelta(minutes=50))],
        [
            _email_thread(happened_at=now - timedelta(days=4),
                          observed_at=now - timedelta(minutes=40), tag="m-1",
                          subject="oldest"),
            _email_thread(happened_at=now - timedelta(days=1),
                          observed_at=now - timedelta(minutes=40), tag="m-3",
                          subject="newest"),
        ],
        [
            _email_thread(happened_at=now - timedelta(days=2),
                          observed_at=now - timedelta(minutes=30), tag="m-2",
                          subject="middle"),
        ],
        [
            _clock_rec(
                EntityType.PERSON, "alice@acme.com", {"email": "alice@acme.com", "role": "from"},
                happened_at=now - timedelta(days=10),
                observed_at=now - timedelta(minutes=5), tag="p-late",
            ),
        ],
    ]

    svc, reader, _ = _svc([])
    for batch in batches:
        reader.add(batch)
        await svc.incremental_update(ORG_A)

    final = await svc.get_state(ORG_A)
    everything = [r for batch in batches for r in batch]

    assert final.to_dict() == build_projection(ORG_A, everything).to_dict()
    assert final.record_count == len(everything)
    # Event order still decides content: the newest message owns the thread.
    (thread,) = list(final.threads())
    assert thread.subject == "newest"
    # Observation order still decides the clock, and it never left reality.
    assert final.as_of is not None and final.as_of <= now
    assert final.watermark.last_timestamp == now - timedelta(minutes=5)


async def test_cache_is_bounded_lru():
    from navis.state.v2.service import _cache_clear, _cache_put, _STATE_CACHE, _STATE_CACHE_MAX

    _cache_clear()
    base_state = build_projection(ORG_A, corpus())
    # Synthesize N > cap distinct cache entries by varying watermark_memory_id.
    from dataclasses import replace
    from navis.state.v2.models import Watermark

    for i in range(_STATE_CACHE_MAX + 8):
        wm = Watermark(
            last_timestamp=base_state.watermark.last_timestamp,
            last_memory_id=uuid.uuid5(uuid.NAMESPACE_DNS, f"fake-wm-{i}"),
        )
        variant = replace(base_state, watermark=wm)
        _cache_put(variant)
    assert len(_STATE_CACHE) == _STATE_CACHE_MAX
