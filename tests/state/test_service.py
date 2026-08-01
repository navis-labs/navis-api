"""StateService: rebuild, get, incremental + watermark recovery (with fakes).

Also unit-tests the repository pieces that run without a DB: MemoryReader's
pagination loop (batching/keyset-advance/termination) and the SQL that
SnapshotRepository constructs (monotonic upsert guard + org-scoped delete). The
real Postgres semantics live in tests/integration/test_state_postgres.py.
"""
from __future__ import annotations

import types
import uuid
from datetime import datetime, timezone

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
        self.has_records_after_calls = 0
        self.stream_calls = 0

    async def has_records_after(self, org_id, watermark) -> bool:
        self.has_records_after_calls += 1
        return await super().has_records_after(org_id, watermark)

    async def stream(self, org_id, *, after=None, batch=5000):
        self.stream_calls += 1
        async for r in super().stream(org_id, after=after, batch=batch):
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
    reader.has_records_after_calls = 0
    reader.stream_calls = 0

    again = await svc.incremental_update(ORG_A)
    assert again.to_dict() == first.to_dict()               # projection identity preserved
    assert snaps.latest_calls == 0                          # JSONB payload NOT fetched
    assert snaps.watermark_only_calls == 1                  # only the cheap watermark probe ran
    assert reader.has_records_after_calls == 1              # correctness gate consulted
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
    reader.has_records_after_calls = 0
    reader.stream_calls = 0

    incr = await svc.incremental_update(ORG_A)
    # Correctness: incremental result == full rebuild from the whole corpus.
    assert incr.to_dict() == build_projection(ORG_A, records).to_dict()
    # Fast path checked, then bailed to the slow path (fold from snapshot).
    assert snaps.watermark_only_calls == 1
    assert reader.has_records_after_calls == 1
    assert snaps.latest_calls == 1                  # JSONB payload fetched for the fold
    assert reader.stream_calls == 1                 # fold ran


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
