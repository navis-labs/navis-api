"""StateService — the SOLE interface to current truth.

Downstream (Reasoning, Decision, Execution, Learning, Intelligence) read
OrganizationalState from here and never query Memory directly.

  rebuild_org        — full deterministic rebuild, STREAMED from memory_records
  incremental_update — fold only memory_records after the snapshot watermark
  get_state          — return the cached projection (rebuild if absent/invalid)

Production posture (all hardening below preserves replay determinism — it only
changes WHEN we rebuild and HOW we read, never WHAT the projection computes):
  * Streaming rebuild: records are folded one-at-a-time straight off the reader's
    deterministic (event_timestamp, memory_id) stream — never materialized into a
    list — so a 10M-record org costs O(entities) memory, not O(records). Folding the
    already-ordered stream is byte-identical to build_projection (which only sorts an
    already-sorted sequence).
  * Snapshot validation + auto-dispose: a cache that is absent, version-drifted, or
    corrupt degrades gracefully to a deterministic rebuild from Memory; the bad row is
    disposed so the rebuild's (monotonic) save can replace it.
  * Per-org advisory lock: a concurrent full rebuild is wasted O(org) work — a loser
    serves the freshest snapshot instead. Non-blocking + transaction-scoped (PgBouncer
    -safe). Never affects correctness, only dedups work.
  * Structured metrics + logs on every path.

State reads ONLY memory_records. No now() leaks into the projection (as_of is
derived from Memory, or passed explicitly for point-in-time queries).
"""
from __future__ import annotations

import hashlib
import uuid
from collections import OrderedDict
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.state.v2.builders import StateBuilder
from navis.state.v2.models import (
    PROJECTION_VERSION,
    STATE_SCHEMA_VERSION,
    OrganizationalState,
)
from navis.state.v2.repository import MemoryReader, SnapshotRepository

logger = get_logger("state.v2")

# Advisory-lock space for State — disjoint from signals/memory/execution (blake2b
# personalization), so a State per-org lock never collides with another domain's lock.
_LOCK_NAMESPACE = b"navis.state"

# ── In-process cache of deserialized OrganizationalState ────────────────────
# Keyed by (org_id, projection_version, watermark_memory_id) so that a
# snapshot whose watermark hasn't moved since the last request short-circuits
# both the JSONB payload fetch and the Python-side ~O(entities) re-parse of
# `full_projection`. Those two steps are the dominant cost on the /context hot
# path — measured p50 ~12s on a warm org before this cache; the same call
# returns in <100ms once the deserialized state is in-process.
#
# Correctness (this cache never returns a stale projection):
#   * Any new memory record past the current watermark advances
#     state.watermark.last_memory_id, which changes the cache key. The old
#     entry becomes unreachable and eventually LRU-evicts.
#   * Every call still verifies via MemoryReader.has_records_after() that
#     nothing new has landed since the cached watermark — a cheap EXISTS
#     query. When new records exist, the fast path falls through to the
#     original fold-from-snapshot flow.
#   * Snapshot disposal (corruption / schema drift) evicts the org from the
#     cache too, so the next call rebuilds from Memory.
#   * `projection_version` is part of the key: a bump automatically misses
#     any in-flight caches without touching this file.
#
# Bounded LRU (32 orgs — a single-tenant / small-team deployment fits in
# under 10 MB assuming ~200 KB per deserialized projection). Uvicorn workers
# each own their own cache; concurrent access within an event loop is safe
# because dict ops don't await.
_STATE_CACHE_MAX = 32
_STATE_CACHE: "OrderedDict[tuple[uuid.UUID, int, uuid.UUID], OrganizationalState]" = OrderedDict()


def _org_lock_key(org_id: uuid.UUID) -> int:
    """Deterministic signed-bigint advisory-lock key for an org (full-uuid blake2b,
    State-namespaced). Stable across processes; ~2^-64 collision probability."""
    digest = hashlib.blake2b(org_id.bytes, digest_size=8, person=_LOCK_NAMESPACE).digest()
    return int.from_bytes(digest, "big", signed=True)


def _cache_key(
    org_id: uuid.UUID,
    projection_version: int,
    watermark_memory_id: uuid.UUID | None,
) -> tuple[uuid.UUID, int, uuid.UUID] | None:
    """Cache key or None for empty projections (never worth caching)."""
    if watermark_memory_id is None:
        return None
    return (org_id, projection_version, watermark_memory_id)


def _cache_get(
    org_id: uuid.UUID,
    projection_version: int,
    watermark_memory_id: uuid.UUID | None,
) -> OrganizationalState | None:
    key = _cache_key(org_id, projection_version, watermark_memory_id)
    if key is None:
        return None
    state = _STATE_CACHE.get(key)
    if state is not None:
        _STATE_CACHE.move_to_end(key)  # LRU touch
    return state


def _cache_put(state: OrganizationalState) -> None:
    key = _cache_key(
        state.org_id, state.projection_version, state.watermark.last_memory_id
    )
    if key is None:
        return
    _STATE_CACHE[key] = state
    _STATE_CACHE.move_to_end(key)
    while len(_STATE_CACHE) > _STATE_CACHE_MAX:
        _STATE_CACHE.popitem(last=False)


def _cache_evict_org(org_id: uuid.UUID) -> None:
    """Drop every cached projection for an org. Called on snapshot disposal so a
    forced rebuild is never served the pre-disposal state."""
    stale = [k for k in _STATE_CACHE if k[0] == org_id]
    for k in stale:
        del _STATE_CACHE[k]


def _cache_clear() -> None:
    """Public-in-tests helper. Not exported from the module __init__."""
    _STATE_CACHE.clear()


class StateService:
    def __init__(
        self,
        db: AsyncSession | None = None,
        *,
        reader: MemoryReader | None = None,
        snapshots: SnapshotRepository | None = None,
    ) -> None:
        self.db = db
        if reader is not None:
            self.reader = reader
        else:
            if db is None:
                raise ValueError("StateService requires `db` when `reader` is not provided")
            self.reader = MemoryReader(db)
        if snapshots is not None:
            self.snapshots = snapshots
        else:
            if db is None:
                raise ValueError("StateService requires `db` when `snapshots` is not provided")
            self.snapshots = SnapshotRepository(db)

    # --- public API ----------------------------------------------------------

    async def rebuild_org(
        self, org_id: uuid.UUID, *, as_of: datetime | None = None
    ) -> OrganizationalState:
        # Single-flight per org (best-effort, non-blocking): a concurrent full rebuild is
        # wasted O(org) work. A loser serves the freshest usable snapshot if one exists,
        # else rebuilds anyway (correctness > dedup). as_of rebuilds bypass the fallback —
        # they must reflect the requested point, not whatever snapshot happens to exist.
        if as_of is None and not await self._acquire_org_lock(org_id):
            metrics.incr("state.rebuild.contended")
            existing = self._deserialize(
                await self.snapshots.latest(org_id, projection_version=PROJECTION_VERSION)
            )
            if existing is not None:
                logger.info("state rebuild contended org=%s -> served snapshot", org_id)
                return existing

        state, facts = await self._fold_stream(org_id, as_of=as_of)
        assert state is not None  # base=None always yields a (possibly empty) projection
        await self.snapshots.save(state)
        _cache_put(state)
        metrics.incr("state.rebuild")
        logger.info("state rebuilt org=%s facts=%d %s", org_id, facts, state.counts())
        return state

    async def incremental_update(self, org_id: uuid.UUID) -> OrganizationalState:
        # Warm-cache fast path — the /context hot path. If we already have the
        # deserialized snapshot in-process AND nothing new has landed past its
        # watermark, we can skip both the JSONB payload fetch and the Python
        # deserialize (the two dominant costs, together ~10s on a warm org).
        #
        # Correctness gate: an EXISTS check against `memory_records` past the
        # cached watermark. If any row landed, we fall through to the original
        # fold-from-snapshot flow and re-derive state deterministically.
        wm = await self.snapshots.watermark_only(
            org_id, projection_version=PROJECTION_VERSION
        )
        if wm is not None:
            cached = _cache_get(org_id, PROJECTION_VERSION, wm[1])
            if cached is not None and not await self.reader.has_records_after(
                org_id, cached.watermark
            ):
                metrics.incr("state.cache.warm_hit")
                return cached

        snap = await self.snapshots.latest(org_id, projection_version=PROJECTION_VERSION)
        prev = self._deserialize(snap)
        if prev is None:
            if snap is not None:
                await self._dispose(org_id)  # unusable/corrupt -> drop so rebuild can overwrite
            return await self.rebuild_org(org_id)

        state, facts = await self._fold_stream(org_id, after=prev.watermark, base=prev)
        if state is None:  # nothing past the watermark
            # Populate the warm cache so the next call takes the fast path
            # above; on cold Uvicorn workers this is the first time we've
            # paid the deserialize cost for this (org, watermark).
            _cache_put(prev)
            return prev
        await self.snapshots.save(state)
        _cache_put(state)
        metrics.incr("state.incremental")
        logger.info("state incremental org=%s +%d facts %s", org_id, facts, state.counts())
        return state

    async def get_state(self, org_id: uuid.UUID) -> OrganizationalState:
        # Warm-cache fast path — mirrors incremental_update. `get_state` is
        # allowed to serve a snapshot even without a fresh-memory check,
        # since the caller has already opted into "cached projection is fine".
        wm = await self.snapshots.watermark_only(
            org_id, projection_version=PROJECTION_VERSION
        )
        if wm is not None:
            cached = _cache_get(org_id, PROJECTION_VERSION, wm[1])
            if cached is not None:
                metrics.incr("state.cache.warm_hit")
                return cached

        snap = await self.snapshots.latest(org_id, projection_version=PROJECTION_VERSION)
        state = self._deserialize(snap)
        if state is not None:
            _cache_put(state)
            metrics.incr("state.cache_hit")
            return state
        if snap is not None:  # present but unusable (corrupt / version drift) -> dispose
            await self._dispose(org_id)
        return await self.rebuild_org(org_id)

    # --- internals -----------------------------------------------------------

    def _deserialize(self, snap) -> OrganizationalState | None:
        """Return the cached projection, or None if it must be rebuilt: absent, corrupt
        JSON, or schema/projection-version drift. NEVER raises — a bad cache degrades to a
        deterministic rebuild from Memory (version is read from the serialized projection,
        so it is validated even though `latest()` already gates on the key)."""
        if snap is None:
            return None
        try:
            state = OrganizationalState.from_dict(snap.full_projection)
        except Exception:
            metrics.incr("state.snapshot.corrupt")
            logger.exception(
                "corrupt state snapshot org=%s -> dispose + rebuild",
                getattr(snap, "organization_id", None),
            )
            return None
        if state.schema_version != STATE_SCHEMA_VERSION or state.projection_version != PROJECTION_VERSION:
            metrics.incr("state.snapshot.schema_mismatch")
            logger.warning(
                "state snapshot version drift org=%s schema=%s/%s projection=%s/%s -> rebuild",
                state.org_id, state.schema_version, STATE_SCHEMA_VERSION,
                state.projection_version, PROJECTION_VERSION,
            )
            return None
        return state

    async def _dispose(self, org_id: uuid.UUID) -> None:
        await self.snapshots.delete(org_id, projection_version=PROJECTION_VERSION)
        # Evict from the in-process cache too — otherwise a post-dispose
        # `get_state` fast path could still serve the pre-dispose projection.
        _cache_evict_org(org_id)
        metrics.incr("state.snapshot.disposed")

    async def _fold_stream(
        self,
        org_id: uuid.UUID,
        *,
        after=None,
        base: OrganizationalState | None = None,
        as_of: datetime | None = None,
    ) -> tuple[OrganizationalState | None, int]:
        """Fold the memory stream WITHOUT materializing the full record set (streaming
        rebuild). Relies on the reader's (event_timestamp, memory_id) ordering — the same
        total order build_projection sorts into — so the result is byte-identical to a full
        build_projection / continue_projection. `base` continues a prior projection
        (incremental, deep-copied lazily so a no-op costs nothing); None starts from empty
        (full). Returns (state_or_None, facts_folded); state is None ONLY for an incremental
        with zero new records (the caller keeps `base`)."""
        builder = None if base is not None else StateBuilder(org_id)
        facts = 0
        async for rec in self.reader.stream(org_id, after=after):
            if builder is None:
                assert base is not None  # builder is None ⟺ base was provided
                builder = StateBuilder.from_state(base)
            builder.apply(rec)
            facts += 1
        if builder is None:
            return None, 0
        return builder.finalize(as_of=as_of), facts

    async def _acquire_org_lock(self, org_id: uuid.UUID) -> bool:
        """Non-blocking, TRANSACTION-scoped per-org advisory lock. True if acquired (or no
        DB — single-flight is N/A in unit tests). Auto-released when the caller's
        transaction ends (no manual unlock → PgBouncer-safe, gap-free)."""
        if self.db is None:
            return True
        res = await self.db.execute(
            text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": _org_lock_key(org_id)}
        )
        return bool(res.scalar())
