"""StateService — the SOLE interface to current truth.

Downstream (Reasoning, Decision, Execution, Learning, Intelligence) read
OrganizationalState from here and never query Memory directly.

  rebuild_org        — full deterministic rebuild, STREAMED from memory_records
  incremental_update — fold only memory_records observed after the snapshot cursor
  get_state          — return the cached projection (rebuild if absent/invalid)

THE TWO CLOCKS. `event_timestamp` says when a business fact HAPPENED and drives the
fold order; `observed_at` (= `memory_records.created_at`) says when Navis LEARNED it
and drives the resume cursor and `as_of`. They are not interchangeable: the Calendar
connector deliberately stamps a signal with the meeting's START, so event time runs
ahead of wall clock and is unusable as a cursor. See `state/v2/models.py`.

TWO GUARDS, both fail into `rebuild_org` rather than into a wrong answer:
  * COMPLETENESS — `reader.count()` vs `state.record_count`. A row that landed
    below an already-advanced cursor is invisible to the cursor and unmissable to
    the count. This is what stops records being silently skipped.
  * ORDERING — the delta's earliest record vs `state.fold_frontier`. The reducer is
    latest-non-null-wins in event order, so folding a record that happened before
    something already folded would clobber a newer fact. Detected, never absorbed.

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
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.state.v2.builders import StateBuilder
from navis.state.v2.models import (
    PROJECTION_VERSION,
    STATE_SCHEMA_VERSION,
    OrganizationalState,
    order_key,
)
from navis.state.v2.repository import MemoryReader, SnapshotRepository

logger  = get_logger("state.v2")

# Advisory-lock space for State — disjoint from signals/memory/execution (blake2b
# personalization), so a State per-org lock never collides with another domain's lock.
_LOCK_NAMESPACE = b"navis.state"

# How many newly-observed records an incremental fold will hold in memory before
# giving up and rebuilding. The delta MUST be materialized (it arrives in
# observation order and has to be sorted into event order before folding), so it
# needs a ceiling. Past it a full rebuild is the better answer anyway: it streams
# at O(entities) memory and does the same amount of folding work.
_MAX_INCREMENTAL_DELTA = 5000

# ── In-process cache of deserialized OrganizationalState ────────────────────
# Keyed by (org_id, projection_version, watermark_memory_id) so that a
# snapshot whose watermark hasn't moved since the last request short-circuits
# both the JSONB payload fetch and the Python-side ~O(entities) re-parse of
# `full_projection`. Those two steps are the dominant cost on the /context hot
# path — measured p50 ~12s on a warm org before this cache; the same call
# returns in <100ms once the deserialized state is in-process.
#
# Correctness (this cache never returns a stale projection):
#   * Any new memory record advances state.watermark.last_memory_id, which
#     changes the cache key. The old entry becomes unreachable and eventually
#     LRU-evicts.
#   * Every call still verifies via MemoryReader.count() that the org holds
#     exactly as many records as the cached projection folded. That replaced a
#     `has_records_after` EXISTS probe, which could only see records ABOVE the
#     cursor and therefore certified a projection with stranded rows below it as
#     authoritative. A count sees both, in the same round-trip.
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
        self, org_id: uuid.UUID, *, as_of: datetime | None = None, force: bool = False
    ) -> OrganizationalState:
        # Single-flight per org (best-effort, non-blocking): a concurrent full rebuild is
        # wasted O(org) work. A loser serves the freshest usable snapshot if one exists,
        # else rebuilds anyway (correctness > dedup). as_of rebuilds bypass the fallback —
        # they must reflect the requested point, not whatever snapshot happens to exist.
        #
        # `force` bypasses it too, and that is not an optimization — it is required
        # for correctness. Every incremental bail-out (`_rebuild_because`) rebuilds
        # PRECISELY BECAUSE the stored snapshot was found incomplete or out of
        # order. Handing that same row back on lock contention would answer "this
        # projection is missing records" with the projection missing records, which
        # is the silent staleness this whole change exists to remove. Losing the
        # lock costs a duplicated fold; serving the rejected snapshot costs the
        # correctness we just proved was absent.
        if as_of is None and not force and not await self._acquire_org_lock(org_id):
            metrics.incr("state.rebuild.contended")
            existing = self._deserialize(
                await self.snapshots.latest(org_id, projection_version=PROJECTION_VERSION)
            )
            if existing is not None:
                logger.info("state rebuild contended org=%s -> served snapshot", org_id)
                return existing

        state, facts = await self._fold_full(org_id, as_of=as_of)
        await self.snapshots.save(state)
        _cache_put(state)
        metrics.incr("state.rebuild")
        self._log_horizon("rebuilt", state, facts)
        return state

    async def incremental_update(self, org_id: uuid.UUID) -> OrganizationalState:
        """Fold everything observed since the snapshot cursor, or rebuild.

        There is no third outcome. Every path below either produces a projection
        that provably folded all `total` records in correct event order, or calls
        `rebuild_org`. Nothing is ever skipped and nothing stale is ever served.
        """
        # One round-trip that answers both "is there anything new?" and "is
        # anything missing?". Cheaper than the fold it guards and strictly more
        # informative than the EXISTS probe it replaced.
        total = await self.reader.count(org_id)

        # Warm-cache fast path — the /context hot path. If we already have the
        # deserialized snapshot in-process AND it accounts for every record the
        # org holds, we skip both the JSONB payload fetch and the Python
        # deserialize (the two dominant costs, together ~10s on a warm org).
        wm = await self.snapshots.watermark_only(
            org_id, projection_version=PROJECTION_VERSION
        )
        if wm is not None:
            cached = _cache_get(org_id, PROJECTION_VERSION, wm[1])
            if cached is not None and cached.record_count == total:
                metrics.incr("state.cache.warm_hit")
                return cached

        snap = await self.snapshots.latest(org_id, projection_version=PROJECTION_VERSION)
        prev = self._deserialize(snap)
        if prev is None:
            if snap is not None:
                await self._dispose(org_id)  # unusable/corrupt -> drop so rebuild can overwrite
            return await self.rebuild_org(org_id)

        delta = await self.reader.observed_since(
            org_id, prev.watermark, limit=_MAX_INCREMENTAL_DELTA
        )
        if delta is None:
            # Bigger than we are willing to hold in memory. A rebuild streams and
            # costs the same work at bounded memory, so it is the cheaper answer.
            return await self._rebuild_because(org_id, "delta_too_large")

        if not delta:
            if prev.record_count != total:
                # Nothing NEW, yet the counts disagree — rows exist below a cursor
                # that has already moved past them. This is the stranding an
                # EXISTS probe could never see, and it is the reason the fast
                # path counts instead of probing.
                return await self._rebuild_because(org_id, "records_stranded")
            # Populate the warm cache so the next call takes the fast path
            # above; on cold Uvicorn workers this is the first time we've
            # paid the deserialize cost for this (org, cursor).
            _cache_put(prev)
            return prev

        # The delta arrives in OBSERVATION order; the reducer needs EVENT order.
        # Sorting here is what lets one sync batch (all rows sharing a single
        # transaction timestamp) fold in the order the facts actually happened.
        delta.sort(key=lambda r: order_key(r.event_timestamp, r.memory_id))
        earliest = delta[0]
        if not prev.fold_frontier.precedes(earliest.event_timestamp, earliest.memory_id):
            # This record happened BEFORE something already folded. Applying it
            # now would overwrite a newer fact and diverge from a full rebuild,
            # so the fold order assumption is void and we re-derive from scratch.
            metrics.incr("state.out_of_order.detected")
            return await self._rebuild_because(org_id, "out_of_order")

        builder = StateBuilder.from_state(prev)
        for rec in delta:
            builder.apply(rec)
        state = builder.finalize(as_of=None)

        if state.record_count != total:
            # The delta was consistent but the arithmetic is not: something landed
            # below the cursor while we were folding, or a concurrent writer
            # committed out of order. Either way this projection cannot prove it
            # is complete, so it is not persisted.
            return await self._rebuild_because(org_id, "count_mismatch")

        await self.snapshots.save(state)
        _cache_put(state)
        metrics.incr("state.incremental")
        self._log_horizon("incremental", state, len(delta))
        return state

    async def _rebuild_because(
        self, org_id: uuid.UUID, reason: str
    ) -> OrganizationalState:
        """Fall back to the deterministic full rebuild, and SAY WHY.

        Every incremental bail-out routes through here so the reason is both
        metered and logged. A silent rebuild is almost as bad as a silent skip:
        `out_of_order` and `records_stranded` are the two signals that tell an
        operator a connector is delivering late or a future-dated event is in
        play, and neither is visible from the projection itself.
        """
        metrics.incr("state.rebuild.fallback", labels={"reason": reason})
        logger.warning(
            "state incremental fold invalid org=%s reason=%s -> full rebuild",
            org_id, reason,
        )
        # force=True: the stored snapshot is the thing we just rejected, so the
        # contention fallback must not hand it back. See `rebuild_org`.
        return await self.rebuild_org(org_id, force=True)

    @staticmethod
    def _log_horizon(kind: str, state: OrganizationalState, facts: int) -> None:
        """Structured, content-free record of where this projection's clock sits.

        Ids, counts and timestamps only — never a payload, a subject or an
        address. `observation_horizon` is the invariant an operator watches:
        it must never exceed wall clock, and the gap between it and now is the
        organization's real projection lag.
        """
        horizon = state.watermark.last_timestamp
        lag_s: float | None = None
        if horizon is not None:
            lag_s = max(
                0.0, (datetime.now(timezone.utc) - horizon).total_seconds()
            )
            metrics.observe_ms("state.projection.lag", lag_s * 1000.0)
        logger.info(
            "state %s org=%s facts=%d records=%d observation_horizon=%s "
            "as_of=%s fold_frontier=%s lag_s=%s %s",
            kind, state.org_id, facts, state.record_count,
            horizon.isoformat() if horizon else None,
            state.as_of.isoformat() if state.as_of else None,
            (
                state.fold_frontier.last_event_timestamp.isoformat()
                if state.fold_frontier.last_event_timestamp
                else None
            ),
            None if lag_s is None else round(lag_s, 3),
            state.counts(),
        )

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
        # AN OBSERVATION CURSOR IN THE FUTURE IS NOT A STALE CACHE, IT IS A LIE.
        #
        # `created_at` is assigned by Postgres at insert, so a cursor derived from
        # it cannot exceed wall clock. If a stored one does, the row was written by
        # something that did not mean observation time by it — which is exactly what
        # production's orphaned `projection_version = 2` row turned out to be.
        #
        # Trusting it is unrecoverable rather than merely wrong: the delta past a
        # future cursor is always empty, and the monotonic upsert guard then refuses
        # every corrected write for being "older". So the invariant is enforced on
        # READ, where a violation still costs one rebuild and disposes the bad row.
        # This is the only clock reference in State, and it validates a CACHE — the
        # projection itself stays a pure function of (memory_records, as_of).
        cursor = state.watermark.last_timestamp
        if cursor is not None and cursor > datetime.now(timezone.utc):
            metrics.incr("state.snapshot.future_cursor")
            logger.error(
                "state snapshot cursor is in the future org=%s cursor=%s -> dispose + rebuild",
                state.org_id, cursor.isoformat(),
            )
            return None
        return state

    async def _dispose(self, org_id: uuid.UUID) -> None:
        await self.snapshots.delete(org_id, projection_version=PROJECTION_VERSION)
        # Evict from the in-process cache too — otherwise a post-dispose
        # `get_state` fast path could still serve the pre-dispose projection.
        _cache_evict_org(org_id)
        metrics.incr("state.snapshot.disposed")

    async def _fold_full(
        self, org_id: uuid.UUID, *, as_of: datetime | None = None
    ) -> tuple[OrganizationalState, int]:
        """Fold the org's ENTIRE memory WITHOUT materializing the record set.

        Relies on the reader's (event_timestamp, memory_id) ordering — the same
        total order `build_projection` sorts into — so the result is byte-identical
        to `build_projection` over the same rows. Always returns a projection: an
        org with no records yields a well-formed empty one, which is what lets the
        caller treat a rebuild as the unconditional recovery path.
        """
        builder = StateBuilder(org_id)
        facts = 0
        async for rec in self.reader.stream(org_id):
            builder.apply(rec)
            facts += 1
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
