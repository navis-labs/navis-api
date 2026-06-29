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


def _org_lock_key(org_id: uuid.UUID) -> int:
    """Deterministic signed-bigint advisory-lock key for an org (full-uuid blake2b,
    State-namespaced). Stable across processes; ~2^-64 collision probability."""
    digest = hashlib.blake2b(org_id.bytes, digest_size=8, person=_LOCK_NAMESPACE).digest()
    return int.from_bytes(digest, "big", signed=True)


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
        metrics.incr("state.rebuild")
        logger.info("state rebuilt org=%s facts=%d %s", org_id, facts, state.counts())
        return state

    async def incremental_update(self, org_id: uuid.UUID) -> OrganizationalState:
        snap = await self.snapshots.latest(org_id, projection_version=PROJECTION_VERSION)
        prev = self._deserialize(snap)
        if prev is None:
            if snap is not None:
                await self._dispose(org_id)  # unusable/corrupt -> drop so rebuild can overwrite
            return await self.rebuild_org(org_id)

        state, facts = await self._fold_stream(org_id, after=prev.watermark, base=prev)
        if state is None:  # nothing past the watermark
            return prev
        await self.snapshots.save(state)
        metrics.incr("state.incremental")
        logger.info("state incremental org=%s +%d facts %s", org_id, facts, state.counts())
        return state

    async def get_state(self, org_id: uuid.UUID) -> OrganizationalState:
        snap = await self.snapshots.latest(org_id, projection_version=PROJECTION_VERSION)
        state = self._deserialize(snap)
        if state is not None:
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
