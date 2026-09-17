"""State V2 data access.

  MemoryReader      — read-only over `memory_records` ONLY (never legacy tables,
                      never signals, never reasoning). Two reads, two clocks:
                      `stream` walks the whole org in EVENT order (the fold order
                      a full rebuild needs), `observed_since` returns the delta in
                      OBSERVATION order (the only order a resume cursor may use).
                      `count` is the completeness probe.
  SnapshotRepository — load/save the cached projection (a cache, not truth).
"""
from __future__ import annotations

import uuid
from typing import AsyncIterator

from sqlalchemy import delete as sa_delete, func, literal, or_, select, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.models import MemoryRecord
from navis.state.v2.builders import MemoryRecordView
from navis.state.v2.models import OrganizationalState, Watermark
from navis.state.v2.snapshots import StateSnapshot


class MemoryReader:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    @staticmethod
    def _view(row: MemoryRecord) -> MemoryRecordView:
        return MemoryRecordView(
            memory_id=row.memory_id,
            org_id=row.organization_id,
            entity_type=row.entity_type,
            entity_id=row.entity_id,
            event_timestamp=row.event_timestamp,
            payload=row.payload or {},
            source_signal_event_id=row.source_signal_event_id,
            schema_version=row.schema_version,
            confidence=row.confidence,
            # The observation clock. `created_at` is DB-assigned
            # (`server_default=func.now()`), never written by the application,
            # so it is the one timestamp on this row that cannot be in the future.
            observed_at=row.created_at,
        )

    async def count(self, org_id: uuid.UUID) -> int:
        """How many memory records this organization has, total.

        THE COMPLETENESS PROBE, and it replaced an EXISTS check that could not
        do the job. `has_records_after(cursor)` answers "is there anything NEW?"
        — which is silent about anything that landed BELOW an already-advanced
        cursor. That is precisely the class of row State was losing: an EXISTS
        probe returns False, the cached projection is served as authoritative,
        and the record is never folded.

        A count answers both questions in the same round-trip: compared against
        `OrganizationalState.record_count` it detects new records AND stranded
        ones, and it cannot be fooled by where in the ordering the gap sits.

        Served from `ix_mem_org_time`, whose leading column is organization_id,
        so this is an index-only scan of one tenant's entries.
        """
        return int(
            await self.db.scalar(
                select(func.count())
                .select_from(MemoryRecord)
                .where(MemoryRecord.organization_id == org_id)
            )
            or 0
        )

    async def stream(
        self, org_id: uuid.UUID, *, batch: int = 5000
    ) -> AsyncIterator[MemoryRecordView]:
        """Every record for the org in EVENT order — the full-rebuild fold order.

        Keyset-paginated on (event_timestamp, memory_id) so a 10M-record org
        streams in O(batch) memory. No `after` parameter, deliberately: resuming
        an event-ordered stream from a cursor is the defect this module was
        rewritten to remove. Incremental reads go through `observed_since`.
        """
        last_ts, last_mid = None, None
        while True:
            stmt = select(MemoryRecord).where(MemoryRecord.organization_id == org_id)
            if last_ts is not None:
                stmt = stmt.where(
                    tuple_(MemoryRecord.event_timestamp, MemoryRecord.memory_id)
                    > tuple_(last_ts, last_mid)
                )
            stmt = stmt.order_by(
                MemoryRecord.event_timestamp.asc(), MemoryRecord.memory_id.asc()
            ).limit(batch)
            rows = (await self.db.scalars(stmt)).all()
            if not rows:
                return
            for row in rows:
                yield self._view(row)
            if len(rows) < batch:
                return
            last_ts, last_mid = rows[-1].event_timestamp, rows[-1].memory_id

    async def observed_since(
        self, org_id: uuid.UUID, cursor: Watermark, *, limit: int
    ) -> list[MemoryRecordView] | None:
        """Records observed strictly after `cursor`, in OBSERVATION order.

        Returns None when the delta exceeds `limit` — the caller must then do a
        full rebuild, which streams and costs the same work at bounded memory.
        Materializing is safe precisely because it is bounded: the caller has to
        SORT this delta into event order before folding it (the reader cannot,
        since the two orders differ), and a sort needs the rows in hand.

        `cursor.last_timestamp is None` means nothing has been folded yet, so
        everything is new — which is also what an empty projection wants.
        """
        stmt = select(MemoryRecord).where(MemoryRecord.organization_id == org_id)
        if cursor.last_timestamp is not None and cursor.last_memory_id is not None:
            stmt = stmt.where(
                tuple_(MemoryRecord.created_at, MemoryRecord.memory_id)
                > tuple_(cursor.last_timestamp, cursor.last_memory_id)
            )
        # limit + 1 so "exactly at the cap" is distinguishable from "over it"
        # without a second COUNT round-trip.
        stmt = stmt.order_by(
            MemoryRecord.created_at.asc(), MemoryRecord.memory_id.asc()
        ).limit(limit + 1)
        rows = (await self.db.scalars(stmt)).all()
        if len(rows) > limit:
            return None
        return [self._view(row) for row in rows]


class SnapshotRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def latest(
        self, org_id: uuid.UUID, *, projection_version: int
    ) -> StateSnapshot | None:
        return await self.db.scalar(
            select(StateSnapshot).where(
                StateSnapshot.organization_id == org_id,
                StateSnapshot.projection_version == projection_version,
            )
        )

    async def watermark_only(
        self, org_id: uuid.UUID, *, projection_version: int
    ) -> tuple[datetime | None, uuid.UUID | None] | None:
        """Return just the snapshot's watermark cursor without the JSONB payload.

        Used by StateService's cache short-circuit: the watermark is all we
        need to compute the in-process cache key, and skipping the ~22KB
        `full_projection` column shaves both the JSONB fetch and the Python
        deserialization from a warm /context call — the two dominant costs on
        that hot path.

        Returns None if the snapshot row is absent for (org, projection_version).
        """
        stmt = select(
            StateSnapshot.watermark_timestamp, StateSnapshot.watermark_memory_id
        ).where(
            StateSnapshot.organization_id == org_id,
            StateSnapshot.projection_version == projection_version,
        )
        row = (await self.db.execute(stmt)).one_or_none()
        return None if row is None else (row[0], row[1])

    async def save(self, state: OrganizationalState) -> None:
        """Upsert the cache for (org, projection_version). Caches are disposable."""
        wm = state.watermark
        values = dict(
            organization_id=state.org_id,
            projection_version=state.projection_version,
            schema_version=state.schema_version,
            watermark_timestamp=wm.last_timestamp,
            watermark_memory_id=wm.last_memory_id,
            as_of=state.as_of,
            full_projection=state.to_dict(),
        )
        # Monotonic upsert (stale-snapshot protection): overwrite ONLY if the incoming
        # watermark is >= the stored one (or the stored snapshot is empty). Two writers
        # racing on one org (e.g. a scheduled rebuild vs. an incremental update) can land
        # out of order; without this guard a stale write could regress the cache to an
        # older watermark, briefly serving Reasoning a backwards projection and re-folding
        # already-folded records. The comparison mirrors `order_key` exactly — Postgres
        # uuid ordering equals memory_id.int ordering — so it is replay-deterministic and
        # never blocks a genuinely newer write. A no-conflict INSERT is unaffected.
        not_stale = or_(
            StateSnapshot.watermark_timestamp.is_(None),
            tuple_(StateSnapshot.watermark_timestamp, StateSnapshot.watermark_memory_id)
            <= tuple_(literal(wm.last_timestamp), literal(wm.last_memory_id)),
        )
        stmt = pg_insert(StateSnapshot).values(**values).on_conflict_do_update(
            index_elements=["organization_id", "projection_version"],
            set_={
                "schema_version": values["schema_version"],
                "watermark_timestamp": values["watermark_timestamp"],
                "watermark_memory_id": values["watermark_memory_id"],
                "as_of": values["as_of"],
                "full_projection": values["full_projection"],
            },
            where=not_stale,
        )
        await self.db.execute(stmt)

    async def delete(
        self, org_id: uuid.UUID, *, projection_version: int | None = None
    ) -> None:
        """Dispose the disposable cache for an org. The snapshot is never a source of
        truth, so dropping it is always safe — the next read rebuilds deterministically
        from Memory. This is the GDPR/erasure primitive: after a subject is erased in
        Memory, drop the org's snapshot so the rebuilt projection no longer carries the
        erased PII. `projection_version=None` drops every version for the org (full
        cache disposal); a value drops just that version. Org *deletion* is already
        handled by the snapshot's ON DELETE CASCADE to organizations."""
        stmt = sa_delete(StateSnapshot).where(StateSnapshot.organization_id == org_id)
        if projection_version is not None:
            stmt = stmt.where(StateSnapshot.projection_version == projection_version)
        await self.db.execute(stmt)
