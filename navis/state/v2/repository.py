"""State V2 data access.

  MemoryReader      — read-only over `memory_records` ONLY (never legacy tables,
                      never signals, never reasoning). Keyset pagination on
                      (event_timestamp, memory_id) so a single-org rebuild streams
                      in deterministic order and incremental reads start at a watermark.
  SnapshotRepository — load/save the cached projection (a cache, not truth).
"""
from __future__ import annotations

import uuid
from typing import AsyncIterator

from sqlalchemy import select, tuple_
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
        )

    async def stream(
        self, org_id: uuid.UUID, *, after: Watermark | None = None, batch: int = 5000
    ) -> AsyncIterator[MemoryRecordView]:
        last_ts = after.last_timestamp if after else None
        last_mid = after.last_memory_id if after else None
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
        stmt = pg_insert(StateSnapshot).values(**values).on_conflict_do_update(
            index_elements=["organization_id", "projection_version"],
            set_={
                "schema_version": values["schema_version"],
                "watermark_timestamp": values["watermark_timestamp"],
                "watermark_memory_id": values["watermark_memory_id"],
                "as_of": values["as_of"],
                "full_projection": values["full_projection"],
            },
        )
        await self.db.execute(stmt)
