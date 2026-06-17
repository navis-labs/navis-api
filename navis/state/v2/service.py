"""StateService — the SOLE interface to current truth.

Downstream (Reasoning, Decision, Execution, Learning, Intelligence) read
OrganizationalState from here and never query Memory directly.

  rebuild_org        — full deterministic rebuild from memory_records (disposable cache refresh)
  incremental_update — fold only memory_records after the snapshot watermark
  get_state          — return the cached projection (rebuild if absent)

State reads ONLY memory_records. No now() leaks into the projection (as_of is
derived from Memory, or passed explicitly for point-in-time queries).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from navis.platform.monitoring import get_logger
from navis.state.v2.builders import build_projection, continue_projection
from navis.state.v2.models import PROJECTION_VERSION, OrganizationalState
from navis.state.v2.repository import MemoryReader, SnapshotRepository

logger = get_logger("state.v2")


class StateService:
    def __init__(self, db=None, *, reader=None, snapshots=None) -> None:
        self.reader = reader if reader is not None else MemoryReader(db)
        self.snapshots = snapshots if snapshots is not None else SnapshotRepository(db)

    async def rebuild_org(
        self, org_id: uuid.UUID, *, as_of: datetime | None = None
    ) -> OrganizationalState:
        records = [v async for v in self.reader.stream(org_id)]
        state = build_projection(org_id, records, as_of=as_of)
        await self.snapshots.save(state)
        logger.info("state rebuilt org=%s facts=%d %s", org_id, len(records), state.counts())
        return state

    async def incremental_update(self, org_id: uuid.UUID) -> OrganizationalState:
        snap = await self.snapshots.latest(org_id, projection_version=PROJECTION_VERSION)
        if snap is None:
            return await self.rebuild_org(org_id)

        prev = OrganizationalState.from_dict(snap.full_projection)
        new = [v async for v in self.reader.stream(org_id, after=prev.watermark)]
        if not new:
            return prev
        state = continue_projection(prev, new)
        await self.snapshots.save(state)
        logger.info("state incremental org=%s +%d facts %s", org_id, len(new), state.counts())
        return state

    async def get_state(self, org_id: uuid.UUID) -> OrganizationalState:
        snap = await self.snapshots.latest(org_id, projection_version=PROJECTION_VERSION)
        if snap is None:
            return await self.rebuild_org(org_id)
        return OrganizationalState.from_dict(snap.full_projection)
