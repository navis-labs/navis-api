"""MemoryReplayService — rebuild Memory from the Signal log.

The core guarantee: drop all memory_records, replay signal_events, and get
byte-identical Memory output (same memory_ids, same payloads). This works because
ids are deterministic and `append` is idempotent — replay simply re-derives and
re-appends; existing rows are no-ops.

Memory reads ONLY the signal log here — never State, never Reasoning.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.repository import MemoryRepository
from navis.memory.service import SignalViewFactory, build_memory_records
from navis.platform.monitoring import get_logger
from navis.signals.base import SignalSource
from navis.signals.repository import SignalRepository

logger = get_logger("memory.replay")


class MemoryReplayService:
    def __init__(self, db: AsyncSession) -> None:
        self.memory_repo = MemoryRepository(db)
        self.signal_repo = SignalRepository(db)

    async def replay_org(
        self,
        org_id: uuid.UUID,
        *,
        source: SignalSource | None = None,
        since: datetime | None = None,
        limit: int = 10000,
    ) -> dict:
        records = await self.signal_repo.for_replay(
            org_id, source=source, since=since, limit=limit
        )
        facts = 0
        new = 0
        for rec in records:
            view = SignalViewFactory.from_record(rec)
            for memory_record in build_memory_records(view):
                facts += 1
                if await self.memory_repo.append(memory_record):
                    new += 1
        logger.info(
            "memory replay org=%s signals=%d facts=%d new=%d", org_id, len(records), facts, new
        )
        return {"signals": len(records), "facts": facts, "new": new}
