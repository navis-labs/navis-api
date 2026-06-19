"""LearningReplayService — rebuild aggregates + patterns from the immutable log.

The replayability proof: delete all learning projections, fold every learning
event back through the SAME deterministic count logic, and get identical
aggregates. Counts commute, so order/timing are irrelevant.
"""
from __future__ import annotations

import uuid

from navis.learning.aggregates import aggregate_keys
from navis.learning.patterns import derive_signal_patterns
from navis.learning.repository import LearningRepository
from navis.platform.monitoring import get_logger

logger = get_logger("learning.replay")


class LearningReplayService:
    def __init__(self, db, *, repo=None) -> None:
        self.repo = repo or LearningRepository(db)

    async def rebuild(self, org_id: uuid.UUID) -> dict:
        await self.repo.delete_projections(org_id)

        events = await self.repo.all_events(org_id)
        for event in events:
            for dimension, key, delta in aggregate_keys(event):
                await self.repo.increment_aggregate(org_id, dimension, key, delta)

        # Patterns recomputed once over the final signal aggregates (pure derivation).
        sig_aggs = await self.repo.signal_aggregates(org_id)
        for fact in derive_signal_patterns(org_id, sig_aggs):
            await self.repo.upsert_pattern(fact)

        await self.repo.flush()
        logger.info("rebuilt learning org=%s from %d events", org_id, len(events))
        return {"events": len(events)}
