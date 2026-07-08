"""LearningReplayService — rebuild aggregates + patterns from the immutable log.

The replayability proof: delete all learning projections, fold every learning
event back through the SAME deterministic count logic, and get identical
aggregates. Counts commute, so order/timing are irrelevant.

Concurrency: rebuild deletes then rewrites the projections, so it must not run
concurrently with the live OUTCOME_OBSERVED consumer for the same org (a live
increment could be wiped by the delete). Run it while ingestion is quiesced, or
under a per-org advisory lock — the same isolation state.v2 rebuild_org uses.
"""
from __future__ import annotations

import uuid

from navis.learning.aggregates import Delta, aggregate_keys
from navis.learning.patterns import derive_signal_patterns
from navis.learning.repository import LearningRepository
from navis.platform.monitoring import get_logger

logger = get_logger("learning.replay")


class LearningReplayService:
    def __init__(self, db, *, repo=None) -> None:
        self.repo = repo or LearningRepository(db)

    async def rebuild(self, org_id: uuid.UUID) -> dict:
        # STREAM the immutable log (keyset-paginated) and fold into bounded in-memory counts.
        # The old path — all_events(limit=1_000_000) — SILENTLY TRUNCATED at 1M (a >1M-event
        # org rebuilt WRONG aggregates) and materialized every event at once (O(events) memory
        # spike). Streaming is O(distinct keys) memory, never truncates, and — because counts
        # commute — produces byte-identical aggregates to a live projection.
        counts: dict[tuple[str, str], Delta] = {}
        total = 0
        async for event in self.repo.stream_events(org_id):
            total += 1
            for dimension, key, delta in aggregate_keys(event):
                prev = counts.get((dimension, key))
                counts[(dimension, key)] = delta if prev is None else Delta(
                    prev.successes + delta.successes,
                    prev.failures + delta.failures,
                    prev.neutrals + delta.neutrals,
                    prev.total + delta.total,
                )

        # Rewrite from the folded totals: delete, then ONE upsert per (dimension, key) — after
        # the delete the aggregate row is absent, so a single increment writes its final count
        # (far fewer writes than one increment per event, identical result).
        await self.repo.delete_projections(org_id)
        for (dimension, key), delta in counts.items():
            await self.repo.increment_aggregate(org_id, dimension, key, delta)

        # Patterns recomputed once over the final signal aggregates (pure derivation).
        sig_aggs = await self.repo.signal_aggregates(org_id)
        for fact in derive_signal_patterns(org_id, sig_aggs):
            await self.repo.upsert_pattern(fact)

        await self.repo.flush()
        logger.info("rebuilt learning org=%s from %d events", org_id, total)
        return {"events": total}
