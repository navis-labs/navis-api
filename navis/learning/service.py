"""LearningService — attribute outcomes to signals, deterministically.

Flow (idempotent + append-only):
    OUTCOME_OBSERVED -> learning_event (immutable, one per outcome)
        -> increment count aggregates (signal / decision / outcome)
        -> derive patterns -> emit LEARNING_UPDATED (with lineage)

No EMA, no datetime.now(), no uuid4, no LLM. Aggregates are pure counts, so the
same events produce the same learning state regardless of order or timing.
"""
from __future__ import annotations

from navis.learning.aggregates import aggregate_keys
from navis.learning.attribution import learning_event_values
from navis.learning.patterns import derive_signal_patterns
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger

logger = get_logger("learning")


class LearningService:
    def __init__(self, db=None, *, repo=None, publisher=None) -> None:
        if repo is not None:
            self.repo = repo
        else:
            from navis.learning.repository import LearningRepository

            self.repo = LearningRepository(db)
        self._publish = publisher or event_bus.publish

    async def record_from_outcome(self, payload: dict) -> bool:
        values = learning_event_values(payload)
        if values is None:
            logger.warning("outcome missing lineage/occurred_at; skipping learning")
            return False

        row, created = await self.repo.append_event(values)
        if not created:
            return False  # idempotent: this outcome already learned

        await self._apply(row)
        await self.repo.flush()
        await self._emit(row)
        return True

    async def _apply(self, learning_event) -> None:
        """Fold one learning event into the count aggregates + refresh its patterns."""
        for dimension, key, delta in aggregate_keys(learning_event):
            await self.repo.increment_aggregate(
                learning_event.organization_id, dimension, key, delta
            )
        await self._refresh_patterns(
            learning_event.organization_id, learning_event.reasoning_signal_ids or []
        )

    async def _refresh_patterns(self, org_id, signal_ids: list) -> None:
        if not signal_ids:
            return
        wanted = {str(s) for s in signal_ids}
        sig_aggs = [a for a in await self.repo.signal_aggregates(org_id) if a.key in wanted]
        for fact in derive_signal_patterns(org_id, sig_aggs):
            await self.repo.upsert_pattern(fact)

    async def _emit(self, o) -> None:
        await self._publish(
            DomainEvent(
                type=EventType.LEARNING_UPDATED,
                org_id=o.organization_id,
                source="learning",
                payload={
                    "outcome_id": str(o.outcome_id),
                    "decision_id": str(o.decision_id) if o.decision_id else None,
                    "reasoning_signal_ids": o.reasoning_signal_ids,
                    "reasoning_version": o.reasoning_version,
                    "outcome_type": o.outcome_type,
                    "result": o.result,
                },
            )
        )
