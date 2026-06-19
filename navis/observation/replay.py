"""ObservationReplayService — re-emit outcomes from the immutable log.

Outcomes are external reality (captured, not computed), so they cannot be
"rebuilt" — but downstream PROJECTIONS (timelines, Learning's weights) can be
rebuilt by replaying the immutable outcome records. Replay re-publishes
OUTCOME_OBSERVED in occurred_at order; consumers stay idempotent.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from navis.observation.repository import OutcomeRepository
from navis.observation.service import ObservationService
from navis.platform.monitoring import get_logger

logger = get_logger("observation.replay")


class ObservationReplayService:
    def __init__(self, db) -> None:
        self.repo = OutcomeRepository(db)
        self.svc = ObservationService(db, repo=self.repo)

    async def replay(
        self, org_id: uuid.UUID, *, since: datetime | None = None, limit: int = 1000
    ) -> int:
        rows = await self.repo.for_replay(org_id, since=since, limit=limit)
        for outcome in rows:
            await self.svc.republish(outcome)
        logger.info("replayed %d outcomes org=%s", len(rows), org_id)
        return len(rows)
