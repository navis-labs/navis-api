"""Intelligence event subscriber — consumes LEARNING_UPDATED only.

THE MISSING TRIGGER. `IntelligenceService.project()` was reachable only through
`POST /api/v1/intelligence/project`, so nothing in the running system ever called
it: `intelligence_insights` held 0 rows while Learning accumulated, and Adaptive —
which reads those insights to propose policy — had nothing to propose from. The
loop was wired end to end and never turned over.

Learning already announces itself. `LearningService._emit` publishes
LEARNING_UPDATED with the outcome/decision/reasoning lineage every time an outcome
is recorded, and that is precisely the moment the projection's inputs change.
Subscribing here means Intelligence recomputes when its evidence moves, rather
than on a timer that would mostly re-derive an unchanged answer.

Safe to run on every event, and safe to miss one:
  * the projection is a FULL recompute over the current LearningView, so a
    dropped event self-heals on the next one — no incremental state to drift;
  * `insight_id` is a deterministic UUIDv5 of (org, type, key, learning_version)
    and the table is append-only with ON CONFLICT DO NOTHING, so re-firing the
    same event writes nothing new;
  * a learning_version of 0 short-circuits before any work.

Intelligence still reads Learning ONLY. This module adds a trigger, not an input:
it hands `project()` an org id and nothing else.
"""
from __future__ import annotations

import uuid

from navis.intelligence.repository import IntelligenceRepository
from navis.intelligence.service import IntelligenceService
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, subscribe
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("intelligence.subscribers")


@subscribe(EventType.LEARNING_UPDATED)
async def on_learning_updated(event: DomainEvent) -> None:
    """Re-project organizational insights from the org's current Learning."""
    if event.org_id is None:
        return
    org_id = event.org_id if isinstance(event.org_id, uuid.UUID) else uuid.UUID(str(event.org_id))

    async with session_scope() as db:
        rows = await IntelligenceService(IntelligenceRepository(db)).project(org_id)

    # Zero is the normal, correct outcome below the sample threshold — counted
    # separately so "the trigger never fired" and "the evidence is still too
    # thin to say anything" stay distinguishable in production.
    metrics.incr("intelligence.projected", value=len(rows))
    if not rows:
        metrics.incr("intelligence.projection.empty")
    logger.info("intelligence projected org=%s insights=%d", org_id, len(rows))
