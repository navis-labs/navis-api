"""Learning event subscriber — consumes OUTCOME_OBSERVED only.

Inbox-guarded (dedupe on the stable outcome_id) so at-least-once delivery and
replay never double-count. Learning reads ONLY this event — never Memory/State/
Reasoning/Decision/Execution/Intelligence.
"""
from __future__ import annotations

from navis.learning.inbox import process_once
from navis.learning.service import LearningService
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, subscribe
from navis.platform.monitoring import get_logger

logger = get_logger("learning.subscribers")

_CONSUMER = "learning"


@subscribe(EventType.OUTCOME_OBSERVED)
async def on_outcome_observed(event: DomainEvent) -> None:
    if event.org_id is None:
        return
    payload = event.payload or {}
    outcome_id = payload.get("outcome_id")
    if not outcome_id:
        logger.warning("OUTCOME_OBSERVED without outcome_id; cannot dedupe; skipping")
        return

    async with session_scope() as db:
        from navis.learning.repository import LearningRepository

        repo = LearningRepository(db)
        svc = LearningService(db, repo=repo)

        async def handler() -> None:
            await svc.record_from_outcome(payload)

        await process_once(repo, _CONSUMER, str(outcome_id), handler)
