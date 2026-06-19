"""Observation event subscribers — the Execution -> Observation link.

Closes the loop: Execution publishes ACTION_EXECUTED / ACTION_FAILED (at-least-once,
with a stable event_id and full lineage); Observation captures each as an immutable
outcome, guarded by the inbox so duplicate deliveries collapse to one.

(ACTION_RETRIED has no dedicated event in Execution V2 — a retry re-emits
ACTION_EXECUTING then EXECUTED/FAILED, which we already capture. No redesign.)
"""
from __future__ import annotations

import uuid

from navis.observation.inbox import process_once
from navis.observation.models import OutcomeType
from navis.observation.service import ObservationService
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, subscribe
from navis.platform.monitoring import get_logger

logger = get_logger("observation.subscribers")

_CONSUMER = "observation"


async def _capture(event: DomainEvent, outcome_type: OutcomeType) -> None:
    if event.org_id is None:
        return
    source_event_id = (event.payload or {}).get("event_id")
    if not source_event_id:
        # Without a stable event id we cannot dedupe; skip rather than risk doubles.
        logger.warning("execution event without event_id; skipping outcome capture")
        return

    async with session_scope() as db:
        from navis.observation.repository import OutcomeRepository

        repo = OutcomeRepository(db)
        svc = ObservationService(db, repo=repo)

        async def handler() -> None:
            await svc.record_from_execution(event, outcome_type, source_event_id)

        await process_once(repo, _CONSUMER, str(source_event_id), handler)


@subscribe(EventType.ACTION_EXECUTED)
async def on_action_executed(event: DomainEvent) -> None:
    await _capture(event, OutcomeType.ACTION_EXECUTED)


@subscribe(EventType.ACTION_FAILED)
async def on_action_failed(event: DomainEvent) -> None:
    await _capture(event, OutcomeType.ACTION_FAILED)
