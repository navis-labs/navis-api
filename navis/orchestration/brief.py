"""BriefOrchestrator — the morning heartbeat.

The pipeline assembly (no business logic of its own):
    State V2 (get_state) -> Reasoning V2 (run_reasoning_v2) -> Decision V2 (decide)

Decision consumes Reasoning only; orchestration is the seam that feeds it.
Exposed both as a coroutine (for the API / tests) and a Celery task (schedule).
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.decisions.models import DailyBrief
from navis.decisions.service import DecisionService
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger
from navis.platform.queue import celery_app, run_async
from navis.reasoning.v2 import REASONING_VERSION, run_reasoning_v2
from navis.state.v2 import StateService

logger = get_logger("orchestration.brief")


async def build_brief(db: AsyncSession, org_id: uuid.UUID) -> DailyBrief:
    """State V2 -> Reasoning V2 -> Decision. Reused by the API and the scheduler.

    Adaptive's ACTIVE policy is resolved HERE and injected into Decision — Decision
    never imports Adaptive. This is where the double loop closes at runtime.
    """
    from navis.orchestration.runtime_policy import resolve_policy

    # incremental_update (not get_state) so the brief's as_of reflects the
    # NEWEST memory_records, not whatever snapshot happens to be cached.
    # get_state serves reads from cache and would freeze as_of at the last
    # rebuild — leaving today's brief permanently stamped "Thursday 9 Jul"
    # even after new signals land. Incremental only folds records past the
    # snapshot watermark, so cost is O(delta) not O(all-memory).
    state = await StateService(db).incremental_update(org_id)
    # Announce that State V2 is current for this org (re-homed from the retired V1
    # state service). Drives the onboarding STATE_READY milestone; State itself stays
    # a pure read model and never touches the event bus.
    await event_bus.publish(
        DomainEvent(type=EventType.STATE_UPDATED, org_id=org_id, source="state.v2",
                    payload=state.counts())
    )
    signals = run_reasoning_v2(state)
    watermark = state.watermark.last_memory_id
    policy = await resolve_policy(db, org_id)
    return await DecisionService(db).decide(
        org_id,
        signals,
        reasoning_version=REASONING_VERSION,
        state_watermark=str(watermark) if watermark else None,
        as_of=state.as_of,
        policy=policy,
    )


async def run_brief_for_org(org_id: uuid.UUID) -> dict:
    async with session_scope() as db:
        brief = await build_brief(db, org_id)
        return {
            "org_id": str(org_id),
            "brief_id": str(brief.id),
            "decisions": len(brief.decisions),
        }


async def propose_actions(db: AsyncSession, org_id: uuid.UUID) -> int:
    """Decision -> Execution bridge: turn the latest brief's decisions into
    REQUESTED ExecutionRequests (idempotent; human approval still required)."""
    from navis.decisions.service import DecisionService
    from navis.execution.proposer import propose_from_decisions
    from navis.execution.service import ExecutionService

    brief = await DecisionService(db).latest_brief(org_id)
    if brief is None:
        return 0
    proposed = propose_from_decisions(brief.decisions or [], org_id)
    svc = ExecutionService(db)
    created = 0
    for p in proposed:
        _req, was_created = await svc.request(
            p.org_id, p.decision_id, p.action_type, p.payload,
            reasoning_signal_ids=p.reasoning_signal_ids,
            reasoning_version=p.reasoning_version,
            state_watermark=p.state_watermark,
        )
        created += int(was_created)
    return created


async def _dispatch_all() -> int:
    """Fan-out: enqueue ONE per-org task per organization (keyset-paginated).

    Avoids a single task sequentially looping every org and avoids loading 1M ids
    into memory — the heavy per-org work parallelizes across the worker fleet.
    """
    from navis.identity.models import Organization

    dispatched, after, page = 0, None, 1000
    while True:
        async with session_scope() as db:
            stmt = select(Organization.id).order_by(Organization.id).limit(page)
            if after is not None:
                stmt = stmt.where(Organization.id > after)
            ids = (await db.scalars(stmt)).all()
        if not ids:
            break
        for oid in ids:
            build_brief_for_org.delay(str(oid))
        dispatched += len(ids)
        after = ids[-1]
        if len(ids) < page:
            break
    return dispatched


@celery_app.task(name="navis.orchestration.brief.build_brief_for_org")
def build_brief_for_org(org_id: str) -> dict:
    return run_async(run_brief_for_org(uuid.UUID(org_id)))


@celery_app.task(name="navis.orchestration.brief.build_all_briefs")
def build_all_briefs() -> dict:
    return {"dispatched": run_async(_dispatch_all())}
