"""BriefOrchestrator — the morning heartbeat.

Sequence (no business logic of its own):
    1. state.rebuild_for_org      (project current reality)
    2. decisions.generate_brief   (rank what matters)

Exposed both as a coroutine (for the API / tests) and a Celery task (schedule).
"""
from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import select

from navis.decisions.service import DecisionService
from navis.platform.database import session_scope
from navis.platform.monitoring import get_logger
from navis.platform.queue import celery_app
from navis.state.service import rebuild_for_org

logger = get_logger("orchestration.brief")


async def run_brief_for_org(org_id: uuid.UUID) -> dict:
    # Step 1: refresh state from memory.
    await rebuild_for_org(org_id)
    # Step 2: assemble the brief from state + reasoning.
    async with session_scope() as db:
        brief = await DecisionService(db).generate_brief(org_id)
        return {"org_id": str(org_id), "brief_id": str(brief.id), "items": len(brief.items)}


async def _run_all() -> list[dict]:
    from navis.identity.models import Organization

    async with session_scope() as db:
        org_ids = (await db.scalars(select(Organization.id))).all()

    results = []
    for org_id in org_ids:
        try:
            results.append(await run_brief_for_org(org_id))
        except Exception:  # pragma: no cover - one org failing must not block others
            logger.exception("brief failed for org=%s", org_id)
    return results


@celery_app.task(name="navis.orchestration.brief.build_all_briefs")
def build_all_briefs() -> list[dict]:
    return asyncio.run(_run_all())
