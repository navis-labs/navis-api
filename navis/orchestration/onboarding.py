"""OnboardingOrchestrator — first-run sequence for a new org.

    1. backfill recent signals from connected integrations
    2. build the first OrganizationalState
    3. generate the first brief (the "holy shit" moment)
"""
from __future__ import annotations

import uuid

from navis.orchestration.brief import run_brief_for_org
from navis.orchestration.sync import sync_integration
from navis.platform.database import session_scope
from navis.platform.monitoring import get_logger

logger = get_logger("orchestration.onboarding")


async def onboard_org(org_id: uuid.UUID) -> dict:
    from sqlalchemy import select

    from navis.identity.models import Integration

    async with session_scope() as db:
        integrations = (
            await db.scalars(
                select(Integration).where(
                    Integration.organization_id == org_id,
                    Integration.is_active.is_(True),
                )
            )
        ).all()

    from navis.orchestration.sync import ensure_watch_for_integration

    ingested = 0
    for integ in integrations:
        result = await sync_integration(integ)
        ingested += result.get("ingested", 0)
        # Start the push channel so future changes arrive in real time.
        try:
            await ensure_watch_for_integration(integ)
        except Exception:
            logger.exception("watch creation failed during onboarding integration=%s", integ.id)

    brief = await run_brief_for_org(org_id)
    logger.info("onboarded org=%s ingested=%d", org_id, ingested)
    return {"ingested": ingested, **brief}
