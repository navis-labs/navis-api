"""State service: build + persist snapshots, announce StateUpdated."""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger
from navis.state.aggregate import OrganizationalState
from navis.state.builder import build_organizational_state
from navis.state.models import OrganizationalStateSnapshot

logger = get_logger("state")


async def rebuild_and_persist(db: AsyncSession, org_id: uuid.UUID) -> OrganizationalState:
    state = await build_organizational_state(db, org_id)
    snapshot = OrganizationalStateSnapshot(
        organization_id=org_id,
        built_at=state.built_at,
        summary=state.summary(),
    )
    db.add(snapshot)
    await db.flush()
    return state


async def rebuild_for_org(org_id: uuid.UUID) -> OrganizationalState:
    """Build a fresh state snapshot and emit StateUpdated for reasoning to react."""
    async with session_scope() as db:
        state = await rebuild_and_persist(db, org_id)

    await event_bus.publish(
        DomainEvent(
            type=EventType.STATE_UPDATED,
            org_id=org_id,
            source="state",
            payload=state.summary(),
        )
    )
    logger.info("state rebuilt org=%s %s", org_id, state.summary())
    return state
