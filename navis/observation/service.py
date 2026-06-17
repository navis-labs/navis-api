"""Record outcomes and announce them for learning."""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from navis.observation.models import Outcome, OutcomeKind
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger

logger = get_logger("observation")


class ObservationService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def record(
        self,
        org_id: uuid.UUID,
        kind: OutcomeKind,
        subject_ref: str | None,
        success: float,
        detail: dict | None = None,
    ) -> Outcome:
        outcome = Outcome(
            organization_id=org_id,
            kind=kind,
            subject_ref=subject_ref,
            success=success,
            detail=detail or {},
        )
        self.db.add(outcome)
        await self.db.flush()
        await event_bus.publish(
            DomainEvent(
                type=EventType.OUTCOME_OBSERVED,
                org_id=org_id,
                source="observation",
                payload={
                    "outcome_id": str(outcome.id),
                    "kind": kind.value,
                    "subject_ref": subject_ref,
                    "success": success,
                },
            )
        )
        logger.info("outcome recorded org=%s kind=%s success=%.2f", org_id, kind.value, success)
        return outcome
