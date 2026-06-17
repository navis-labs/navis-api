"""Learning service: turn observed outcomes into updated confidence weights.

Subscribes to OUTCOME_OBSERVED and maintains a running success-rate per signal
kind. Reasoning/decisions can later read ConfidenceWeight to scale a raw signal's
confidence by how well that kind of alert has historically panned out.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select

from navis.learning.models import ConfidenceWeight, LearningEvent
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, event_bus, subscribe
from navis.platform.monitoring import get_logger

logger = get_logger("learning")

# Exponential-moving-average factor: how fast new outcomes move the dial.
_ALPHA = 0.2


def _kind_from_ref(subject_ref: str | None) -> str:
    if not subject_ref:
        return "unknown"
    return subject_ref.split(":", 1)[0]


async def record_outcome(org_id: uuid.UUID, signal_kind: str, success: float) -> float:
    async with session_scope() as db:
        db.add(
            LearningEvent(organization_id=org_id, signal_kind=signal_kind, success=success)
        )
        weight = await db.scalar(
            select(ConfidenceWeight).where(
                ConfidenceWeight.organization_id == org_id,
                ConfidenceWeight.signal_kind == signal_kind,
            )
        )
        if weight is None:
            weight = ConfidenceWeight(
                organization_id=org_id, signal_kind=signal_kind, samples=0, success_rate=0.5
            )
            db.add(weight)
        weight.samples += 1
        weight.success_rate = (1 - _ALPHA) * weight.success_rate + _ALPHA * success
        new_rate = weight.success_rate

    await event_bus.publish(
        DomainEvent(
            type=EventType.LEARNING_UPDATED,
            org_id=org_id,
            source="learning",
            payload={"signal_kind": signal_kind, "success_rate": new_rate},
        )
    )
    logger.info("learning org=%s kind=%s rate=%.3f", org_id, signal_kind, new_rate)
    return new_rate


@subscribe(EventType.OUTCOME_OBSERVED)
async def on_outcome_observed(event: DomainEvent) -> None:
    if event.org_id is None:
        return
    signal_kind = _kind_from_ref(event.payload.get("subject_ref"))
    success = float(event.payload.get("success", 0.0))
    await record_outcome(event.org_id, signal_kind, success)
