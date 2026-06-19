"""ObservationService — append the outcome truth, announce it (with lineage).

Records reality only (no success/confidence/prediction). Every outcome is
append-only and idempotent. On record it emits OUTCOME_OBSERVED carrying the
structured lineage (reasoning_signal_ids, execution_id, decision_id, ...) so
Learning attributes outcomes to the signals that caused them — not to free-form
subject_ref prefixes.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from navis.observation.models import (
    OUTCOME_SCHEMA_VERSION,
    Outcome,
    OutcomeType,
    compute_outcome_id,
)
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger

logger = get_logger("observation")

# Back-compat transport hint for the (not-yet-rewired) Learning subscriber.
# Observation does NOT store this — it is a fact->label mapping that belongs to
# Learning. Kept in the EVENT only until Learning consumes outcome_type directly.
_SUCCESS_HINT = {
    OutcomeType.REPLY_RECEIVED.value: 1.0,
    OutcomeType.MEETING_ACCEPTED.value: 1.0,
    OutcomeType.MEETING_COMPLETED.value: 1.0,
    OutcomeType.ACTION_EXECUTED.value: 0.5,
    OutcomeType.ACTION_FAILED.value: 0.0,
    OutcomeType.NO_RESPONSE.value: 0.0,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _maybe_uuid(value) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str) and value:
        try:
            return uuid.UUID(value)
        except ValueError:
            return None
    return None


class ObservationService:
    def __init__(self, db=None, *, repo=None, clock=_utcnow, publisher=None) -> None:
        if repo is not None:
            self.repo = repo
        else:
            from navis.observation.repository import OutcomeRepository

            self.repo = OutcomeRepository(db)
        self._now = clock
        # Injectable for tests; defaults to the global event bus.
        self._publish = publisher or event_bus.publish

    async def record(
        self,
        *,
        org_id: uuid.UUID,
        source_event_id: str,
        outcome_type: OutcomeType,
        occurred_at: datetime | None = None,
        execution_id: uuid.UUID | None = None,
        decision_id: uuid.UUID | None = None,
        reasoning_signal_ids: list[str] | None = None,
        reasoning_version: int | None = None,
        state_watermark: str | None = None,
        provider_reference: dict | None = None,
        payload: dict | None = None,
    ) -> Outcome | None:
        recorded = self._now()
        values = dict(
            outcome_id=compute_outcome_id(source_event_id, outcome_type.value),
            organization_id=org_id,
            execution_id=execution_id,
            decision_id=decision_id,
            reasoning_signal_ids=reasoning_signal_ids or [],
            reasoning_version=reasoning_version,
            state_watermark=state_watermark,
            provider_reference=provider_reference or {},
            outcome_type=outcome_type.value,
            payload=payload or {},
            schema_version=OUTCOME_SCHEMA_VERSION,
            source_event_id=source_event_id,
            occurred_at=occurred_at or recorded,
            recorded_at=recorded,
        )
        row, created = await self.repo.append(values)
        if created:
            await self._emit(row)
            logger.info("outcome %s %s org=%s", outcome_type.value, row.outcome_id, org_id)
        return row if created else None

    async def record_from_execution(
        self, event: DomainEvent, outcome_type: OutcomeType, source_event_id: str
    ) -> Outcome | None:
        p = event.payload or {}
        outcome_block = p.get("outcome") or {}
        return await self.record(
            org_id=event.org_id,
            source_event_id=source_event_id,
            outcome_type=outcome_type,
            execution_id=_maybe_uuid(p.get("execution_id")),
            decision_id=_maybe_uuid(p.get("decision_id")),
            reasoning_signal_ids=p.get("reasoning_signal_ids", []),
            reasoning_version=p.get("reasoning_version"),
            state_watermark=p.get("state_watermark"),
            provider_reference=outcome_block.get("provider_reference", {}),
            payload={"action_type": p.get("action_type"), "error": p.get("error")},
        )

    async def record_from_webhook(self, webhook) -> Outcome | None:
        """Map a provider webhook to an outcome by its provider_reference, inheriting
        the lineage of the original action's outcome. Unknown references are ignored."""
        prior = await self.repo.by_provider_external_id(webhook.provider, webhook.external_id)
        if prior is None:
            logger.warning("webhook %s ref=%s matched no prior outcome", webhook.provider, webhook.external_id)
            return None
        return await self.record(
            org_id=prior.organization_id,
            source_event_id=webhook.source_event_id,
            outcome_type=webhook.outcome_type,
            occurred_at=webhook.occurred_at,
            execution_id=prior.execution_id,
            decision_id=prior.decision_id,
            reasoning_signal_ids=prior.reasoning_signal_ids,
            reasoning_version=prior.reasoning_version,
            state_watermark=prior.state_watermark,
            provider_reference=prior.provider_reference,
            payload=webhook.raw,
        )

    async def republish(self, outcome: Outcome) -> None:
        """Re-emit OUTCOME_OBSERVED for an existing outcome (replay)."""
        await self._emit(outcome)

    async def _emit(self, o: Outcome) -> None:
        await self._publish(
            DomainEvent(
                type=EventType.OUTCOME_OBSERVED,
                org_id=o.organization_id,
                source="observation",
                payload={
                    "outcome_id": str(o.outcome_id),
                    "org_id": str(o.organization_id),
                    "outcome_type": o.outcome_type,
                    "execution_id": str(o.execution_id) if o.execution_id else None,
                    "decision_id": str(o.decision_id) if o.decision_id else None,
                    # Structured lineage for Learning (the audit's required fix).
                    "reasoning_signal_ids": o.reasoning_signal_ids,
                    "reasoning_version": o.reasoning_version,
                    "state_watermark": o.state_watermark,
                    # Reality time, so Learning can record occurred_at without now().
                    "occurred_at": o.occurred_at.isoformat() if o.occurred_at else None,
                    # Back-compat for the current Learning subscriber (to be retired):
                    "subject_ref": f"execution:{o.execution_id}" if o.execution_id else None,
                    "success": _SUCCESS_HINT.get(o.outcome_type, 0.0),
                },
            )
        )
