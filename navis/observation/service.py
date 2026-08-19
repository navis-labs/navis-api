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

# `_SUCCESS_HINT` lived here and has been REMOVED, along with the `success` and
# `subject_ref` fields it fed into OUTCOME_OBSERVED.
#
# It was a fact->label mapping — outcome_type -> a 0.0/0.5/1.0 score — described
# in its own comment as back-compat for a "not-yet-rewired" Learning subscriber.
# Learning was rewired: `learning.attribution.classify` maps outcome_type to a
# LearningResult itself, and nothing anywhere reads `success` or `subject_ref`
# (verified across navis/ and tests/). They were dead fields.
#
# Dead is not the reason to delete them; MISLEADING is. Observation's one law is
# that it records what happened and never what it was worth — the outcome table
# is asserted to carry no success/confidence/score column. Emitting a success
# SCORE in the event broke that law on the way out, and the two mappings had
# already drifted: this table scored ACTION_EXECUTED 0.5 while Learning's
# classifier calls it NEUTRAL. A second, stale opinion about what an outcome
# means is exactly what the boundary exists to prevent.


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
        # WHAT THE PAYLOAD CARRIES, and why each field is here.
        #
        # `failure_kind` is the stable classification Execution computed at the
        # point of failure (execution/contract.py::FailureKind). Without it every
        # failure was one undifferentiated `action_failed` and the only way to
        # tell "the org lost its entitlement" from "Gmail was down" was to sniff
        # substrings out of `error`.
        #
        # `reason` is the human's own words when they rejected a proposal. It is
        # the single most informative field Observation can hold — a person
        # saying why they said no — and it was previously discarded with the rest
        # of the rejection event.
        #
        # Keys are omitted rather than stored as None so a reader can distinguish
        # "not applicable to this outcome type" from "applicable but absent".
        payload: dict = {"action_type": p.get("action_type")}
        for key in ("error", "failure_kind", "reason"):
            value = p.get(key)
            if value is not None:
                payload[key] = value
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
            payload=payload,
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
        """THE OBSERVATION CONTRACT — the stable shape downstream may consume.

        Every field is a FACT or a LINEAGE POINTER. There is deliberately no
        score, label, confidence or judgement: what an outcome is worth is
        Learning's question, and answering it here would put a second opinion in
        the system that could drift from Learning's (which is exactly what the
        removed `success` hint had already done).

            outcome_id            deterministic; the consumer's dedupe key
            org_id                the owning organization, from the persisted row
            outcome_type          WHAT happened, from the OutcomeType vocabulary
            execution_id          WHICH action this is about
            decision_id           WHAT Navis decided
            reasoning_signal_ids  WHY it decided that
            reasoning_version     which reasoning produced it
            state_watermark       the world state it was decided against
            provider_reference    the external object, for correlating webhooks
            occurred_at           reality time, so a consumer never calls now()
            payload               action_type, and cause fields where they apply
                                  (error / failure_kind / reason)

        `provider_reference` and `payload` are new to this event. Both were
        already ON the outcome row and simply were not published, so a consumer
        had to re-read the database to answer "which message was that?" or "why
        did it fail?" — and Learning is explicitly forbidden from reading any
        other module's tables.
        """
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
                    "provider_reference": o.provider_reference or {},
                    # Reality time, so Learning can record occurred_at without now().
                    "occurred_at": o.occurred_at.isoformat() if o.occurred_at else None,
                    "payload": o.payload or {},
                },
            )
        )
