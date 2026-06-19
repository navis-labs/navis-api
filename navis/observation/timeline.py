"""Outcome timeline reconstruction.

A pure projection over the immutable outcome log, ordered by occurred_at (then
created_at as a stable tiebreak). Reconstructs e.g.:

    action_executed -> reply_received -> meeting_accepted -> meeting_completed

Timelines are computed on read, so they are trivially rebuildable: drop nothing,
just read the append-only outcomes again.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    outcome_id: str
    outcome_type: str
    occurred_at: str
    execution_id: str | None
    decision_id: str | None
    provider_reference: dict[str, Any]
    payload: dict[str, Any]


async def reconstruct(
    repo,
    org_id: uuid.UUID,
    *,
    execution_id: uuid.UUID | None = None,
    decision_id: uuid.UUID | None = None,
) -> list[TimelineEvent]:
    rows = await repo.timeline(org_id, execution_id=execution_id, decision_id=decision_id)
    return [
        TimelineEvent(
            outcome_id=str(r.outcome_id),
            outcome_type=r.outcome_type,
            occurred_at=r.occurred_at.isoformat(),
            execution_id=str(r.execution_id) if r.execution_id else None,
            decision_id=str(r.decision_id) if r.decision_id else None,
            provider_reference=r.provider_reference,
            payload=r.payload,
        )
        for r in rows
    ]
