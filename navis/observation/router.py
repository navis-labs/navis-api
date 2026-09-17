"""Observation HTTP surface — the append-only outcome log, read-only.

WHAT THIS CLOSES. `outcomes` has recorded what happened after every execution
since Observation shipped, carrying full lineage back to Execution -> Decision ->
ReasoningSignals -> State watermark. The only routes that touched Observation
were the provider WEBHOOKS, which write. So the one table that can answer "did
it work?" had no reader, and a founder surface asking that question had to
either invent an answer or show a zero.

READS, NEVER WRITES. Every handler here goes through `OutcomeRepository`, which
has exactly one write op (`append`) and is not reachable from this module. An
outcome is immutable by design — a new fact is a new row — so there is nothing
here to update even in principle.

NO INTERPRETATION. Observation stores outcome TRUTH and never what it was worth
(`observation.service` removed a success-score hint for precisely this reason).
This surface publishes the same fields and adds no label, score or verdict: what
an outcome MEANS is Learning's answer, served by `/learning/summary`.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.authz import Principal
from navis.identity.throttle import rate_limit
from navis.observation.models import Outcome
from navis.observation.repository import OutcomeRepository
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session

router = APIRouter(prefix="/outcomes", tags=["observation"])

# Page size ceiling. The log is append-only and grows with every execution, so an
# unbounded read is a read that gets slower forever.
_MAX_LIMIT = 200


def serialize_outcome(o: Outcome) -> dict[str, Any]:
    """One outcome, exactly as stored.

    Module-level so the decision-history read model publishes the identical
    block — one outcome shape in the codebase, not two that can drift.

    Every field is a FACT or a LINEAGE POINTER, the same rule
    `ObservationService._emit` follows for the event. `payload` carries the
    cause fields Observation captured (`action_type`, `error`, `failure_kind`,
    `reason`) and is passed through untouched.
    """
    return {
        "outcome_id": str(o.outcome_id),
        "outcome_type": o.outcome_type,
        # ── Lineage: WHICH action, WHAT was decided, WHY it was decided ──
        "execution_id": str(o.execution_id) if o.execution_id else None,
        "decision_id": str(o.decision_id) if o.decision_id else None,
        "reasoning_signal_ids": [str(s) for s in (o.reasoning_signal_ids or [])],
        "reasoning_version": o.reasoning_version,
        "state_watermark": o.state_watermark,
        # The external object, for correlating a provider webhook back to this.
        "provider_reference": o.provider_reference or {},
        # Reality time and ingest time are different questions and stay separate.
        "occurred_at": o.occurred_at.isoformat() if o.occurred_at else None,
        "recorded_at": o.recorded_at.isoformat() if o.recorded_at else None,
        "payload": o.payload or {},
    }


@router.get(
    "",
    dependencies=[Depends(rate_limit("read", "rate_limit_read_per_window"))],
)
async def list_outcomes(
    decision_id: uuid.UUID | None = None,
    execution_id: uuid.UUID | None = None,
    limit: int = _MAX_LIMIT,
    p: Principal = Depends(require_capability(Capability.OBSERVATION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """What actually happened, newest first.

    ORG SCOPE IS NOT A PARAMETER. `p.org_id` comes from the authenticated
    principal and is passed to every repository call; a `decision_id` or
    `execution_id` belonging to another tenant simply matches nothing, because
    the filter is applied ON TOP of the organization predicate rather than
    instead of it.

    Filtered reads use `OutcomeRepository.timeline`, which is ascending — a
    single decision's outcomes are a sequence and read forwards. The unfiltered
    feed uses `recent`, which is descending. Both orders are the repository's;
    neither is re-sorted here.
    """
    # Clamped rather than declared with `Query(ge=…, le=…)`, matching every
    # other read handler in this codebase: those are called directly in tests,
    # where a `Query` default arrives as a `Query` object instead of an int.
    limit = max(1, min(limit, _MAX_LIMIT))
    repo = OutcomeRepository(db)
    filtered = decision_id is not None or execution_id is not None
    if filtered:
        rows = await repo.timeline(
            p.org_id, execution_id=execution_id, decision_id=decision_id
        )
        rows = list(rows)[:limit]
    else:
        rows = list(await repo.recent(p.org_id, limit=limit))

    return {
        "outcomes": [serialize_outcome(o) for o in rows],
        "total": len(rows),
        # WHETHER THIS PAGE IS THE WHOLE ANSWER. A client that cannot tell a
        # full result from a truncated one will report the page size as a count
        # — the same defect `DecisionService.obligations` documents.
        "truncated": len(rows) >= limit,
        "order": "occurred_at.asc" if filtered else "occurred_at.desc",
    }
