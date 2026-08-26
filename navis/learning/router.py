"""Learning HTTP surface — read-only, and it reads the contract, not the tables.

WHAT THIS CLOSES. `navis.learning.metrics` is documented as "THE LEARNING READ
CONTRACT — the only supported way to consume Learning", and Intelligence and
Adaptive both consume it in-process. Nothing exposed it over HTTP, so a founder
surface asking "what has Navis learned?" had no answer available and the
Decisions page filled the gap with zeros — `Learnings stored: 0`,
`Average impact: +0%` — while `learning_events` held real rows.

NO NEW DERIVATION LIVES HERE. Every number below is either a stored integer or
`LearningSnapshot`'s own derivation of one. `success_rate` is Learning's
(`aggregates.success_rate`), the delivery/judgement/world grouping is Learning's
(`outcomes_by_surface`), and the totals are Learning's (`LearningSnapshot
.totals`). A second opinion computed at the edge is exactly the drift the read
contract exists to prevent.

NULL IS A MEASUREMENT AND ZERO IS NOT. A rate over zero decisive samples is
reported as `null`, never as `0.0`. The difference is the difference between
"nothing has been decided yet" and "everything Navis tried failed", and a
surface cannot recover the first from the second once this layer has flattened
it.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.authz import Principal
from navis.identity.throttle import rate_limit
from navis.learning.metrics import AggregateFact, LearningSnapshot, learning_snapshot
from navis.learning.models import AggregateDimension
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session

router = APIRouter(prefix="/learning", tags=["learning"])


def _aggregate(a: AggregateFact) -> dict[str, Any]:
    """One counter, with the two values Learning derives from it.

    `decisive` and `success_rate` are properties on `AggregateFact`; publishing
    them means a client never has to know that neutrals are excluded from a
    rate, which is a rule that belongs to Learning and not to a renderer.
    """
    return {
        "dimension": a.dimension,
        "key": a.key,
        "successes": a.successes,
        "failures": a.failures,
        "neutrals": a.neutrals,
        "total": a.total,
        "decisive": a.decisive,
        "success_rate": a.rate,
    }


def serialize_snapshot(snapshot: LearningSnapshot) -> dict[str, Any]:
    """The wire shape of one organization's learning state.

    Module-level and not nested in the handler so the decision-history read
    model can publish the same block without a second serializer — one shape,
    one place, no drift between two surfaces reporting one set of counts.
    """
    totals = snapshot.totals()
    return {
        "org_id": str(snapshot.org_id),
        # Monotonic event count. Consumers use it as a version key, which is why
        # Learning keeps it a count rather than a timestamp.
        "learning_version": snapshot.learning_version,
        # Logical time — the newest occurred_at in the log, never wall clock.
        "as_of": snapshot.as_of.isoformat() if snapshot.as_of else None,
        "state_watermark": snapshot.state_watermark,
        "reasoning_version": snapshot.reasoning_version,
        "totals": {
            "learning_events": totals.events,
            "successes": totals.successes,
            "failures": totals.failures,
            "neutrals": totals.neutrals,
            "decisive": totals.decisive,
            # None until something decisive has happened. See the module note.
            "success_rate": totals.success_rate,
        },
        # Delivery / judgement / world, grouped by Learning's own classifier. A
        # surface absent from this map has recorded no outcome of that kind —
        # which is a different fact from a surface holding a zero.
        "by_surface": {
            surface: [_aggregate(a) for a in aggs]
            for surface, aggs in snapshot.outcomes_by_surface().items()
        },
        # Per-outcome-type counters. The only TYPE-STABLE keys Learning has, so
        # the only ones a client can reason about by name.
        "by_outcome_type": [
            _aggregate(a) for a in snapshot.by_dimension(AggregateDimension.OUTCOME)
        ],
        # Per-reasoning-signal counters — the stable episode key (see
        # `orchestration.decision_history`). Decision-dimension counters are
        # deliberately NOT published: `metrics._has_enough_evidence` already
        # drops the ones below Learning's own evidence threshold, so the set
        # here would be a partial list that looks complete.
        "by_signal": [
            _aggregate(a) for a in snapshot.by_dimension(AggregateDimension.SIGNAL)
        ],
        # Derived facts, not recommendations: "this signal kind is reliable".
        # Empty until a signal has MIN_SAMPLES decisive outcomes.
        "patterns": [
            {
                "signal_id": p.signal_id,
                "kind": p.kind,
                "success_rate": p.success_rate,
                "samples": p.samples,
            }
            for p in snapshot.patterns
        ],
    }


@router.get(
    "/summary",
    dependencies=[Depends(rate_limit("read", "rate_limit_read_per_window"))],
)
async def learning_summary(
    p: Principal = Depends(require_capability(Capability.LEARNING)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Everything Learning is willing to say about this organization.

    Org-scoped by construction: `learning_snapshot` filters every one of its
    statements on the principal's `org_id` and the snapshot carries it, so this
    handler cannot widen the scope even by mistake. No `user_id` or `org_id`
    query parameter is accepted — the tenant comes from the authenticated
    principal and nowhere else.
    """
    return serialize_snapshot(await learning_snapshot(db, p.org_id))
