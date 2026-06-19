"""Read-only learning metrics — what actually works, per dimension.

Pure derivations over the count aggregates (no recomputation of intelligence).
Reasoning/Intelligence may read these later; Learning only exposes facts.
"""
from __future__ import annotations

import uuid

from navis.learning.aggregates import success_rate
from navis.learning.models import AggregateDimension
from navis.learning.repository import LearningRepository


async def success_rates(
    db, org_id: uuid.UUID, dimension: AggregateDimension
) -> dict[str, dict]:
    """{key: {successes, failures, neutrals, total, success_rate}} for a dimension."""
    repo = LearningRepository(db)
    rows = await repo.aggregates(org_id, dimension=dimension.value)
    return {
        r.key: {
            "successes": r.successes,
            "failures": r.failures,
            "neutrals": r.neutrals,
            "total": r.total,
            "success_rate": success_rate(r.successes, r.failures),
        }
        for r in rows
    }
