"""In-memory fakes for Intelligence V2 (append-only insights over a LearningView)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from navis.intelligence.insights import AggRow, LearningView, PatternRow

ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")
AS_OF = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def make_view(
    *,
    org_id: uuid.UUID = ORG_A,
    learning_version: int = 12,
    as_of: datetime | None = AS_OF,
    signal_aggs: list[AggRow] | None = None,
    decision_aggs: list[AggRow] | None = None,
    outcome_aggs: list[AggRow] | None = None,
    patterns: list[PatternRow] | None = None,
    lineage: dict[str, list[str]] | None = None,
) -> LearningView:
    return LearningView(
        org_id=org_id,
        learning_version=learning_version,
        as_of=as_of,
        signal_aggs=signal_aggs or [],
        decision_aggs=decision_aggs or [],
        outcome_aggs=outcome_aggs or [],
        patterns=patterns or [],
        lineage=lineage or {},
    )


def rich_view(org_id: uuid.UUID = ORG_A, learning_version: int = 30) -> LearningView:
    """A view that exercises all six generators."""
    return make_view(
        org_id=org_id,
        learning_version=learning_version,
        signal_aggs=[
            AggRow("sig-A", successes=8, failures=2, neutrals=1, total=11),
            AggRow("sig-B", successes=3, failures=7, neutrals=0, total=10),
            AggRow("sig-thin", successes=1, failures=1, neutrals=0, total=2),  # below MIN
        ],
        decision_aggs=[AggRow("dec-1", successes=6, failures=4, neutrals=0, total=10)],
        outcome_aggs=[
            AggRow("reply_received", 0, 0, 0, 7),
            AggRow("meeting_accepted", 0, 0, 0, 3),
            AggRow("action_executed", 0, 0, 0, 20),
            AggRow("action_failed", 0, 0, 0, 5),
            AggRow("no_response", 0, 0, 0, 8),
        ],
        patterns=[
            PatternRow("sig-A", "reliable", 0.8, 11),
            PatternRow("sig-B", "noisy", 0.3, 10),
        ],
        lineage={
            "sig-A": ["e1", "e2", "e3"],
            "sig-B": ["e4"],
            "dec-1": ["e1", "e4"],
        },
    )


class FakeIntelligenceRepo:
    """Holds a fixed LearningView (the immutable upstream) + append-only insights."""

    def __init__(self, view: LearningView) -> None:
        self.view = view
        self.insights: dict[uuid.UUID, SimpleNamespace] = {}

    async def build_view(self, org_id: uuid.UUID) -> LearningView:
        assert org_id == self.view.org_id
        return self.view

    async def append_insight(self, values: dict[str, Any]) -> bool:
        iid = values["insight_id"]
        if iid in self.insights:
            return False  # idempotent: same deterministic id collapses
        self.insights[iid] = SimpleNamespace(**values)
        return True

    async def get_insights(self, org_id: uuid.UUID):
        rows = [r for r in self.insights.values() if r.organization_id == org_id]
        return sorted(rows, key=lambda r: (r.insight_type, r.key))

    async def delete_org_insights(self, org_id: uuid.UUID) -> None:
        self.insights = {
            k: v for k, v in self.insights.items() if v.organization_id != org_id
        }

    async def flush(self) -> None:
        pass


def snapshot(rows) -> set[tuple]:
    """Comparable, order-independent fingerprint of a set of insight rows."""
    return {
        (
            str(r["insight_id"]),
            r["insight_type"],
            r["key"],
            r["title"],
            r["summary"],
            r["sample_size"],
            r["confidence"],
            tuple(r["supporting_learning_event_ids"]),
            r["learning_version"],
            r["created_at"],
        )
        for r in rows
    }
