"""In-memory fakes for Intelligence V2 (append-only insights over a LearningView)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from navis.intelligence.insights import (
    AggRow,
    ContextEntityRow,
    ContextLink,
    LearningView,
    OrganizationalRealityView,
    PatternRow,
    RealitySignalRow,
)

ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")
AS_OF = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def reality_signal(signal_id: str = "sig-A", **over) -> RealitySignalRow:
    """One flattened ReasoningSignal, primitives only."""
    fields: dict[str, Any] = {
        "signal_id": signal_id,
        "signal_type": "commitment_slipping",
        "entity_type": "commitment",
        "entity_id": "ent-1",
        "confidence": 0.9,
        "severity": 0.7,
        "engine_name": "commitment",
        "engine_version": "1",
        "evidence": {"thread_id": "t-1"},
    }
    fields.update(over)
    return RealitySignalRow(**fields)


def make_reality(
    *,
    org_id: uuid.UUID = ORG_A,
    as_of: datetime | None = AS_OF,
    state_watermark: str | None = "wm-state-1",
    entity_counts: dict[str, int] | None = None,
    signals: tuple[RealitySignalRow, ...] | None = None,
    reasoning_version: int = 4,
    context: tuple[ContextEntityRow, ...] = (),
) -> OrganizationalRealityView:
    return OrganizationalRealityView(
        org_id=org_id,
        as_of=as_of,
        state_watermark=state_watermark,
        reasoning_version=reasoning_version,
        entity_counts=entity_counts if entity_counts is not None else {"person": 3},
        signals=signals if signals is not None else (reality_signal(),),
        context=context,
    )


def context_row(**over) -> ContextEntityRow:
    """One entity CONTEXT selected, with the citations it carries."""
    fields: dict[str, Any] = {
        "entity_type": "commitment",
        "entity_id": "ent-1",
        "rank": 0,
        "reason": "open, oldest activity",
        "memory_ids": ("mem-1", "mem-2"),
        "signal_event_ids": ("sig-evt-1",),
    }
    fields.update(over)
    return ContextEntityRow(**fields)


def rich_reality(org_id: uuid.UUID = ORG_A, **over) -> OrganizationalRealityView:
    """A reality view that exercises every reality engine that can fire.

    Shapes mirror what the real reasoning engines emit — same signal_type values,
    same evidence keys — so a change in either side shows up as a test failure
    rather than as a silently empty insight in production.
    """
    signals = (
        reality_signal(
            "sig-cool-1", signal_type="relationship_cooling",
            entity_type="relationship", entity_id="rel-1",
            confidence=0.75, severity=0.6,
            engine_name="relationship_reasoning", engine_version="2.0.0",
            evidence={"days_since_contact": 9.0, "person_email": "amy@acme.com"},
        ),
        reality_signal(
            "sig-cool-2", signal_type="relationship_cooling",
            entity_type="relationship", entity_id="rel-2",
            confidence=0.89, severity=0.6,
            engine_name="relationship_reasoning", engine_version="2.0.0",
            evidence={"days_since_contact": 21.0, "person_email": "bo@beta.io"},
        ),
        reality_signal(
            "sig-commit-1", signal_type="commitment_slipping",
            entity_type="commitment", entity_id="com-1",
            confidence=0.7, severity=0.6,
            engine_name="commitment_reasoning", engine_version="2.0.0",
            evidence={"age_days": 12.0, "speaker_email": "amy@acme.com"},
        ),
        reality_signal(
            "sig-followup-1", signal_type="commitment_follow_up",
            entity_type="commitment", entity_id="com-2",
            confidence=0.85, severity=0.7,
            engine_name="commitment_follow_up", engine_version="1.0.0",
            evidence={
                "person_email": "cara@gamma.dev", "commitment": "send the deck",
                "due_at": "2026-05-20T00:00:00+00:00", "due_text": "Friday",
                "overdue_days": 11.5, "thread_id": "t-9",
            },
        ),
        reality_signal(
            "sig-opp-1", signal_type="opportunity",
            entity_type="relationship", entity_id="rel-3",
            confidence=0.6, severity=0.5,
            engine_name="opportunity", engine_version="2.1.0",
            evidence={"days_since": 1.0, "person_email": "dev@delta.co",
                      "relevance_evidence": ["human_reply"]},
        ),
        reality_signal(
            "sig-lev-1", signal_type="leverage_point",
            entity_type="company", entity_id="co-1",
            confidence=0.65, severity=0.4,
            engine_name="leverage", engine_version="2.0.0",
            evidence={"people": 4, "domain": "acme.com"},
        ),
        reality_signal(
            "sig-load-1", signal_type="priority_load",
            entity_type=None, entity_id=None,
            confidence=0.7, severity=0.7,
            engine_name="priority", engine_version="2.0.0",
            evidence={"slipping": 2, "cooling": 2, "stalled": 0},
        ),
    )
    fields = {
        "org_id": org_id,
        "entity_counts": {"person": 18, "company": 12, "relationship": 16,
                          "commitment": 2, "thread": 34, "meeting": 3},
        "signals": signals,
    }
    fields.update(over)
    return make_reality(**fields)


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
    state_watermark: str | None = None,
    signal_lineage: dict[str, list[str]] | None = None,
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
        state_watermark=state_watermark,
        signal_lineage=signal_lineage or {},
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

    async def get_insights(self, org_id: uuid.UUID, *, include_superseded: bool = False):
        """Mirrors the real read boundary: current projections only by default.

        Calls the SAME `current_insights` the repository does rather than
        reimplementing the rule — a double that picks rows by its own logic would
        only ever confirm the test author's assumption about which one is newest.
        """
        from navis.intelligence.repository import current_insights

        rows = [r for r in self.insights.values() if r.organization_id == org_id]
        if include_superseded:
            return sorted(rows, key=lambda r: (r.insight_type, r.key))
        return current_insights(rows)

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


def context_link(**over) -> ContextLink:
    """One persisted edge on a context entity."""
    fields: dict[str, Any] = {
        "role": "commitments",
        "entity_type": "commitment",
        "entity_id": "com-1",
    }
    fields.update(over)
    return ContextLink(**fields)


def connected_counterparty_context(
    *, rel_id: str = "rel-1", com_id: str = "com-1",
    person_id: str = "per-1", meeting_ids: tuple[str, ...] = ("mtg-1", "mtg-2"),
) -> tuple[ContextEntityRow, ...]:
    """THE AJITESH SHAPE — one counterparty, the edges State really persists.

    Mirrors the production graph the audit found on org 071ae55b verbatim:

        commitment   --owner---------> person
        person       --relationships-> relationship
        relationship --commitments---> commitment
        relationship --meetings------> meeting, meeting
    """
    return (
        context_row(
            entity_type="commitment", entity_id=com_id, rank=0,
            reason="open, oldest activity",
            links=(context_link(role="owner", entity_type="person", entity_id=person_id),),
        ),
        context_row(
            entity_type="relationship", entity_id=rel_id, rank=1,
            reason="stalest interaction",
            memory_ids=("mem-rel-1",), signal_event_ids=("sig-evt-rel-1",),
            links=(
                context_link(role="commitments", entity_type="commitment", entity_id=com_id),
                *(
                    context_link(role="meetings", entity_type="meeting", entity_id=m)
                    for m in meeting_ids
                ),
            ),
        ),
        context_row(
            entity_type="person", entity_id=person_id, rank=2,
            reason="most recently seen",
            memory_ids=("mem-per-1",), signal_event_ids=(),
            links=(
                context_link(
                    role="relationships", entity_type="relationship", entity_id=rel_id
                ),
            ),
        ),
    )
