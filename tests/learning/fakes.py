"""In-memory fakes for Learning V2 (append-only events + count aggregates + inbox)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from navis.learning.aggregates import Delta
from navis.learning.models import AggregateDimension
from navis.learning.patterns import PatternFact

ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")
DECISION = uuid.UUID("33333333-3333-3333-3333-333333333333")
EXECUTION = uuid.UUID("44444444-4444-4444-4444-444444444444")


class FakeLearningRepo:
    def __init__(self) -> None:
        self.events: dict[uuid.UUID, SimpleNamespace] = {}
        self.aggs: dict[tuple, SimpleNamespace] = {}  # (org, dim, key) -> agg
        self.patterns: dict[uuid.UUID, PatternFact] = {}
        self.inbox: set[tuple] = set()

    async def append_event(self, values: dict[str, Any]):
        eid = values["id"]
        if eid in self.events:
            return self.events[eid], False
        row = SimpleNamespace(created_at=datetime.now(timezone.utc), **values)
        self.events[eid] = row
        return row, True

    async def increment_aggregate(self, org_id, dimension, key, delta: Delta):
        k = (org_id, dimension, key)
        agg = self.aggs.get(k)
        if agg is None:
            agg = SimpleNamespace(
                organization_id=org_id, dimension=dimension, key=key,
                successes=0, failures=0, neutrals=0, total=0,
            )
            self.aggs[k] = agg
        agg.successes += delta.successes
        agg.failures += delta.failures
        agg.neutrals += delta.neutrals
        agg.total += delta.total

    async def upsert_pattern(self, fact: PatternFact):
        self.patterns[fact.pattern_id] = fact

    async def inbox_claim(self, consumer, event_id) -> bool:
        key = (consumer, event_id)
        if key in self.inbox:
            return False
        self.inbox.add(key)
        return True

    async def stream_events(self, org_id, *, batch=5000):
        rows = [e for e in self.events.values() if e.organization_id == org_id]
        for e in sorted(rows, key=lambda e: (e.occurred_at, str(e.id))):
            yield e

    async def aggregates(self, org_id, *, dimension=None):
        return [
            a for a in self.aggs.values()
            if a.organization_id == org_id and (dimension is None or a.dimension == dimension)
        ]

    async def signal_aggregates(self, org_id):
        return await self.aggregates(org_id, dimension=AggregateDimension.SIGNAL.value)

    async def signal_aggregates_for(self, org_id, keys):
        keyset = set(keys)
        return [
            a for a in self.aggs.values()
            if a.organization_id == org_id
            and a.dimension == AggregateDimension.SIGNAL.value
            and a.key in keyset
        ]

    async def delete_projections(self, org_id):
        self.aggs = {k: v for k, v in self.aggs.items() if k[0] != org_id}
        self.patterns = {k: v for k, v in self.patterns.items() if v.org_id != org_id}

    async def flush(self):
        pass


class FakePublisher:
    def __init__(self) -> None:
        self.events = []

    async def __call__(self, event) -> None:
        self.events.append(event)


def outcome_payload(
    *,
    org_id: uuid.UUID = ORG_A,
    outcome_id: str,
    outcome_type: str,
    signals: list[str] | None = None,
    decision_id: uuid.UUID = DECISION,
    occurred_at: str = "2026-06-01T12:00:00+00:00",
    provider_reference: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """An OUTCOME_OBSERVED payload, shaped exactly as `ObservationService._emit`
    publishes it — including the nested `payload` block that carries the cause
    fields (action_type / error / failure_kind / reason)."""
    return {
        "outcome_id": outcome_id,
        "org_id": str(org_id),
        "outcome_type": outcome_type,
        "execution_id": str(EXECUTION),
        "decision_id": str(decision_id),
        "reasoning_signal_ids": signals or ["sig-A", "sig-B"],
        "reasoning_version": 1,
        "state_watermark": "wm-9",
        "provider_reference": provider_reference if provider_reference is not None else {},
        "occurred_at": occurred_at,
        "payload": payload if payload is not None else {},
    }


def agg(repo: FakeLearningRepo, dimension: str, key: str, org=ORG_A):
    return repo.aggs.get((org, dimension, key))
