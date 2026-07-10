"""Knowledge Graph (Learning-scoped) — the causal influence structure Learning encodes.

Deterministic derivation over learning_events: which reasoning signals and decisions
co-occur with which OUTCOME TYPES, and how often. This is the semantic relationship
Learning actually supports — genuinely NEW information (the per-source outcome-type
distribution is not present in the per-dimension learning_aggregates), never a
duplication of them.

Scope (deliberate, per the Learning-only boundary): the ORGANIZATIONAL ENTITY graph —
People / Teams / Projects / Customers / Goals / Meetings and the relationships between
them — is NOT built here. Those entities live in State/Memory, which Intelligence must
never read; State already computes their cross-links (state/v2/folds.py). Representing
them here would violate both the boundary (test_no_forbidden_pipeline_imports) and
"do not duplicate State/Memory". This module represents ONLY the
signal <-> decision -> outcome influence that Learning alone provides.

Pure + deterministic: same events => same graph. No clock, uuid, random, or ML.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class InfluenceEdge:
    source_kind: str   # "signal" | "decision"
    source: str        # reasoning signal_id / decision_id
    outcome_type: str  # the outcome type it co-occurred with
    count: int         # learning events joining source -> outcome_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source": self.source,
            "outcome_type": self.outcome_type,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class InfluenceGraph:
    edges: tuple[InfluenceEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"edges": [e.to_dict() for e in self.edges]}


def build_influence_graph(events: Iterable[Any]) -> InfluenceGraph:
    """learning_events -> deterministic signal/decision -> outcome-type influence edges.

    `events` are rows exposing `.reasoning_signal_ids`, `.decision_id`, `.outcome_type`.
    Edges are counted by co-occurrence and emitted in a total, deterministic order, so
    the same events always produce the same graph (replay-safe)."""
    counts: dict[tuple[str, str, str], int] = {}
    for e in events:
        otype = e.outcome_type or ""
        if not otype:
            continue
        for sid in (e.reasoning_signal_ids or []):
            key = ("signal", str(sid), otype)
            counts[key] = counts.get(key, 0) + 1
        if e.decision_id is not None:
            key = ("decision", str(e.decision_id), otype)
            counts[key] = counts.get(key, 0) + 1
    edges = tuple(
        InfluenceEdge(kind, source, otype, n)
        for (kind, source, otype), n in sorted(counts.items())
    )
    return InfluenceGraph(edges=edges)
