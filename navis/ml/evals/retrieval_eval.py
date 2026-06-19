"""Retrieval quality: recall@k over a labeled golden set (deterministic)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    query: str
    relevant_ids: frozenset[str]


def recall_at_k(retrieved_ids: list[str], relevant: frozenset[str], k: int) -> float:
    if not relevant:
        return 1.0
    top = set(retrieved_ids[:k])
    return round(len(top & relevant) / len(relevant), 4)


def mean_recall_at_k(results: list[tuple[list[str], frozenset[str]]], k: int) -> float:
    if not results:
        return 0.0
    return round(sum(recall_at_k(r, rel, k) for r, rel in results) / len(results), 4)
