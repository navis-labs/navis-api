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
    # Guard: k<=0 retrieves nothing. A negative k must NOT become a Python negative slice
    # (which drops from the END and silently inflates recall — a broken statistic).
    top = set(retrieved_ids[:k]) if k > 0 else set()
    return round(len(top & relevant) / len(relevant), 4)


def mean_recall_at_k(results: list[tuple[list[str], frozenset[str]]], k: int) -> float:
    if not results:
        return 0.0
    return round(sum(recall_at_k(r, rel, k) for r, rel in results) / len(results), 4)
