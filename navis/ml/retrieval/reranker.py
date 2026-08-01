"""Reranker — reorder candidate hits by relevance to the query.

Default is a deterministic lexical overlap rerank (no network); an online provider's
rerank can be substituted via the registry. Reranking only REORDERS — it never
rewrites or invents content, so it cannot fabricate facts.
"""
from __future__ import annotations

import re

from navis.ml.retrieval.vector_search import Hit

_TOKEN = re.compile(r"[a-z0-9]+")


def _overlap(query: str, text: str) -> float:
    q = set(_TOKEN.findall(query.lower()))
    d = set(_TOKEN.findall(text.lower()))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


def rerank(
    query: str,
    hits: list[Hit],
    *,
    alpha: float = 0.7,
    min_score: float = 0.15,
) -> list[Hit]:
    """Blend vector score (alpha) with lexical overlap (1-alpha). Stable order.

    Hits scoring below `min_score` are dropped: without this, top-K always
    returns something, so unrelated memories (e.g. a generic commitment whose
    embedding lifts on many queries) leak into "related knowledge" on every
    turn. The default is intentionally conservative — tune from telemetry.
    """
    def blended(h: Hit) -> float:
        return alpha * h.score + (1 - alpha) * _overlap(query, h.source_text)

    scored = [(blended(h), h) for h in hits]
    scored = [(s, h) for s, h in scored if s >= min_score]
    scored.sort(key=lambda sh: (-sh[0], str(sh[1].memory_id)))
    return [h for _, h in scored]
