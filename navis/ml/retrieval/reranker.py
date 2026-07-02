"""Reranker — reorder candidate hits by relevance to the query.

ONE RESPONSIBILITY -> ONE HOME: this is the SINGLE home for retrieval-stage reranking
(over `Hit`s). The provider-capability fallback `MLProvider.rerank` (generic cosine over
raw strings) is a distinct concern; keep them separate — do not add a third reranker.

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


def rerank(query: str, hits: list[Hit], *, alpha: float = 0.7) -> list[Hit]:
    """Blend vector score (alpha) with lexical overlap (1-alpha). Stable order."""
    def blended(h: Hit) -> float:
        return alpha * h.score + (1 - alpha) * _overlap(query, h.source_text)

    return sorted(hits, key=lambda h: (-blended(h), str(h.memory_id)))
