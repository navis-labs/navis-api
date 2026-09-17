"""Reranker — reorder candidate hits by relevance, weighted by signal importance.

ONE RESPONSIBILITY -> ONE HOME: this is the SINGLE home for retrieval-stage reranking
(over `Hit`s). The provider-capability fallback `MLProvider.rerank` (generic cosine over
raw strings) is a distinct concern; keep them separate — do not add a third reranker.

Default is a deterministic lexical overlap rerank (no network); an online provider's
rerank can be substituted via the registry. Reranking only REORDERS — it never
rewrites or invents content, so it cannot fabricate facts.

IMPORTANCE
A founder inbox is ~90% newsletters and receipts, and semantic similarity happily
returns semantically-similar marketing: "What am I waiting on?" matched
"Still thinking about it?👀". Relevance alone cannot tell those apart, because
the marketing IS relevant to the words in the question — it just doesn't matter.

So relevance decides WHETHER a fact is a candidate, and importance decides WHERE
it lands among the candidates. The factor is symmetric around neutral, so a
high-importance fact is lifted and a low-importance one damped:

    factor(i) = 1 + w·(i − 0.5)        i = 0.5 → 1.0, i.e. no change at all

Symmetric rather than damping-only for a measured reason: the four calendar RSVPs
that answer "What changed since yesterday?" score 0.132–0.140 blended, just under
the 0.15 floor. A damping factor (≤1.0) pushes them further under and the query
returns nothing; lifting them clears the floor and they surface.

`w` is derived, not chosen — see IMPORTANCE_WEIGHT.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Mapping

from navis.ml.retrieval.vector_search import Hit

_TOKEN = re.compile(r"[a-z0-9]+")

# The midpoint of the importance scale. A fact scored here — or not scored at all —
# ranks exactly as it does today. Unscored MUST mean "unchanged", never "unimportant":
# a fact whose backfill hasn't run yet must not be pushed down for it.
NEUTRAL_IMPORTANCE = 0.5

# Derived from the benchmark, not picked. For each founder query we measured the
# ratio factor(high)/factor(low) needed to lift the genuinely important fact past
# the marketing currently outranking it:
#
#   What am I waiting on?          need > 1.115
#   What changed since yesterday?  need > 1.275   <- binding
#
# With ratio(w) = (1 + 0.5w) / (1 − 0.25w), w = 0.6 gives 1.53 — clears the
# measured requirement with margin while staying a nudge, not an override
# (max lift ×1.30, max damp ×0.76). Larger values scored marginally better on a
# four-query benchmark over a thin corpus; that is overfitting, so the remaining
# headroom should come from telemetry instead.
IMPORTANCE_WEIGHT = 0.6


def _overlap(query: str, text: str) -> float:
    q = set(_TOKEN.findall(query.lower()))
    d = set(_TOKEN.findall(text.lower()))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


def importance_factor(importance: float, weight: float = IMPORTANCE_WEIGHT) -> float:
    """Multiplicative, never additive.

    Cosine spans only ~0.08–0.15 across the whole candidate pool while importance
    spans 0–1, so ADDING importance would erase relevance entirely. Multiplying
    preserves relevance ordering within an importance band.
    """
    return 1.0 + weight * (importance - NEUTRAL_IMPORTANCE)


@dataclass(frozen=True, slots=True)
class RankComponents:
    """Why a hit ranked where it did — every term, kept separable."""

    memory_id: uuid.UUID
    cosine: float
    lexical: float
    relevance: float      # alpha·cosine + (1-alpha)·lexical
    importance: float     # NEUTRAL when the signal has no score
    factor: float
    final: float
    scored: bool          # False => importance was unknown, factor is exactly 1.0


def rank_scores(
    query: str,
    hits: list[Hit],
    *,
    alpha: float = 0.7,
    importance: Mapping[uuid.UUID, float] | None = None,
    importance_weight: float = IMPORTANCE_WEIGHT,
) -> list[tuple[RankComponents, Hit]]:
    """Score every hit and return the components alongside it, best first.

    This is the explainable form: `rerank` is a thin wrapper that throws the
    components away. Telemetry, evaluation and debugging use this so ranking
    never becomes a black box.
    """
    lookup = importance or {}
    out: list[tuple[RankComponents, Hit]] = []
    for h in hits:
        lexical = _overlap(query, h.source_text)
        relevance = alpha * h.score + (1 - alpha) * lexical
        raw = lookup.get(h.memory_id)
        scored = raw is not None
        value = NEUTRAL_IMPORTANCE if raw is None else raw
        factor = importance_factor(value, importance_weight)
        out.append(
            (
                RankComponents(
                    memory_id=h.memory_id,
                    cosine=h.score,
                    lexical=lexical,
                    relevance=relevance,
                    importance=value,
                    factor=factor,
                    final=relevance * factor,
                    scored=scored,
                ),
                h,
            )
        )
    out.sort(key=lambda ch: (-ch[0].final, str(ch[0].memory_id)))
    return out


def rerank(
    query: str,
    hits: list[Hit],
    *,
    alpha: float = 0.7,
    min_score: float = 0.15,
    importance: Mapping[uuid.UUID, float] | None = None,
    importance_weight: float = IMPORTANCE_WEIGHT,
) -> list[Hit]:
    """Blend vector score (alpha) with lexical overlap (1-alpha), then weight by
    signal importance. Stable order.

    Hits scoring below `min_score` are dropped: without this, top-K always
    returns something, so unrelated memories (e.g. a generic commitment whose
    embedding lifts on many queries) leak into "related knowledge" on every
    turn. The default is intentionally conservative — tune from telemetry.

    The floor is applied to the IMPORTANCE-WEIGHTED score, so it gates on "is
    this worth showing the founder", not on lexical overlap alone. With
    `importance=None` every factor is exactly 1.0 and this is the previous
    behaviour, unchanged.
    """
    return [
        h
        for components, h in rank_scores(
            query, hits, alpha=alpha,
            importance=importance, importance_weight=importance_weight,
        )
        if components.final >= min_score
    ]
