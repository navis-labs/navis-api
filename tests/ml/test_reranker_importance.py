"""Importance-weighted reranking (Phase 2A).

Relevance decides whether a fact is a candidate; importance decides where it
lands among the candidates. The weight is derived from the benchmark, not chosen
— see `IMPORTANCE_WEIGHT` — and these tests pin the properties that derivation
depends on.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from navis.ml.retrieval.reranker import (
    IMPORTANCE_WEIGHT,
    NEUTRAL_IMPORTANCE,
    importance_factor,
    rank_scores,
    rerank,
)
from navis.ml.retrieval.vector_search import Hit

NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _the_faked_index_has_rows(monkeypatch):
    """The retriever test below fakes its I/O and models a POPULATED index, which
    `MemoryRetriever` now confirms before it embeds (`index_has_rows`)."""
    import navis.ml.retrieval.retriever as mod

    async def _has_rows(db, org_id, *, model, version):
        return True

    monkeypatch.setattr(mod, "index_has_rows", _has_rows)


def hit(text: str, cosine: float, entity_type: str = "thread") -> Hit:
    return Hit(uuid.uuid4(), cosine, entity_type, text, NOW)


# ── the factor ─────────────────────────────────────────────────────────────


def test_neutral_importance_changes_nothing():
    """The property everything else rests on: 0.5 must be a true no-op."""
    assert importance_factor(NEUTRAL_IMPORTANCE) == 1.0


def test_factor_is_symmetric_lifting_and_damping():
    assert importance_factor(1.0) > 1.0   # lifts
    assert importance_factor(0.0) < 1.0   # damps
    assert importance_factor(1.0) == pytest.approx(1 + IMPORTANCE_WEIGHT * 0.5)


def test_factor_clears_the_measured_requirement():
    """The benchmark needed ratio(1.00 vs 0.25) > 1.275 to promote the real facts."""
    assert importance_factor(1.0) / importance_factor(0.25) > 1.275


def test_factor_is_a_nudge_not_an_override():
    """Importance must not be able to invert a large relevance gap."""
    strong_but_boring = 0.40 * importance_factor(0.0)
    weak_but_important = 0.20 * importance_factor(1.0)
    assert strong_but_boring > weak_but_important


# ── backwards compatibility ────────────────────────────────────────────────


def test_without_importance_behaviour_is_unchanged():
    hits = [hit("alpha beta", 0.5), hit("gamma", 0.4), hit("beta", 0.45)]
    assert rerank("alpha beta", hits) == rerank("alpha beta", hits, importance={})


def test_unscored_facts_are_treated_as_neutral_not_unimportant():
    """A fact whose backfill hasn't run must not be pushed down for it."""
    a, b = hit("quarterly pricing review", 0.50), hit("quarterly pricing review", 0.50)
    with_partial = rank_scores("pricing", [a, b], importance={a.memory_id: 0.5})
    by_id = {c.memory_id: c for c, _ in with_partial}
    assert by_id[a.memory_id].final == pytest.approx(by_id[b.memory_id].final)
    assert by_id[a.memory_id].scored is True
    assert by_id[b.memory_id].scored is False


# ── ordering ───────────────────────────────────────────────────────────────


def test_importance_promotes_a_fact_past_a_near_tie():
    """The measured failure: marketing narrowly outranking the real commitment."""
    marketing = hit("Still thinking about it?", 0.30)
    commitment = hit("Please review the Q3 forecast by Friday", 0.28)
    imp = {marketing.memory_id: 0.10, commitment.memory_id: 1.00}
    assert rerank("what am I waiting on", [marketing, commitment])[0] is marketing
    assert rerank("what am I waiting on", [marketing, commitment], importance=imp)[0] is commitment


def test_ordering_is_deterministic_and_stable():
    hits = [hit(f"fact {i}", 0.3) for i in range(6)]
    imp = {h.memory_id: 0.5 for h in hits}
    assert rerank("fact", hits, importance=imp) == rerank("fact", hits, importance=imp)


# ── the gate ───────────────────────────────────────────────────────────────


def test_gate_applies_to_the_weighted_score_so_importance_can_rescue():
    """Calendar RSVPs scored 0.132-0.140 blended — under the 0.15 floor. Lifting
    them is the whole reason the factor is symmetric rather than damping-only."""
    borderline = hit("Accepted: partnership @ Wed Aug 5", 0.20)
    assert rerank("what changed", [borderline]) == []
    kept = rerank("what changed", [borderline], importance={borderline.memory_id: 0.90})
    assert kept == [borderline]


def test_low_importance_can_be_damped_below_the_gate():
    marketing = hit("Monsoon Madness Ends Today", 0.222)
    assert rerank("what changed", [marketing]) == [marketing]
    assert rerank("what changed", [marketing], importance={marketing.memory_id: 0.0}) == []


def test_spam_is_pushed_down_but_the_gate_is_what_removes_it():
    spam = hit("Pending Action: Survey Assigned", 0.30)
    strong = rank_scores("survey", [spam], importance={spam.memory_id: 0.0})[0][0]
    plain = rank_scores("survey", [spam])[0][0]
    assert strong.final < plain.final


# ── explainability ─────────────────────────────────────────────────────────


def test_rank_scores_exposes_every_term():
    h = hit("Q3 pricing deck", 0.40)
    components, returned = rank_scores("pricing", [h], importance={h.memory_id: 0.9})[0]
    assert returned is h
    assert components.cosine == 0.40
    assert 0.0 <= components.lexical <= 1.0
    assert components.relevance == pytest.approx(0.7 * 0.40 + 0.3 * components.lexical)
    assert components.importance == 0.9
    assert components.factor == pytest.approx(importance_factor(0.9))
    assert components.final == pytest.approx(components.relevance * components.factor)
    assert components.scored is True


def test_rerank_is_rank_scores_plus_the_gate():
    hits = [hit("a b c", 0.5), hit("d", 0.05)]
    imp = {h.memory_id: 0.5 for h in hits}
    kept = rerank("a b c", hits, importance=imp)
    scored = [h for c, h in rank_scores("a b c", hits, importance=imp) if c.final >= 0.15]
    assert kept == scored


def test_reranking_never_invents_or_rewrites_a_fact():
    hits = [hit("original text", 0.5)]
    out = rerank("text", hits, importance={hits[0].memory_id: 1.0})
    assert out[0].source_text == "original text"
    assert out[0].memory_id == hits[0].memory_id


# ── candidate selection (Phase 2B) ─────────────────────────────────────────


async def test_retriever_asks_for_a_wide_enough_candidate_pool():
    """The reranker can only order what it receives. k*3 was the truncation that
    kept the decision facts (cosine ranks 24/35/36) out of the answer entirely."""
    import navis.ml.retrieval.retriever as retriever_mod
    from navis.ml.retrieval.retriever import CANDIDATE_MULTIPLIER, MemoryRetriever
    from navis.ml.embeddings.pins import EmbeddingPin
    from navis.ml.providers.deterministic import DeterministicProvider

    assert CANDIDATE_MULTIPLIER >= 10, "narrower pools measurably lose important facts"

    seen = {}

    async def fake_search(db, org_id, qvec, *, model, version, k, since, until):
        seen["k"] = k
        return []

    async def fake_lineage(db, _org_id, ids):
        return {}

    original_search = retriever_mod.search
    original_lineage = retriever_mod.lineage_by_memory_id
    retriever_mod.search = fake_search
    retriever_mod.lineage_by_memory_id = fake_lineage
    try:
        pin = EmbeddingPin(DeterministicProvider(), "navis-hash-256", 1)
        await MemoryRetriever(None, pin=pin).search(uuid.uuid4(), "what am I waiting on", k=8)
    finally:
        retriever_mod.search = original_search
        retriever_mod.lineage_by_memory_id = original_lineage

    assert seen["k"] == 8 * CANDIDATE_MULTIPLIER


def test_widening_cannot_displace_a_better_ranked_fact():
    """A deeper pool must only ADD options — never reorder what already won."""
    top = hit("Decision: Q3 launch postponed", 0.40)
    deep = [hit(f"newsletter {i}", 0.20) for i in range(40)]
    imp = {top.memory_id: 1.0, **{h.memory_id: 0.1 for h in deep}}
    narrow = rerank("what did we decide", [top] + deep[:5], importance=imp)
    wide = rerank("what did we decide", [top] + deep, importance=imp)
    assert narrow[0] is top and wide[0] is top
