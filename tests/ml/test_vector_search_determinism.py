"""P0-3: candidate retrieval must be deterministic and complete.

The original defect was `SELECT … LIMIT 5000` with no ORDER BY. SQL gives no
order guarantee without one, so above the limit Postgres returned an arbitrary
subset — free to differ between identical calls, and free to omit the best
match entirely.

These tests drive `search()` against a fake session that reproduces real SQL
semantics: it honours ORDER BY / WHERE / LIMIT, and — crucially — returns rows
in a HOSTILE order when no ORDER BY is present, which is exactly what a planner
is entitled to do.
"""
from __future__ import annotations

import uuid


from navis.ml.retrieval.vector_search import (
    CANDIDATE_PAGE_SIZE,
    MAX_CANDIDATES,
    Hit,
    search,
)


class FakeRow:
    def __init__(self, emb, event_timestamp=None):
        self.MemoryEmbedding = emb
        self.event_timestamp = event_timestamp


class FakeEmbedding:
    def __init__(self, memory_id, vector, entity_type="thread", source_text="t"):
        self.memory_id = memory_id
        self.embedding = vector
        self.entity_type = entity_type
        self.source_text = source_text


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeSession:
    """Mimics the parts of SQL that matter: keyset WHERE, ORDER BY, LIMIT.

    `shuffle=True` returns unordered pages when the statement has no ORDER BY —
    the behaviour that made the old code non-deterministic.
    """

    def __init__(self, embeddings, *, shuffle=False):
        self._all = list(embeddings)
        self._shuffle = shuffle
        self.pages = 0

    async def execute(self, stmt):
        self.pages += 1
        compiled = str(stmt)
        rows = list(self._all)

        after = None
        for param, value in (stmt.compile().params or {}).items():
            if isinstance(value, uuid.UUID) and "memory_id" in param:
                after = value
        if after is not None:
            rows = [e for e in rows if str(e.memory_id) > str(after)]

        if "ORDER BY" in compiled.upper():
            rows.sort(key=lambda e: str(e.memory_id))
        elif self._shuffle:
            rows.reverse()  # a planner is free to do this

        limit = stmt._limit
        if limit is not None:
            rows = rows[:limit]
        return FakeResult([FakeRow(e) for e in rows])


def _corpus(n: int, *, best_at: int, dims: int = 4):
    """n vectors; the one at `best_at` is the unique best match for [1,0,0,0]."""
    out = []
    for i in range(n):
        vec = [0.1, 1.0, 0.0, 0.0]
        if i == best_at:
            vec = [1.0, 0.0, 0.0, 0.0]
        out.append(FakeEmbedding(uuid.uuid4(), vec, source_text=f"fact {i}"))
    return out


QUERY = [1.0, 0.0, 0.0, 0.0]


async def _search(session, **kw):
    return await search(session, uuid.uuid4(), QUERY, model="m", version=1, **kw)


# ── determinism ────────────────────────────────────────────────────────────


async def test_identical_calls_return_identical_results():
    corpus = _corpus(50, best_at=17)
    a = await _search(FakeSession(corpus, shuffle=True), k=5, page_size=10)
    b = await _search(FakeSession(list(reversed(corpus)), shuffle=True), k=5, page_size=10)
    assert [h.memory_id for h in a] == [h.memory_id for h in b]


async def test_row_order_from_the_database_cannot_change_the_result():
    """The scan orders by memory_id, so planner whim is filtered out."""
    corpus = _corpus(40, best_at=33)
    ordered = await _search(FakeSession(corpus), k=5, page_size=7)
    hostile = await _search(FakeSession(corpus, shuffle=True), k=5, page_size=7)
    assert [h.memory_id for h in ordered] == [h.memory_id for h in hostile]


async def test_page_size_does_not_change_the_result():
    """Pagination is a traversal detail; the answer must not depend on it."""
    corpus = _corpus(60, best_at=41)
    results = [
        [h.memory_id for h in await _search(FakeSession(corpus), k=6, page_size=p)]
        for p in (1, 7, 25, 1000)
    ]
    assert all(r == results[0] for r in results)


# ── completeness: nothing relevant is silently skipped ─────────────────────


async def test_the_best_match_is_found_even_when_it_sorts_last_by_memory_id():
    """The old code could miss this entirely; keyset order is unrelated to score."""
    corpus = _corpus(30, best_at=0)
    corpus[0].memory_id = uuid.UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")
    hits = await _search(FakeSession(corpus, shuffle=True), k=3, page_size=4)
    assert hits[0].memory_id == corpus[0].memory_id


async def test_the_best_match_is_found_when_it_sorts_first_by_memory_id():
    corpus = _corpus(30, best_at=0)
    corpus[0].memory_id = uuid.UUID("00000000-0000-4000-8000-000000000000")
    hits = await _search(FakeSession(corpus, shuffle=True), k=3, page_size=4)
    assert hits[0].memory_id == corpus[0].memory_id


async def test_every_candidate_is_visited_exactly_once():
    corpus = _corpus(37, best_at=5)
    session = FakeSession(corpus)
    hits = await _search(session, k=37, page_size=10)
    assert len({h.memory_id for h in hits}) == 37, "a row was skipped or duplicated"


async def test_a_corpus_far_larger_than_the_old_5000_limit_is_fully_scanned():
    """The exact failure mode: the best fact sits beyond the old truncation."""
    corpus = _corpus(6000, best_at=5999)
    hits = await _search(FakeSession(corpus, shuffle=True), k=3, page_size=500)
    assert hits[0].memory_id == corpus[5999].memory_id


# ── ranking semantics are unchanged ────────────────────────────────────────


async def test_result_is_ordered_by_score_then_memory_id():
    corpus = _corpus(20, best_at=3)
    hits = await _search(FakeSession(corpus), k=20, page_size=6)
    keys = [(-h.score, str(h.memory_id)) for h in hits]
    assert keys == sorted(keys), "output must match the -score, memory_id rule"


async def test_ties_break_on_lowest_memory_id():
    ids = sorted(uuid.uuid4() for _ in range(5))
    corpus = [FakeEmbedding(i, [1.0, 0.0, 0.0, 0.0]) for i in ids]
    hits = await _search(FakeSession(corpus, shuffle=True), k=3, page_size=2)
    assert [h.memory_id for h in hits] == ids[:3]


async def test_paged_scan_matches_scoring_the_whole_corpus_at_once():
    """Equivalence with the single-pass implementation it replaces."""
    corpus = _corpus(45, best_at=12)
    from navis.ml.math import cosine

    expected = sorted(
        (Hit(e.memory_id, cosine(QUERY, e.embedding), e.entity_type, e.source_text)
         for e in corpus),
        key=lambda h: (-h.score, str(h.memory_id)),
    )[:8]
    hits = await _search(FakeSession(corpus, shuffle=True), k=8, page_size=6)
    assert [h.memory_id for h in hits] == [h.memory_id for h in expected]
    assert [round(h.score, 12) for h in hits] == [round(h.score, 12) for h in expected]


async def test_fewer_candidates_than_k_returns_them_all():
    corpus = _corpus(3, best_at=1)
    assert len(await _search(FakeSession(corpus), k=8, page_size=10)) == 3


async def test_empty_corpus_returns_empty():
    assert await _search(FakeSession([]), k=8) == []


# ── the safety ceiling is observable, not silent ───────────────────────────


async def test_ceiling_bounds_work_and_stays_deterministic():
    corpus = _corpus(100, best_at=99)
    session = FakeSession(corpus, shuffle=True)
    a = await _search(session, k=5, page_size=10, max_candidates=30)
    b = await _search(FakeSession(list(reversed(corpus)), shuffle=True),
                      k=5, page_size=10, max_candidates=30)
    # Truncated, but at a defined point: the first 30 by memory_id, identically.
    assert [h.memory_id for h in a] == [h.memory_id for h in b]


async def test_ceiling_is_far_above_any_real_corpus():
    """It must never bite in normal operation — the largest org holds ~3k."""
    assert MAX_CANDIDATES >= 100_000
    assert CANDIDATE_PAGE_SIZE >= 100


async def test_no_arbitrary_truncation_below_the_ceiling():
    """Between the old 5000 limit and the ceiling, everything is still scanned."""
    corpus = _corpus(9000, best_at=8500)
    hits = await _search(FakeSession(corpus, shuffle=True), k=1, page_size=1000)
    assert hits[0].memory_id == corpus[8500].memory_id
