"""Regression tests for navis/ml/retrieval/vector_search.py — deterministic candidate selection.

The candidate fetch must ORDER BY the PK so that when an org has more embeddings than the cap,
the SAME set is scored every run (replay-safe). Also verifies the top-k sort is deterministic
(score desc, memory_id tiebreak) and tenant/version scoped. Fake session — no Postgres needed.
"""
from __future__ import annotations

import types
import uuid

from sqlalchemy.dialects import postgresql

from navis.ml.retrieval.vector_search import Hit, search

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self._rows = rows
        self.stmts: list = []

    async def scalars(self, stmt):
        self.stmts.append(stmt)
        return _Result(self._rows)


def _row(mid: int, vec, *, entity_type="person", text="t"):
    return types.SimpleNamespace(
        memory_id=uuid.UUID(int=mid), embedding=vec, entity_type=entity_type, source_text=text
    )


def _sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect())).upper()


async def test_candidate_query_is_ordered_and_scoped():
    sess = _Session([])
    await search(sess, ORG, [0.0, 1.0], model="m", version=1, k=8)  # type: ignore[arg-type]
    sql = _sql(sess.stmts[0])
    assert "ORDER BY MEMORY_EMBEDDINGS.MEMORY_ID" in sql          # deterministic candidate set
    assert "LIMIT" in sql
    assert "ORGANIZATION_ID" in sql and "MODEL" in sql and "VERSION" in sql  # tenant + version scoped


async def test_topk_orders_by_similarity():
    rows = [_row(2, [1.0, 0.0], text="hit"), _row(1, [0.0, 1.0], text="miss")]
    hits = await search(_Session(rows), ORG, [1.0, 0.0], model="m", version=1, k=2)  # type: ignore[arg-type]
    assert [h.memory_id.int for h in hits] == [2, 1]             # aligned vector first (score desc)
    assert isinstance(hits[0], Hit) and hits[0].source_text == "hit"


async def test_score_ties_break_by_memory_id():
    rows = [_row(5, [1.0, 0.0]), _row(3, [1.0, 0.0])]           # equal score
    hits = await search(_Session(rows), ORG, [1.0, 0.0], model="m", version=1, k=2)  # type: ignore[arg-type]
    assert [h.memory_id.int for h in hits] == [3, 5]            # stable tiebreak: lower id first


async def test_hit_is_immutable():
    import dataclasses

    h = Hit(uuid.uuid4(), 0.5, "person", "t")
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        h.score = 0.9  # type: ignore[misc]
