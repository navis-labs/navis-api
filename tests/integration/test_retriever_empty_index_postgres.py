"""The empty-index short-circuit, proved against real PostgreSQL.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://.../navis_test \
        pytest tests/integration/test_retriever_empty_index_postgres.py -q

`MemoryRetriever` now asks `index_has_rows` before it embeds the query, and
returns `[]` without embedding when the answer is no. That is only correct if
"no" is a PROOF that `search` would have returned nothing — for any query
vector, any time window, on either of its paths. The unit tests fake the
database; this module is where the claim meets the real filter.

Three organizations, one table:
  * POP    — three vectors under the read pin's (model, version);
  * EMPTY  — none at all;
  * OTHER  — vectors, but only under a DIFFERENT model, which the read pin must
             not count (a re-embed in progress looks exactly like this).

The provider is a counting fake, so "was the query embedded?" is observed
directly and no network is involved. Rows are inserted with the JSONB
`embedding` only, so the module holds whether or not another module has added
or dropped the typed `embedding_vec` column: `search` answers through its JSONB
path either way.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("NAVIS_TEST_DATABASE_URL"),
        reason="set NAVIS_TEST_DATABASE_URL (disposable Postgres) to run these",
    ),
    pytest.mark.asyncio(loop_scope="module"),
]

POP = uuid.UUID("cccccccc-0000-4000-8000-0000000000e1")
EMPTY = uuid.UUID("cccccccc-0000-4000-8000-0000000000e2")
OTHER = uuid.UUID("cccccccc-0000-4000-8000-0000000000e3")
MODEL = "test-embedder-empty-index"
OTHER_MODEL = "test-embedder-another-model"
NOW = datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc)


class _Provider:
    def __init__(self, *, fail: bool = False):
        self.calls = 0
        self.fail = fail

    async def embed(self, texts):
        from navis.ml.providers.base import EmbedResult
        from navis.ml.retrieval.vector_search import ANN_DIM

        self.calls += 1
        if self.fail:
            raise ConnectionError("embedding provider unavailable")
        return EmbedResult(vectors=[[1.0] + [0.0] * (ANN_DIM - 1)], model=MODEL, dim=ANN_DIM)


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401  register every model
    from navis.platform.database import Base

    url = os.environ["NAVIS_TEST_DATABASE_URL"]
    name = url.rsplit("/", 1)[-1].split("?")[0]
    assert name.startswith("navis_test"), (
        f"refusing to run against database {name!r}: point "
        "NAVIS_TEST_DATABASE_URL at a disposable database"
    )
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def session(engine):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        await _seed(s)
        yield s
        await s.rollback()

    async with engine.begin() as conn:
        for table in ("memory_embeddings", "memory_records", "signal_events", "organizations"):
            column = "id" if table == "organizations" else "organization_id"
            await conn.execute(
                text(f"DELETE FROM {table} WHERE {column} IN (:a, :b, :c)"),
                {"a": POP, "b": EMPTY, "c": OTHER},
            )


async def _seed(s) -> None:
    """Raw SQL for the same reason as `test_vector_search_fallback_postgres`:
    the rows are mostly foreign keys the test is not about."""
    from sqlalchemy import text

    from navis.ml.retrieval.vector_search import ANN_DIM

    for org in (POP, EMPTY, OTHER):
        await s.execute(
            text("INSERT INTO organizations (id, name, slug) VALUES (:o, 'EmptyIdx', :slug) "
                 "ON CONFLICT (id) DO NOTHING"),
            {"o": org, "slug": f"empty-idx-{uuid.uuid4().hex[:8]}"},
        )
    for org, model in ((POP, MODEL), (OTHER, OTHER_MODEL)):
        sig = uuid.uuid4()
        await s.execute(
            text("INSERT INTO signal_events (id, organization_id, source, event_type, external_id,"
                 " external_version, occurred_at, payload, metadata) VALUES (:i, :o, 'gmail',"
                 " 'message', :x, '1', :t, '{}'::jsonb, '{}'::jsonb)"),
            {"i": sig, "o": org, "x": f"empty-idx-{sig.hex[:8]}", "t": NOW},
        )
        for i in range(3):
            mid = uuid.uuid4()
            await s.execute(
                text("INSERT INTO memory_records (memory_id, organization_id, entity_type,"
                     " entity_id, source_signal_event_id, extractor_name, extractor_version,"
                     " schema_version, event_timestamp, confidence, dedup_key, payload)"
                     " VALUES (:m, :o, 'person', :e, :s, 'test', '1', 1, :t, 1.0, :d, '{}'::jsonb)"),
                {"m": mid, "o": org, "e": uuid.uuid4(), "s": sig, "t": NOW, "d": f"ei-{mid.hex[:12]}"},
            )
            vec = [0.0] * ANN_DIM
            vec[0] = 1.0 - (i * 0.5)
            vec[1] = i * 0.5
            await s.execute(
                text("INSERT INTO memory_embeddings (id, organization_id, memory_id, model, version,"
                     " dim, embedding, entity_type, source_text) VALUES (:i, :o, :m, :mo, 1, :dim,"
                     " CAST(:emb AS jsonb), 'person', :st)"),
                {"i": uuid.uuid4(), "o": org, "m": mid, "mo": model, "dim": ANN_DIM,
                 "emb": json.dumps(vec), "st": f"works on the pricing review ({i})"},
            )
    await s.commit()


def _retriever(session, provider, model=MODEL):
    from navis.ml.embeddings.pins import EmbeddingPin
    from navis.ml.retrieval.retriever import MemoryRetriever

    return MemoryRetriever(session, pin=EmbeddingPin(provider, model, 1))


async def test_the_check_is_false_exactly_where_search_has_nothing(session):
    from navis.ml.retrieval.vector_search import ANN_DIM, index_has_rows, search

    probe = [1.0] + [0.0] * (ANN_DIM - 1)
    for org, model, version in (
        (POP, MODEL, 1), (EMPTY, MODEL, 1), (OTHER, MODEL, 1),
        (OTHER, OTHER_MODEL, 1), (POP, MODEL, 2),
    ):
        has = await index_has_rows(session, org, model=model, version=version)
        found = await search(session, org, probe, model=model, version=version, k=10)
        assert has == bool(found), (org, model, version)


async def test_an_empty_org_returns_the_same_empty_answer_without_embedding(session):
    from navis.ml.retrieval.vector_search import ANN_DIM, search

    provider = _Provider()
    hits = await _retriever(session, provider).search(EMPTY, "what is the pricing review status")

    old_path = await search(session, EMPTY, [1.0] + [0.0] * (ANN_DIM - 1), model=MODEL, version=1, k=80)
    assert hits == old_path == []
    assert provider.calls == 0


async def test_a_populated_org_returns_exactly_what_the_old_path_returned(session, monkeypatch):
    import navis.ml.retrieval.retriever as R

    now_provider = _Provider()
    now = await _retriever(session, now_provider).search(POP, "pricing review")

    async def always(db, org_id, *, model, version):  # the path before the check existed
        return True

    monkeypatch.setattr(R, "index_has_rows", always)
    before_provider = _Provider()
    before = await _retriever(session, before_provider).search(POP, "pricing review")

    assert [(h.memory_id, round(h.score, 9)) for h in now] == [
        (h.memory_id, round(h.score, 9)) for h in before
    ]
    assert len(now) > 0
    assert now_provider.calls == before_provider.calls == 1


async def test_the_read_pin_decides_another_models_vectors_do_not_count(session):
    provider = _Provider()

    assert await _retriever(session, provider).search(OTHER, "pricing review") == []
    assert provider.calls == 0

    other_provider = _Provider()
    assert await _retriever(session, other_provider, model=OTHER_MODEL).search(OTHER, "pricing review")
    assert other_provider.calls == 1


async def test_a_window_with_nothing_in_it_still_takes_the_normal_path(session):
    """The check ignores the window on purpose: it may only ever err towards
    doing the work. An org with vectors, asked about a window that holds none,
    embeds and searches as before and gets the same empty answer."""
    provider = _Provider()
    far = NOW + timedelta(days=365)

    hits = await _retriever(session, provider).search(POP, "pricing review", since=far)

    assert hits == []
    assert provider.calls == 1


async def test_an_outage_fails_a_populated_org_and_cannot_fail_an_empty_one(session):
    with pytest.raises(ConnectionError):
        await _retriever(session, _Provider(fail=True)).search(POP, "pricing review")

    await session.rollback()  # nothing ran on it; keep the module's session clean
    assert await _retriever(session, _Provider(fail=True)).search(EMPTY, "pricing review") == []
