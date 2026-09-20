"""The pgvector -> JSONB fallback, proved against real PostgreSQL.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://.../navis_test \
        pytest tests/integration/test_vector_search_fallback_postgres.py -q

WHY THIS NEEDS A REAL DATABASE. The defect is not in the Python control flow —
that reads correctly, and every in-process fake of a failing query reproduces
the `return None` and stops there. It is in what PostgreSQL does to the SESSION
when a statement fails: the transaction enters an aborted state and refuses
every subsequent command until it is rolled back. So the fallback query, issued
on the same session a moment later, fails with a DIFFERENT error than the one
that was handled, and a search that was supposed to degrade raises a 500
instead. Only a real server has that behaviour.

THE FAILURE IS NOT HYPOTHETICAL, and the fixture below is the production
condition rather than an invented one. `memory_embeddings.embedding_vec` is
added by raw DDL inside a migration, so `Base.metadata.create_all` does not
produce it — as `test_observation_contract` already records. Any database
standing at that point (a schema built by `create_all`, or one where the
migration has not run) answers the pgvector statement with
`UndefinedColumnError`, which is exactly the swallowed exception.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("NAVIS_TEST_DATABASE_URL"),
        reason="set NAVIS_TEST_DATABASE_URL (disposable Postgres) to run these",
    ),
    pytest.mark.asyncio(loop_scope="module"),
]

ORG = uuid.UUID("cccccccc-0000-4000-8000-00000000000f")
MODEL = "test-embedder"
NOW = datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc)


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
    """A session whose schema has NO `embedding_vec`, which is the whole point.

    The column is dropped rather than merely left uncreated so the module is
    correct whichever order it runs in: another module may have run the raw
    DDL against this database already.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    async with engine.begin() as conn:
        await conn.execute(
            text("ALTER TABLE memory_embeddings DROP COLUMN IF EXISTS embedding_vec")
        )

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        await _seed(s)
        yield s
        await s.rollback()

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM memory_embeddings WHERE organization_id = :o"), {"o": ORG}
        )
        await conn.execute(
            text("DELETE FROM memory_records WHERE organization_id = :o"), {"o": ORG}
        )


async def _seed(s) -> None:
    """Raw SQL rather than the ORM, because the rows this needs are almost all
    foreign keys: `memory_records` requires an organization AND a signal event,
    and neither is what the test is about. Inserting the columns the schema
    actually demands keeps the fixture honest about its dependencies instead of
    tracking every model's optional fields."""
    import json

    from sqlalchemy import text

    from navis.ml.retrieval.vector_search import ANN_DIM

    await s.execute(
        text(
            "INSERT INTO organizations (id, name, slug) VALUES (:o, 'Vec', :slug) "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {"o": ORG, "slug": f"vec-{uuid.uuid4().hex[:8]}"},
    )
    sig = uuid.uuid4()
    await s.execute(
        text(
            "INSERT INTO signal_events "
            "(id, organization_id, source, event_type, external_id, external_version,"
            " occurred_at, payload, metadata) "
            "VALUES (:i, :o, 'gmail', 'message', :x, '1', :t, '{}'::jsonb, '{}'::jsonb)"
        ),
        {"i": sig, "o": ORG, "x": f"vec-{sig.hex[:8]}", "t": NOW},
    )

    for i in range(3):
        mid = uuid.uuid4()
        await s.execute(
            text(
                "INSERT INTO memory_records "
                "(memory_id, organization_id, entity_type, entity_id,"
                " source_signal_event_id, extractor_name, extractor_version,"
                " schema_version, event_timestamp, confidence, dedup_key, payload) "
                "VALUES (:m, :o, 'person', :e, :s, 'test', '1', 1, :t, 1.0, :d,"
                " '{}'::jsonb)"
            ),
            {"m": mid, "o": ORG, "e": uuid.uuid4(), "s": sig, "t": NOW,
             "d": f"vec-{mid.hex[:12]}"},
        )
        # DIRECTION, not magnitude. Cosine normalises length, so scaling one
        # vector leaves the score at 1.0 and the ranking unobservable. Rotating
        # towards a second axis gives strictly decreasing similarity against
        # the query [1, 0, ...]: 1.0, then ~0.707, then 0.0.
        vec = [0.0] * ANN_DIM
        vec[0] = 1.0 - (i * 0.5)
        vec[1] = i * 0.5
        await s.execute(
            text(
                "INSERT INTO memory_embeddings "
                "(id, organization_id, memory_id, model, version, dim, embedding,"
                " entity_type, source_text) "
                "VALUES (:i, :o, :m, :mo, 1, :dim, CAST(:emb AS jsonb), 'person', :st)"
            ),
            {"i": uuid.uuid4(), "o": ORG, "m": mid, "mo": MODEL, "dim": ANN_DIM,
             "emb": json.dumps(vec), "st": f"fact {i}"},
        )
    await s.commit()


async def test_a_pgvector_failure_still_returns_the_jsonb_answer(session):
    """THE REGRESSION. Before the fix this raised InFailedSQLTransactionError.

    The pgvector statement fails (no `embedding_vec`), which aborts the
    transaction; the JSONB scan then runs on that same session. A search that
    degrades must still answer.
    """
    from navis.ml.retrieval.vector_search import ANN_DIM, search

    query = [1.0] + [0.0] * (ANN_DIM - 1)
    hits = await search(session, ORG, query, model=MODEL, version=1, k=3)

    assert len(hits) == 3, "the JSONB fallback did not answer"
    # Ranked, not merely returned — proving the fallback really scored.
    assert hits[0].source_text == "fact 0"
    assert hits[0].score > hits[-1].score


async def test_the_session_is_still_usable_after_the_swallowed_failure(session):
    """The property the fallback depends on, asserted directly.

    Stated separately from the search above because this is the thing that was
    actually broken: a caller that handles the pgvector error and then issues
    ANY further statement on the same session must not inherit an aborted
    transaction.
    """
    from sqlalchemy import text

    from navis.ml.retrieval.vector_search import ANN_DIM, _search_pgvector

    query = [1.0] + [0.0] * (ANN_DIM - 1)
    assert await _search_pgvector(
        session, ORG, query, model=MODEL, version=1, k=3, since=None, until=None
    ) is None, "the fixture no longer reproduces a pgvector failure"

    # The next statement is the one that used to raise.
    assert (await session.execute(text("SELECT 1"))).scalar() == 1
