"""Org-scoped vector search over memory_embeddings (brute-force cosine).

Exact top-k: every vector an org owns is scored, and the best k are returned.
The scale upgrade is a pgvector ANN index behind the SAME signature — callers
never change. Always tenant-scoped: an org can only search its own vectors.

We always join `MemoryRecord` to bring back `event_timestamp` alongside each hit
so the LLM can answer date-scoped questions ("meetings this week"). Optional
`since` / `until` half-open bounds narrow the candidate set on the same column.

DETERMINISM
    This previously issued `SELECT … LIMIT 5000` with no ORDER BY. SQL does not
    guarantee row order without one, so above 5,000 vectors Postgres returned an
    ARBITRARY subset — chosen by whatever the planner found convenient, and free
    to differ between two identical calls. The best-matching memory could simply
    never be scored, and the same question could answer differently each time.
    At ~6 facts per email that threshold arrives around 800 emails.

    The scan is now keyset-paginated in `memory_id` order. `memory_id` is unique
    within a (organization, model, version) — enforced by
    `uq_mem_model_version` — so it is a total order over exactly the rows the
    filter selects: every row is visited once, in the same sequence, on every
    call. Ranking is unchanged; only the traversal is.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.models import MemoryRecord
from navis.ml.embeddings.models import MemoryEmbedding
from navis.ml.embeddings.pins import EMBEDDING_VERSION
from navis.ml.math import cosine
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("ml.retrieval.vector_search")

# Rows pulled per round trip. Bounds peak memory to one page plus the running
# top-k, rather than the whole corpus.
CANDIDATE_PAGE_SIZE = 1000

# Safety ceiling on a single search. Brute-force cosine has no way to skip work,
# so this exists to stop one pathological org pinning a worker — NOT to trim the
# corpus. Deliberately far above any real org (the largest today holds ~3k
# vectors); reaching it is logged and counted, never silent.
MAX_CANDIDATES = 200_000


@dataclass(frozen=True, slots=True)
class Hit:
    memory_id: uuid.UUID
    score: float
    entity_type: str
    source_text: str
    event_timestamp: datetime | None = None


# Width of the typed `embedding_vec` column (migration 0038). Vectors of any
# other width — e.g. the 256-dim deterministic index kept for rollback — have no
# typed column and take the Python path below.
ANN_DIM = 1536


async def _search_pgvector(
    db: AsyncSession,
    org_id: uuid.UUID,
    query_vector: list[float],
    *,
    model: str,
    version: int,
    k: int,
    since: datetime | None,
    until: datetime | None,
) -> list[Hit] | None:
    """Exact top-k computed in Postgres. Returns None if unavailable.

    EXACT, NOT APPROXIMATE. There is no ANN index behind this: `ORDER BY
    embedding_vec <=> :q LIMIT k` is a full scan done server-side, so it returns
    what scoring every vector in Python returns — it simply stops shipping 785
    rows x 1536 floats of JSONB across the network to do it. That transfer, not
    the arithmetic, was the 7.2s cost in a 17.7s chat turn.

    An HNSW/IVFFlat index would make this approximate and could silently drop the
    best match, which tests/ml/test_vector_search_determinism.py exists to
    prevent. That is an accuracy decision to take deliberately, not a free win.

    `<=>` is cosine DISTANCE, so similarity is `1 - distance` — the same value
    `navis.ml.math.cosine` produces. Ties break on `memory_id` exactly as
    `_rank_key` does, so ordering is identical to the Python path.

    Returns None (rather than raising) when the typed column is absent or holds
    no rows for this index, so a database that has not run migration 0038, or the
    rollback pin, transparently keeps the previous behaviour.
    """
    literal = "[" + ",".join(repr(float(x)) for x in query_vector) + "]"
    params: dict[str, object] = {
        "q": literal, "org": org_id, "model": model, "version": version, "k": k,
    }
    # Added conditionally, never as `:since IS NULL OR ...`: asyncpg cannot infer
    # a type for an untyped NULL parameter and the whole statement fails — which
    # this function would then swallow, silently costing the speedup while the
    # results still looked right.
    window = ""
    if since is not None:
        window += " AND r.event_timestamp >= :since"
        params["since"] = since
    if until is not None:
        window += " AND r.event_timestamp < :until"
        params["until"] = until
    sql = text(
        f"""
        SELECT e.memory_id, e.entity_type, e.source_text, r.event_timestamp,
               1 - (e.embedding_vec <=> CAST(:q AS vector)) AS score
          FROM memory_embeddings e
          JOIN memory_records r ON r.memory_id = e.memory_id
         WHERE e.organization_id = :org
           AND e.model = :model
           AND e.version = :version
           AND e.embedding_vec IS NOT NULL
           {window}
         ORDER BY e.embedding_vec <=> CAST(:q AS vector), e.memory_id
         LIMIT :k
        """
    )
    try:
        # ── THE SAVEPOINT IS LOAD-BEARING, NOT DEFENSIVE STYLE ─────────
        # Catching the exception is not enough on PostgreSQL. A failed
        # statement puts the whole transaction into an aborted state, and the
        # server then refuses EVERY later command on that session with
        # InFailedSQLTransactionError. The JSONB scan below runs on this same
        # session, so without a savepoint the degrade path raised a second,
        # unrelated error and a search that was designed to survive pgvector
        # being unavailable returned a 500 instead.
        #
        # `begin_nested()` issues SAVEPOINT / ROLLBACK TO SAVEPOINT, which
        # discards only this statement and leaves the caller's transaction
        # exactly as it was. Nothing is committed here: the enclosing unit of
        # work still decides that.
        #
        # Reproduced by `tests/integration/test_vector_search_fallback_postgres`
        # against a schema with no `embedding_vec`, which is the real shape of
        # a database whose raw-DDL migration has not run.
        async with db.begin_nested():
            rows = (await db.execute(sql, params)).all()
    except Exception:  # noqa: BLE001 — degrade to the Python path, never fail a search
        logger.exception("pgvector search unavailable; falling back to in-process scan")
        metrics.incr("ml.retrieval.pgvector_unavailable")
        return None
    if not rows:
        # No typed rows for this index — the caller's model may predate the
        # backfill. Let the JSONB path answer rather than returning "no results".
        return None
    metrics.incr("ml.retrieval.pgvector_hit")
    return [
        Hit(r.memory_id, float(r.score), r.entity_type, r.source_text, r.event_timestamp)
        for r in rows
    ]


def _rank_key(hit: Hit) -> tuple[float, str]:
    """The one ordering rule: best score first, then lowest memory_id.

    Used both to trim the running top-k and to order the result, so a paged scan
    returns exactly what scoring the whole corpus in one list would.
    """
    return (-hit.score, str(hit.memory_id))


async def search(
    db: AsyncSession,
    org_id: uuid.UUID,
    query_vector: list[float],
    *,
    model: str,
    version: int = EMBEDDING_VERSION,
    k: int = 8,
    page_size: int = CANDIDATE_PAGE_SIZE,
    max_candidates: int = MAX_CANDIDATES,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[Hit]:
    """Return the k best-matching memories for `query_vector`, exactly.

    Semantics are identical to scoring every candidate in one list and sorting:
    the running top-k is trimmed with the SAME key used for the final order, so
    a fact can only be dropped once k strictly better facts have been seen.
    """
    # Fast path: let Postgres do the comparison and return only the k best rows.
    # Same answer, without shipping the corpus — see `_search_pgvector`.
    if len(query_vector) == ANN_DIM:
        hits = await _search_pgvector(
            db, org_id, query_vector, model=model, version=version, k=k,
            since=since, until=until,
        )
        if hits is not None:
            return hits

    base = (
        select(MemoryEmbedding, MemoryRecord.event_timestamp)
        .join(MemoryRecord, MemoryRecord.memory_id == MemoryEmbedding.memory_id)
        .where(
            MemoryEmbedding.organization_id == org_id,
            MemoryEmbedding.model == model,
            MemoryEmbedding.version == version,
        )
    )
    if since is not None:
        base = base.where(MemoryRecord.event_timestamp >= since)
    if until is not None:
        base = base.where(MemoryRecord.event_timestamp < until)

    best: list[Hit] = []
    after: uuid.UUID | None = None
    scanned = 0

    while scanned < max_candidates:
        page_stmt = base.order_by(MemoryEmbedding.memory_id).limit(
            min(page_size, max_candidates - scanned)
        )
        if after is not None:
            page_stmt = page_stmt.where(MemoryEmbedding.memory_id > after)

        rows = (await db.execute(page_stmt)).all()
        if not rows:
            break

        best.extend(
            Hit(
                r.MemoryEmbedding.memory_id,
                cosine(query_vector, r.MemoryEmbedding.embedding),
                r.MemoryEmbedding.entity_type,
                r.MemoryEmbedding.source_text,
                r.event_timestamp,
            )
            for r in rows
        )
        # Trim with the final ordering so the survivors are always the true
        # best-so-far; nothing is discarded on any basis other than rank.
        best.sort(key=_rank_key)
        del best[k:]

        scanned += len(rows)
        after = rows[-1].MemoryEmbedding.memory_id
        if len(rows) < page_size:
            break
    else:
        # Loop exhausted the ceiling rather than the corpus.
        metrics.incr("ml.retrieval.candidate_ceiling_hit")
        logger.warning(
            "vector search hit the %d-candidate ceiling for org=%s model=%s v=%d; "
            "results are exact only over the first %d vectors by memory_id",
            max_candidates, org_id, model, version, scanned,
        )

    # A counter, not a latency histogram: the running total of vectors scored is
    # what shows the brute-force cost growing with corpus size.
    metrics.incr("ml.retrieval.candidates_scanned", value=scanned)
    return best
