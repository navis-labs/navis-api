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

from sqlalchemy import select
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
