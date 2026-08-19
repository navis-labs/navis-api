"""Server-side exact vector search via pgvector.

Retrieval was scoring every vector in Python: `search()` paged the whole corpus
out of Postgres and ran `cosine()` per row. On the production endpoint that is
785 rows x 1536 floats of JSONB pulled across the wire for every question —
measured at 7.2s of a 17.7s chat turn, and growing linearly with the corpus.

This adds a real `vector` column so Postgres can do the comparison and return
only the k best rows.

EXACT, NOT APPROXIMATE — deliberately no HNSW/IVFFlat index. `ORDER BY
embedding_vec <=> :q LIMIT k` without an index is an exact scan computed
server-side: same answer as scoring the whole corpus, just without shipping it.
An ANN index would make retrieval approximate and could silently omit the best
match, which is precisely the defect tests/ml/test_vector_search_determinism.py
was written to prevent ("the best match is found even when it sorts last").
Adding one later is a deliberate accuracy trade-off, not a free speedup.

The JSONB `embedding` column is KEPT and still written. It is the portable
source of truth, it holds the 256-dim `navis-hash-256` rows that a `vector(1536)`
column cannot, and it is what the rollback pin reads.

Backfill is bounded and idempotent: only rows whose stored dim is 1536, only
where the column is still NULL. Safe to re-run.

Revision ID: 0038_pgvector_embeddings
Revises: 0037_execution_source_events
"""
from __future__ import annotations

from alembic import op

revision = "0038_pgvector_embeddings"
down_revision = "0037_execution_source_events"
branch_labels = None
depends_on = None

# The write pin's dimension (text-embedding-3-*). A fixed width is required for a
# typed vector column; other models keep using the JSONB path.
ANN_DIM = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        f"ALTER TABLE memory_embeddings "
        f"ADD COLUMN IF NOT EXISTS embedding_vec vector({ANN_DIM})"
    )
    # JSONB array -> vector. Cast through text: pgvector parses the '[a,b,...]'
    # literal directly, which is exactly how a JSONB array renders.
    op.execute(
        f"""
        UPDATE memory_embeddings
           SET embedding_vec = (embedding::text)::vector({ANN_DIM})
         WHERE dim = {ANN_DIM}
           AND embedding_vec IS NULL
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE memory_embeddings DROP COLUMN IF EXISTS embedding_vec")
    # The extension is left installed: other objects may depend on it, and
    # dropping it is not something a table migration should decide.
