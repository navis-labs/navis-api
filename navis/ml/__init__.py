"""ML layer — strictly ASSISTIVE. It never owns truth.

The deterministic pipeline (Signals→…→Adaptive) is the source of truth and contains
no ML. This layer adds: a vendor-agnostic provider abstraction, append-only versioned
embeddings over Memory, org-scoped semantic retrieval, RAG grounding (the LLM only
ever sees a GroundedContext, never the database), narrative generation (Decision→
narrative, never narrative→Decision), routing, telemetry, and evals.

INVARIANT (enforced by test): no pipeline domain imports navis.ml. ML may READ the
truth layers for retrieval; it may never inject itself into them.
"""
from navis.ml.providers import (
    EmbedResult,
    GenRequest,
    GenResult,
    MLProvider,
    get_embedding_provider,
    get_provider,
)
from navis.ml.rag import ContextBuilder, GroundedContext

__all__ = [
    "MLProvider", "GenRequest", "GenResult", "EmbedResult",
    "get_provider", "get_embedding_provider", "ContextBuilder", "GroundedContext",
]
