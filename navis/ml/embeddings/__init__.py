"""Append-only, versioned embeddings over immutable Memory facts."""
from navis.ml.embeddings.admission import admits
from navis.ml.embeddings.models import (
    EmbeddingBackfillProgress,
    MemoryEmbedding,
    compute_embedding_id,
    compute_progress_id,
)
from navis.ml.embeddings.pins import (
    EMBEDDING_VERSION,
    EmbeddingPin,
    UnknownEmbeddingModel,
    pins_are_aligned,
    read_pin,
    write_pin,
)
from navis.ml.embeddings.service import BackfillPage, EmbeddingService, memory_text

__all__ = [
    "MemoryEmbedding", "EmbeddingBackfillProgress", "compute_embedding_id",
    "compute_progress_id", "EmbeddingService", "BackfillPage", "EMBEDDING_VERSION",
    "memory_text", "EmbeddingPin", "read_pin", "write_pin", "pins_are_aligned",
    "UnknownEmbeddingModel", "admits",
]
