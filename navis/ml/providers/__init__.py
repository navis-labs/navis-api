"""Provider abstraction — vendor-agnostic ML access (dependency inversion)."""
from navis.ml.providers.base import (
    EmbedResult,
    GenRequest,
    GenResult,
    MLProvider,
    ProviderInfo,
    RankedDoc,
)
from navis.ml.providers.registry import get_embedding_provider, get_provider

__all__ = [
    "MLProvider", "GenRequest", "GenResult", "EmbedResult", "RankedDoc", "ProviderInfo",
    "get_provider", "get_embedding_provider",
]
