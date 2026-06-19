"""Provider selection — the only place that decides which vendor answers a call.

Selection is config-driven and degrades to the deterministic provider when no key
is set, so the system always has a working ML backend (offline + testable). The
embedding provider is pinned separately so retrieval stays stable even if the
generation vendor changes (re-embedding is an explicit, versioned operation).
"""
from __future__ import annotations

from functools import lru_cache

from navis.ml.providers.base import MLProvider
from navis.ml.providers.deterministic import DeterministicProvider
from navis.platform.config import settings


def _build(name: str) -> MLProvider:
    if name == "azure":
        from navis.ml.providers.azure import AzureProvider

        return AzureProvider()
    if name == "claude":
        from navis.ml.providers.claude import ClaudeProvider

        return ClaudeProvider()
    if name == "gemini":
        from navis.ml.providers.gemini import GeminiProvider

        return GeminiProvider()
    return DeterministicProvider()


def _configured_generation_provider() -> str:
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        return "azure"
    if settings.anthropic_api_key:
        return "claude"
    if getattr(settings, "google_api_key", None):
        return "gemini"
    return "deterministic"


@lru_cache(maxsize=8)
def get_provider(name: str | None = None) -> MLProvider:
    return _build(name or _configured_generation_provider())


@lru_cache(maxsize=1)
def get_embedding_provider() -> MLProvider:
    """Embeddings: Azure if configured (real vectors), else deterministic.

    Pinned independently so the stored embedding `model`/`version` is stable and
    re-embedding is a deliberate migration, never an accidental drift.
    """
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        return _build("azure")
    return DeterministicProvider()
