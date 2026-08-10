"""Embedding pins — which (provider, model, version) index is written vs read.

Before this module, `EmbeddingService` (writer) and `MemoryRetriever` (reader)
both called `get_embedding_provider()`, and `vector_search.search()` filtered on
that single model. Switching the embedding provider therefore moved reads to an
index with zero rows: every query returned nothing until a full backfill landed.

Splitting the pins removes that window. The write pin follows configuration; the
read pin can be held on the CURRENT, complete index while the next one is built
behind it. Moving the read pin is the cutover, and moving it back is the
rollback — `memory_embeddings` is append-only, so the previous index is still
there, whole.

A pin carries its PROVIDER, not just the model name: a query has to be embedded
by the same model that produced the stored vectors, or the cosine comparison is
meaningless. Reading `navis-hash-256` while writing Azure means query embedding
must still go through the deterministic embedder.
"""
from __future__ import annotations

from dataclasses import dataclass

from navis.ml.providers import MLProvider, get_embedding_provider
from navis.ml.providers.deterministic import DeterministicProvider
from navis.platform.config import settings

# Bumped when the TEXT fed to the embedder changes shape (see
# `navis.ml.embeddings.service.memory_text`) rather than when the model changes.
# model and version are independent axes of the same index identity.
EMBEDDING_VERSION = 1


class UnknownEmbeddingModel(RuntimeError):
    """A read pin names an index no configured provider can produce."""


@dataclass(frozen=True, slots=True)
class EmbeddingPin:
    """One index: the provider that produces it + how it is addressed in SQL."""

    provider: MLProvider
    model: str
    version: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return (await self.provider.embed(texts)).vectors


def write_pin() -> EmbeddingPin:
    """Where new vectors are written — follows the configured provider."""
    provider = get_embedding_provider()
    return EmbeddingPin(provider, provider.info.embed_model, EMBEDDING_VERSION)


def _provider_for_model(model: str) -> MLProvider:
    """Resolve the provider that produces `model`.

    Deliberately strict: an unrecognised pin raises instead of silently falling
    back. A typo that quietly routed queries to the wrong embedder would return
    plausible-looking but meaningless results — far worse than failing to start.
    """
    if model == DeterministicProvider.info.embed_model:
        return DeterministicProvider()
    configured = get_embedding_provider()
    if model == configured.info.embed_model:
        return configured
    if settings.azure_openai_embed_deployment and model == settings.azure_openai_embed_deployment:
        from navis.ml.providers.azure import AzureProvider

        return AzureProvider()
    raise UnknownEmbeddingModel(
        f"embedding_read_model={model!r} is not produced by any configured provider. "
        f"Expected {DeterministicProvider.info.embed_model!r}, "
        f"{configured.info.embed_model!r}, or the configured Azure embed deployment."
    )


def read_pin() -> EmbeddingPin:
    """Which index retrieval searches.

    Unset (default) => follow the write pin, i.e. today's behaviour unchanged.
    """
    model = settings.embedding_read_model
    if not model:
        return write_pin()
    version = settings.embedding_read_version
    return EmbeddingPin(
        _provider_for_model(model),
        model,
        EMBEDDING_VERSION if version is None else version,
    )


def pins_are_aligned() -> bool:
    """True when reads and writes address the same index (no migration in flight).

    Useful as a health signal: a long-lived split means a backfill was started
    and never finished.
    """
    r, w = read_pin(), write_pin()
    return (r.model, r.version) == (w.model, w.version)
