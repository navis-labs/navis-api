"""The MLProvider contract — the dependency-inversion boundary for all ML.

No domain imports a vendor SDK. Every ML call goes through MLProvider, so Azure /
Claude / Gemini are swappable and the truth pipeline never couples to a model.

Three capabilities, each returning a result that carries token/cost/latency so the
telemetry layer can account for every call:
  * generate(GenRequest) -> GenResult        (narrative text only — never decisions)
  * embed(list[str])      -> EmbedResult      (deterministic, versioned vectors)
  * rerank(query, docs)   -> list[RankedDoc]  (ordering, never rewriting)
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GenRequest:
    system: str
    user: str
    max_tokens: int = 1024
    temperature: float = 0.2
    # The prompt identity (for telemetry + regression evals); set by generation layer.
    prompt_id: str | None = None
    prompt_version: int | None = None


@dataclass(frozen=True, slots=True)
class GenResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class EmbedResult:
    vectors: list[list[float]]
    model: str
    dim: int
    input_tokens: int = 0
    latency_ms: int = 0


@dataclass(frozen=True, slots=True)
class RankedDoc:
    index: int
    score: float


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    name: str
    gen_model: str
    embed_model: str
    embed_dim: int
    online: bool  # True if it makes network calls (False = deterministic/offline)


class MLProvider(abc.ABC):
    """All providers implement this. Implementations keep SDK imports lazy."""

    info: ProviderInfo

    @abc.abstractmethod
    async def generate(self, req: GenRequest) -> GenResult: ...

    @abc.abstractmethod
    async def embed(self, texts: list[str]) -> EmbedResult: ...

    async def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        """Default rerank: cosine of query vs doc embeddings (provider-agnostic)."""
        from navis.ml.math import cosine

        q = (await self.embed([query])).vectors[0]
        doc_vecs = (await self.embed(docs)).vectors
        scored = [RankedDoc(i, cosine(q, v)) for i, v in enumerate(doc_vecs)]
        return sorted(scored, key=lambda d: (-d.score, d.index))
