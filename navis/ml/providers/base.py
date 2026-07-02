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


# --- Normalized provider errors -------------------------------------------------------------
# ONE RESPONSIBILITY -> ONE HOME for provider-failure semantics: raw vendor/SDK exceptions are
# mapped to these at the MLProvider boundary and NEVER leak past it, so callers (and the
# resilience wrapper) reason about failure generically — the class-level `retryable` flag
# decides retry with no vendor knowledge.
class MLProviderError(Exception):
    """Base for every normalized provider failure. `retryable` is a policy flag."""

    retryable: bool = False


class MLTimeout(MLProviderError):
    """The call exceeded its deadline (connect or response)."""

    retryable = True


class MLRateLimited(MLProviderError):
    """Provider throttled the request (HTTP 429). Retryable with backoff."""

    retryable = True


class MLTransient(MLProviderError):
    """Transient server/connection error (HTTP 5xx, reset, circuit open)."""

    retryable = True


class MLAuthError(MLProviderError):
    """Authentication/authorization failure (bad key, 401/403). Not retryable."""

    retryable = False


class MLContentFiltered(MLProviderError):
    """Provider refused/blocked the content (safety filter). Not retryable."""

    retryable = False


class MLPermanent(MLProviderError):
    """Permanent request error (bad request/4xx, unsupported). Not retryable."""

    retryable = False


def normalize_provider_exc(exc: BaseException) -> MLProviderError:
    """Map a raw vendor/transport exception to the MLProviderError taxonomy WITHOUT importing
    any vendor SDK — classification is duck-typed on HTTP status + class-name substrings, so it
    works across Azure/OpenAI/Anthropic/Gemini/httpx. Already-normalized errors pass through.
    An UNCLASSIFIED exception maps to MLPermanent (not retried) so a code bug is never retried."""
    if isinstance(exc, MLProviderError):
        return exc
    if isinstance(exc, TimeoutError):  # asyncio.TimeoutError is TimeoutError on 3.11+
        return MLTimeout(str(exc) or "timeout")
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__.lower()
    text = f"{name}:{exc}"
    if status == 429 or "ratelimit" in name or "rate_limit" in name:
        return MLRateLimited(text)
    if status in (401, 403) or "authentic" in name or "permission" in name:
        return MLAuthError(text)
    if "contentfilter" in name or "content_filter" in name or "blocked" in name or "safety" in name:
        return MLContentFiltered(text)
    if isinstance(status, int) and 500 <= status < 600:
        return MLTransient(text)
    if "timeout" in name:
        return MLTimeout(text)
    if "connection" in name or "connect" in name:
        return MLTransient(text)
    if isinstance(status, int) and 400 <= status < 500:
        return MLPermanent(text)
    return MLPermanent(text)


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

    async def health(self) -> bool:
        """Structural health: True once the provider is constructed/configured. The real-time
        operational signal is the resilience wrapper's circuit breaker (see registry)."""
        return True

    async def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        """Provider-capability reranker (FALLBACK): cosine of query vs doc embeddings,
        provider-agnostic, over raw doc strings.

        ONE RESPONSIBILITY -> ONE HOME: the single home for retrieval-stage reranking is
        `navis.ml.retrieval.reranker.rerank` (blends vector score + lexical overlap over
        Hits). This method is the generic provider default for callers holding raw strings.
        The two are intentionally distinct — do NOT add a third reranker; extend one."""
        from navis.ml.math import cosine

        q = (await self.embed([query])).vectors[0]
        doc_vecs = (await self.embed(docs)).vectors
        scored = [RankedDoc(i, cosine(q, v)) for i, v in enumerate(doc_vecs)]
        return sorted(scored, key=lambda d: (-d.score, d.index))
