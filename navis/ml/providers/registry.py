"""Provider selection + the reliability boundary — the only place that decides which vendor
answers a call, and the ONE HOME for provider timeout / retry / circuit-breaking.

Selection is config-driven and degrades to the deterministic provider when no key is set, so
the system always has a working ML backend (offline + testable). The embedding provider is
pinned separately so retrieval stays stable even if the generation vendor changes (re-embedding
is an explicit, versioned operation).

Reliability: every ONLINE provider is wrapped in `ResilientProvider` (timeout, bounded jittered
retry, circuit breaker, vendor-exception normalization). The offline DeterministicProvider is
returned UNWRAPPED — it cannot fail and must stay pure, so the deterministic/replay path never
touches the reliability machinery. Wrapping is output-transparent: a successful call returns the
inner provider's exact result, so determinism and replay are unaffected.
"""
from __future__ import annotations

import asyncio
import random
import time
from functools import lru_cache

from navis.ml.providers.base import (
    EmbedResult,
    GenRequest,
    GenResult,
    MLProvider,
    MLProviderError,
    MLTimeout,
    MLTransient,
    normalize_provider_exc,
)
from navis.ml.providers.deterministic import DeterministicProvider
from navis.platform.config import settings
from navis.platform.monitoring import get_logger

logger = get_logger("ml.providers")

_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_MAX_ATTEMPTS = 3  # 1 initial try + up to 2 retries
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_S = 30.0
_BACKOFF_BASE_S = 0.2
_BACKOFF_CAP_S = 5.0

# Underlying online providers we constructed (for lifecycle shutdown via close_all).
_LIVE: set[MLProvider] = set()


class _CircuitBreaker:
    """Per-provider, in-process breaker: opens after N consecutive failures, half-opens after a
    cooldown (the next call is a probe). Best-effort under threads — counter races only nudge the
    threshold and are benign; a distributed breaker is a future upgrade."""

    def __init__(self, threshold: int = _BREAKER_THRESHOLD, cooldown_s: float = _BREAKER_COOLDOWN_S):
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._fails = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if (time.monotonic() - self._opened_at) >= self._cooldown_s:
            return False  # half-open: allow one probe through
        return True

    def record_success(self) -> None:
        self._fails = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._fails += 1
        if self._fails >= self._threshold:
            self._opened_at = time.monotonic()


class ResilientProvider(MLProvider):
    """Reliability wrapper for an ONLINE provider — the ONE HOME for timeout, bounded retry,
    circuit breaking, and vendor-exception normalization. Output-transparent (a success returns
    the inner result verbatim), so determinism/replay are unaffected. Never wraps the offline
    provider."""

    def __init__(
        self,
        inner: MLProvider,
        *,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        breaker: _CircuitBreaker | None = None,
        sleep=asyncio.sleep,
    ) -> None:
        self._inner = inner
        self.info = inner.info
        self._timeout_s = timeout_s
        self._max_attempts = max(1, max_attempts)
        self._breaker = breaker or _CircuitBreaker()
        self._sleep = sleep

    async def generate(self, req: GenRequest) -> GenResult:
        # Non-idempotent. Retry ONLY on connect/timeout/transient — for non-streaming generate
        # there is no partial output, so a retry is safe; rate-limit is deliberately NOT retried
        # here (avoid any double-processing/cost ambiguity). When streaming lands (Phase 9),
        # retry-after-first-token must be disabled.
        return await self._call("generate", lambda: self._inner.generate(req),
                                retry_on=(MLTimeout, MLTransient))

    async def embed(self, texts: list[str]) -> EmbedResult:
        # Idempotent: safe to retry on any retryable error (including rate-limit).
        return await self._call("embed", lambda: self._inner.embed(texts), retry_on=None)

    async def health(self) -> bool:
        """Operational health = breaker not open (free, no probe/cost)."""
        return not self._breaker.is_open

    async def aclose(self) -> None:
        closer = getattr(self._inner, "aclose", None)
        if closer is not None:
            await closer()

    async def _call(self, op: str, fn, *, retry_on):
        if self._breaker.is_open:
            raise MLTransient(f"{self.info.name}.{op}: circuit open")
        last: MLProviderError | None = None
        for attempt in range(self._max_attempts):
            try:
                result = await asyncio.wait_for(fn(), timeout=self._timeout_s)
                self._breaker.record_success()
                return result
            except Exception as exc:  # noqa: BLE001 — normalize EVERYTHING; vendor errors never leak
                err = normalize_provider_exc(exc)
                self._breaker.record_failure()
                last = err
                retryable = err.retryable if retry_on is None else isinstance(err, retry_on)
                if not retryable or attempt == self._max_attempts - 1:
                    raise err
                backoff = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2 ** attempt)) + random.uniform(
                    0, _BACKOFF_BASE_S
                )
                logger.warning(
                    "ml %s.%s attempt %d failed (%s) — retrying in %.2fs",
                    self.info.name, op, attempt + 1, type(err).__name__, backoff,
                )
                await self._sleep(backoff)
        # Unreachable: max_attempts >= 1, so the loop always returns or raises above. The
        # explicit fallback keeps the type non-optional (never `raise None`).
        raise last if last is not None else MLTransient(f"{self.info.name}.{op}: no attempt made")


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


def _wrap(provider: MLProvider) -> MLProvider:
    """Wrap ONLINE providers in the reliability layer; return the offline provider unwrapped
    (it cannot fail and must stay pure — preserving replay/determinism)."""
    if not provider.info.online:
        return provider
    _LIVE.add(provider)
    return ResilientProvider(
        provider,
        timeout_s=float(getattr(settings, "ml_request_timeout_s", _DEFAULT_TIMEOUT_S)),
        max_attempts=int(getattr(settings, "ml_max_attempts", _DEFAULT_MAX_ATTEMPTS)),
    )


def _configured_generation_provider() -> str:
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        return "azure"
    if settings.anthropic_api_key:
        return "claude"
    if getattr(settings, "google_api_key", None):
        return "gemini"
    return "deterministic"


def configured_providers() -> tuple[str, ...]:
    """Provider NAMES that have credentials configured, always ending in 'deterministic' (the
    always-available offline fallback). Config knowledge lives HERE (with construction), so the
    capability router can decide policy without reading secrets or constructing anything."""
    names: list[str] = []
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        names.append("azure")
    if settings.anthropic_api_key:
        names.append("claude")
    if getattr(settings, "google_api_key", None):
        names.append("gemini")
    names.append("deterministic")
    return tuple(names)


@lru_cache(maxsize=8)
def get_provider(name: str | None = None) -> MLProvider:
    return _wrap(_build(name or _configured_generation_provider()))


@lru_cache(maxsize=1)
def get_embedding_provider() -> MLProvider:
    """Embeddings: Azure if configured (real vectors), else deterministic.

    Pinned independently so the stored embedding `model`/`version` is stable and re-embedding is
    a deliberate migration, never an accidental drift.
    """
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        return _wrap(_build("azure"))
    return DeterministicProvider()


async def close_all() -> None:
    """Best-effort provider lifecycle shutdown: close pooled clients and clear the caches. Call
    from a long-running process's shutdown, or from a job's finally when using asyncio.run (each
    asyncio.run loop builds its own pooled client)."""
    for provider in list(_LIVE):
        closer = getattr(provider, "aclose", None)
        if closer is None:
            continue
        try:
            await closer()
        except Exception:  # noqa: BLE001 — shutdown must not raise
            logger.exception("ml provider close failed: %s", getattr(provider.info, "name", "?"))
    _LIVE.clear()
    get_provider.cache_clear()
    get_embedding_provider.cache_clear()
