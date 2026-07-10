"""Phase 3 regression tests — provider reliability boundary.

Covers exception normalization, bounded retry with idempotency-aware policy (generate does NOT
retry rate-limit; embed does), circuit breaking, timeout mapping, and that the offline provider
is never wrapped (determinism/replay path stays pure). All fault-injected via a fake provider;
`sleep` is stubbed so retries don't wait.
"""
from __future__ import annotations

import asyncio

import pytest

from navis.ml.providers.base import (
    EmbedResult,
    GenRequest,
    GenResult,
    MLAuthError,
    MLPermanent,
    MLProvider,
    MLProviderError,
    MLRateLimited,
    MLTimeout,
    MLTransient,
    ProviderInfo,
    normalize_provider_exc,
)
from navis.ml.providers.deterministic import DeterministicProvider
from navis.ml.providers.registry import ResilientProvider, _CircuitBreaker, _wrap

_REQ = GenRequest(system="s", user="u")


async def _noop(_seconds: float) -> None:
    return None


class _Fake(MLProvider):
    def __init__(self, *, fails: int = 0, exc: BaseException | None = None,
                 delay: float = 0.0, online: bool = True):
        self.info = ProviderInfo("fake", "gm", "em", 8, online=online)
        self._fails = fails
        self._exc = exc or MLTransient("boom")
        self._delay = delay
        self.calls = 0

    async def generate(self, req: GenRequest) -> GenResult:
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self.calls <= self._fails:
            raise self._exc
        return GenResult("ok", "gm")

    async def embed(self, texts: list[str]) -> EmbedResult:
        self.calls += 1
        if self.calls <= self._fails:
            raise self._exc
        return EmbedResult([[0.0] * 8], "em", 8)


def _rp(fake: _Fake, **kw) -> ResilientProvider:
    kw.setdefault("sleep", _noop)
    return ResilientProvider(fake, **kw)


# --- normalization (vendor errors never leak) -------------------------------

def test_normalize_maps_status_and_type():
    def exc(status):
        e = Exception("x")
        e.status_code = status
        return e

    assert isinstance(normalize_provider_exc(exc(429)), MLRateLimited)
    assert isinstance(normalize_provider_exc(exc(500)), MLTransient)
    assert isinstance(normalize_provider_exc(exc(401)), MLAuthError)
    assert isinstance(normalize_provider_exc(exc(400)), MLPermanent)
    assert isinstance(normalize_provider_exc(TimeoutError()), MLTimeout)
    assert isinstance(normalize_provider_exc(ValueError("bug")), MLPermanent)  # unknown → not retried
    passthrough = MLTimeout("t")
    assert normalize_provider_exc(passthrough) is passthrough


async def test_raw_vendor_exception_never_leaks():
    with pytest.raises(MLProviderError):  # a raw ValueError comes back normalized
        await _rp(_Fake(fails=1, exc=ValueError("raw")), max_attempts=1).generate(_REQ)


# --- retry policy -----------------------------------------------------------

async def test_retry_succeeds_after_transient():
    fake = _Fake(fails=2, exc=MLTransient("blip"))
    result = await _rp(fake, max_attempts=3).generate(_REQ)
    assert result.text == "ok" and fake.calls == 3


async def test_generate_does_not_retry_rate_limit():
    fake = _Fake(fails=2, exc=MLRateLimited("429"))
    with pytest.raises(MLRateLimited):
        await _rp(fake, max_attempts=3).generate(_REQ)
    assert fake.calls == 1  # non-streaming generate never retries a 429


async def test_embed_retries_rate_limit():
    fake = _Fake(fails=2, exc=MLRateLimited("429"))
    result = await _rp(fake, max_attempts=3).embed(["a"])
    assert result.dim == 8 and fake.calls == 3  # idempotent → retryable


async def test_non_retryable_fails_fast():
    fake = _Fake(fails=1, exc=MLAuthError("401"))
    with pytest.raises(MLAuthError):
        await _rp(fake, max_attempts=3).generate(_REQ)
    assert fake.calls == 1


async def test_timeout_maps_and_is_retryable():
    fake = _Fake(delay=0.05)  # slower than the deadline
    with pytest.raises(MLTimeout):
        await _rp(fake, max_attempts=1, timeout_s=0.001).generate(_REQ)


# --- circuit breaker --------------------------------------------------------

async def test_breaker_opens_and_fails_fast():
    fake = _Fake(fails=999, exc=MLTransient("down"))
    breaker = _CircuitBreaker(threshold=3, cooldown_s=100.0)
    rp = _rp(fake, max_attempts=1, breaker=breaker)
    for _ in range(3):
        with pytest.raises(MLTransient):
            await rp.generate(_REQ)
    assert fake.calls == 3 and await rp.health() is False  # breaker now open
    with pytest.raises(MLTransient) as ei:
        await rp.generate(_REQ)
    assert "circuit open" in str(ei.value) and fake.calls == 3  # inner not called


async def test_healthy_before_failures():
    assert await _rp(_Fake()).health() is True


# --- offline provider is never wrapped (determinism/replay stays pure) ------

def test_offline_provider_not_wrapped():
    d = DeterministicProvider()
    assert _wrap(d) is d  # returned verbatim, no resilience machinery


def test_online_provider_is_wrapped():
    fake = _Fake(online=True)
    wrapped = _wrap(fake)
    assert isinstance(wrapped, ResilientProvider) and wrapped.info is fake.info
