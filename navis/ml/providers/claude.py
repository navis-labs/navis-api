"""ClaudeProvider — Anthropic. Lazy SDK import; embeddings fall back to the
deterministic embedder (Anthropic has no first-party embedding endpoint), so
retrieval stays vendor-portable while generation uses Claude."""
from __future__ import annotations

import time

from navis.ml.providers.base import EmbedResult, GenRequest, GenResult, MLProvider, ProviderInfo
from navis.ml.providers.deterministic import DeterministicProvider
from navis.platform.config import settings

DEFAULT_MODEL = "claude-opus-4-8"
FAST_MODEL = "claude-haiku-4-5-20251001"


class ClaudeProvider(MLProvider):
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model = model
        self.info = ProviderInfo("claude", model, "navis-hash-256", 256, online=True)
        self._embedder = DeterministicProvider()

    async def generate(self, req: GenRequest) -> GenResult:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        start = time.monotonic()
        resp = await client.messages.create(
            model=self._model,
            system=req.system,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            messages=[{"role": "user", "content": req.user}],
        )
        latency = int((time.monotonic() - start) * 1000)
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = getattr(resp, "usage", None)
        return GenResult(
            text=text,
            model=self._model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_ms=latency,
        )

    async def embed(self, texts: list[str]) -> EmbedResult:
        return await self._embedder.embed(texts)
