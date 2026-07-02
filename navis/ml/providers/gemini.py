"""GeminiProvider — Google Gemini (chat + embeddings). Lazy SDK import.

Kept minimal and future-ready; activated only when a Google API key is configured.
"""
from __future__ import annotations

import asyncio
import time

from navis.ml.providers.base import EmbedResult, GenRequest, GenResult, MLProvider, ProviderInfo
from navis.platform.config import settings

DEFAULT_MODEL = "gemini-1.5-pro"
EMBED_MODEL = "text-embedding-004"
EMBED_DIM = 768


class GeminiProvider(MLProvider):
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model = model
        self.info = ProviderInfo("gemini", model, EMBED_MODEL, EMBED_DIM, online=True)
        self._configured = False

    def _genai(self):
        # Configure the SDK once per instance (module-global config; genai owns no client we
        # pool). embed() offloads its sync SDK call off the event loop (below); request batching
        # via Gemini's batch endpoint remains a future enhancement.
        import google.generativeai as genai

        if not self._configured:
            genai.configure(api_key=getattr(settings, "google_api_key", None))
            self._configured = True
        return genai

    async def generate(self, req: GenRequest) -> GenResult:
        genai = self._genai()
        model = genai.GenerativeModel(self._model, system_instruction=req.system)
        start = time.monotonic()
        resp = await model.generate_content_async(
            req.user,
            generation_config={"max_output_tokens": req.max_tokens, "temperature": req.temperature},
        )
        latency = int((time.monotonic() - start) * 1000)
        usage = getattr(resp, "usage_metadata", None)
        return GenResult(
            text=resp.text or "",
            model=self._model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency_ms=latency,
        )

    async def embed(self, texts: list[str]) -> EmbedResult:
        genai = self._genai()
        start = time.monotonic()

        # google-generativeai's embed_content is SYNCHRONOUS. Run it OFF the event loop so a
        # Gemini embed never blocks other concurrent coroutines (generate already uses the async
        # API). True request batching would need Gemini's batch endpoint — a future enhancement.
        def _embed_all() -> list[list[float]]:
            return [genai.embed_content(model=EMBED_MODEL, content=t)["embedding"] for t in texts]

        vectors = await asyncio.to_thread(_embed_all)
        latency = int((time.monotonic() - start) * 1000)
        return EmbedResult(
            vectors=vectors, model=EMBED_MODEL,
            dim=len(vectors[0]) if vectors else EMBED_DIM, latency_ms=latency,
        )
