"""GeminiProvider — Google Gemini (chat + embeddings). Lazy SDK import.

Kept minimal and future-ready; activated only when a Google API key is configured.
"""
from __future__ import annotations

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

    def _genai(self):
        import google.generativeai as genai

        genai.configure(api_key=getattr(settings, "google_api_key", None))
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
        vectors = [
            genai.embed_content(model=EMBED_MODEL, content=t)["embedding"] for t in texts
        ]
        latency = int((time.monotonic() - start) * 1000)
        return EmbedResult(
            vectors=vectors, model=EMBED_MODEL,
            dim=len(vectors[0]) if vectors else EMBED_DIM, latency_ms=latency,
        )
