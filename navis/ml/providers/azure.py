"""AzureProvider — Azure OpenAI (chat + embeddings). Lazy SDK import."""
from __future__ import annotations

import time

from navis.ml.providers.base import EmbedResult, GenRequest, GenResult, MLProvider, ProviderInfo
from navis.platform.config import settings

DEFAULT_DEPLOYMENT = "gpt-4o-mini"
EMBED_DEPLOYMENT = "text-embedding-3-small"
EMBED_DIM = 1536


class AzureProvider(MLProvider):
    def __init__(self, deployment: str | None = None) -> None:
        model = deployment or settings.azure_openai_deployment or DEFAULT_DEPLOYMENT
        self._deployment = model
        self.info = ProviderInfo("azure", model, EMBED_DEPLOYMENT, EMBED_DIM, online=True)

    def _client(self):
        from openai import AsyncAzureOpenAI

        return AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint or "",
            api_version=settings.azure_openai_api_version,
        )

    async def generate(self, req: GenRequest) -> GenResult:
        client = self._client()
        start = time.monotonic()
        resp = await client.chat.completions.create(
            model=self._deployment,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            messages=[
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
        )
        latency = int((time.monotonic() - start) * 1000)
        usage = getattr(resp, "usage", None)
        return GenResult(
            text=resp.choices[0].message.content or "",
            model=self._deployment,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency,
        )

    async def embed(self, texts: list[str]) -> EmbedResult:
        client = self._client()
        start = time.monotonic()
        resp = await client.embeddings.create(model=EMBED_DEPLOYMENT, input=texts)
        latency = int((time.monotonic() - start) * 1000)
        vectors = [d.embedding for d in resp.data]
        usage = getattr(resp, "usage", None)
        return EmbedResult(
            vectors=vectors,
            model=EMBED_DEPLOYMENT,
            dim=len(vectors[0]) if vectors else EMBED_DIM,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            latency_ms=latency,
        )
