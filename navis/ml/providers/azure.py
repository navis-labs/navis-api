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
        self._client_obj = None
        self._client_loop = None

    def _client(self):
        # Reuse one AsyncAzureOpenAI (and its httpx connection pool) per event loop: reused
        # across all calls on a long-lived server loop (pooling), and rebuilt if the running
        # loop changed (e.g. a Celery asyncio.run task on a fresh loop) so a client bound to a
        # closed loop is never reused.
        import asyncio

        from openai import AsyncAzureOpenAI

        loop = asyncio.get_running_loop()
        if self._client_obj is None or self._client_loop is not loop:
            self._client_obj = AsyncAzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint or "",
                api_version=settings.azure_openai_api_version,
                # ResilientProvider (registry) is the SOLE retry owner. The openai SDK defaults
                # to max_retries=2; leaving it would DOUBLE-retry (up to 3x3 = 9 provider calls
                # per logical call during an incident — retry amplification).
                max_retries=0,
            )
            self._client_loop = loop
        return self._client_obj

    async def aclose(self) -> None:
        client, self._client_obj, self._client_loop = self._client_obj, None, None
        if client is not None:
            await client.close()

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
