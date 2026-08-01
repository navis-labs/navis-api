"""AzureProvider — Azure OpenAI (chat + embeddings). Lazy SDK import.

Azure routes ALL calls (chat + embeddings) by DEPLOYMENT NAME, not by the
underlying model name. Your Azure resource can host any number of deployments;
each one has its own SKU and quota. Chat and embeddings need SEPARATE
deployments — the same deployment cannot serve both. This provider therefore
takes two deployment names:

  * `settings.azure_openai_deployment`       — for chat.completions
  * `settings.azure_openai_embed_deployment` — for embeddings

If either is unset, that capability degrades gracefully:
  * chat unset  → uses `DEFAULT_DEPLOYMENT` (may 404 if you haven't created it),
  * embed unset → `has_embedding()` returns False so the registry can pick the
    deterministic provider for embeddings instead of trying an Azure call that
    will 404.
"""
from __future__ import annotations

import time

from navis.ml.providers.base import EmbedResult, GenRequest, GenResult, MLProvider, ProviderInfo
from navis.platform.config import settings

DEFAULT_DEPLOYMENT = "gpt-4o-mini"
# Reported when no embed deployment is configured — kept so ProviderInfo has a
# sane placeholder; NOT used as a fallback deployment name (Azure would 404).
UNSET_EMBED_MODEL = "azure-embed-unconfigured"
EMBED_DIM = 1536  # text-embedding-3-small / -large default; correct for both


class AzureProvider(MLProvider):
    def __init__(
        self,
        deployment: str | None = None,
        embed_deployment: str | None = None,
    ) -> None:
        gen_model = deployment or settings.azure_openai_deployment or DEFAULT_DEPLOYMENT
        embed_model = embed_deployment or settings.azure_openai_embed_deployment
        self._deployment = gen_model
        self._embed_deployment = embed_model  # may be None → embed() will fail loudly
        # The ProviderInfo.embed_model IS the value stored in memory_embeddings.model.
        # Using the deployment name (not the underlying model name) means one Azure
        # resource can serve two different embedding SKUs across environments without
        # collision — the model column reflects the routing target, which is what
        # matters for retrieval-time filter correctness.
        self.info = ProviderInfo(
            "azure",
            gen_model,
            embed_model or UNSET_EMBED_MODEL,
            EMBED_DIM,
            online=True,
        )

    def has_embedding(self) -> bool:
        """True iff an Azure embedding deployment is configured. Used by the
        registry to decide whether to route embeddings to Azure or fall back
        to the deterministic provider — a missing deployment name means Azure
        would 404 on every embed call."""
        return self._embed_deployment is not None

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
        if self._embed_deployment is None:
            # Fail loudly rather than silently 404 — the registry is supposed
            # to route around this via has_embedding(), so reaching here means
            # something upstream bypassed that check.
            raise RuntimeError(
                "AzureProvider.embed called but AZURE_OPENAI_EMBED_DEPLOYMENT "
                "is not configured. Either set the env var to your embedding "
                "deployment name, or let the registry fall back to the "
                "deterministic embedder (get_embedding_provider handles this)."
            )
        client = self._client()
        start = time.monotonic()
        resp = await client.embeddings.create(model=self._embed_deployment, input=texts)
        latency = int((time.monotonic() - start) * 1000)
        vectors = [d.embedding for d in resp.data]
        usage = getattr(resp, "usage", None)
        return EmbedResult(
            vectors=vectors,
            model=self._embed_deployment,
            dim=len(vectors[0]) if vectors else EMBED_DIM,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            latency_ms=latency,
        )
