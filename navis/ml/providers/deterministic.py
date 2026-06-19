"""DeterministicProvider — the default when no API key is set.

Offline, network-free, and fully deterministic, so the whole ML stack runs and is
testable without a vendor. Embeddings use feature hashing (a real, well-known
vectorizer): tokens hash into a fixed-dim bucket vector, L2-normalized — so lexical
overlap yields cosine similarity. Generation is a faithful, grounded template over
the provided context (it never invents facts and never emits decisions).

This is NOT a mock: it is a correct, lower-fidelity implementation that preserves
every contract (determinism, lineage, token accounting) the online providers honor.
"""
from __future__ import annotations

import hashlib
import re

from navis.ml.math import l2_normalize
from navis.ml.providers.base import (
    EmbedResult,
    GenRequest,
    GenResult,
    MLProvider,
    ProviderInfo,
)

_DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _hash_embed(text: str, dim: int = _DIM) -> list[float]:
    vec = [0.0] * dim
    for tok in _tokens(text):
        h = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
        bucket = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0  # signed hashing reduces collisions
        vec[bucket] += sign
    return l2_normalize(vec)


class DeterministicProvider(MLProvider):
    info = ProviderInfo(
        name="deterministic",
        gen_model="navis-deterministic-1",
        embed_model="navis-hash-256",
        embed_dim=_DIM,
        online=False,
    )

    async def generate(self, req: GenRequest) -> GenResult:
        # Grounded template: echo the structured context, never fabricate.
        text = (
            "[deterministic-narrative]\n"
            f"{req.user.strip()}"
        )
        return GenResult(
            text=text,
            model=self.info.gen_model,
            input_tokens=len(_tokens(req.system)) + len(_tokens(req.user)),
            output_tokens=len(_tokens(text)),
            latency_ms=0,
        )

    async def embed(self, texts: list[str]) -> EmbedResult:
        vectors = [_hash_embed(t) for t in texts]
        return EmbedResult(
            vectors=vectors,
            model=self.info.embed_model,
            dim=self.info.embed_dim,
            input_tokens=sum(len(_tokens(t)) for t in texts),
            latency_ms=0,
        )
