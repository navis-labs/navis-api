"""Capability-based model routing — the ONE HOME for "which provider for this capability".

Callers declare a CAPABILITY (reasoning, extraction, embedding, structured output, tool calling,
...), never a vendor or model name. The router maps capability -> a concrete provider using a
static, config-aware preference list. It is:

  * capability-based — callers never name Claude/Azure/Gemini; only a Capability.
  * DETERMINISTIC — a pure function of (capability, configured providers). No randomness, no
    clock, no health/runtime state; the same inputs always pick the same provider. (Health-aware
    failover is the gateway's concern, deliberately kept OUT of this pure policy layer.)
  * SEPARATE FROM CONSTRUCTION — it constructs nothing and reads no secrets; it asks the
    registry which providers are configured and delegates building to `get_provider`.

  fast    -> cheapest capable model         deep/accurate -> strongest model
Everything goes through MLProvider, so swapping the underlying vendor changes nothing here.
"""
from __future__ import annotations

import enum

from navis.ml.providers.base import MLProvider
from navis.ml.providers.registry import (
    configured_providers,
    get_embedding_provider,
    get_provider,
)


class Capability(str, enum.Enum):
    """What a caller needs — never a model name. New capabilities plug in by adding a row to
    `_CAPABILITY_PREFERENCE`; callers and providers are untouched."""

    REASONING = "reasoning"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"
    LONG_CONTEXT = "long_context"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLING = "tool_calling"
    FAST = "fast"
    ACCURATE = "accurate"


# Per-capability provider-NAME preference (most-capable first). EMPTY BY DESIGN: a capability
# routes to the deployment's configured-default order until there is EVIDENCE (from the
# evaluation layer, Phase 10) that a specific vendor serves it better. This keeps routing
# backward-compatible (the same operational default as today) and free of unjustified vendor
# opinions — inventing a vendor ranking here silently changes production routing. Provider
# strings, when added, appear ONLY here (never in business logic), e.g. gated on a benchmark:
#   _CAPABILITY_PREFERENCE = {Capability.LONG_CONTEXT: ("claude", "gemini", "azure")}
_CAPABILITY_PREFERENCE: dict[Capability, tuple[str, ...]] = {}


def route(capability: Capability) -> MLProvider:
    """Capability -> concrete MLProvider. DETERMINISTIC: the first provider in the capability's
    static preference that is configured, else the first configured provider (deterministic when
    nothing is set). Constructs nothing itself — `get_provider` (registry) owns construction."""
    if capability is Capability.EMBEDDING:
        # One home for embedding-provider selection: the registry's pinned resolver (Azure-or-
        # deterministic), so stored vector model/version can never drift via the router.
        return get_embedding_provider()
    configured = configured_providers()  # e.g. ("azure", "deterministic")
    for name in _CAPABILITY_PREFERENCE.get(capability, ()):
        if name in configured:
            return get_provider(name)
    return get_provider(configured[0])  # first configured (deterministic if nothing is set)
