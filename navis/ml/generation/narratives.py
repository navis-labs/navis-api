"""Narrative generation — briefs, summaries, drafts, explanations (the PRODUCT generation layer).

This layer sits ABOVE the gateway. It ONLY: builds a product request, names a CAPABILITY (never
a vendor), references a versioned prompt by id, passes a GroundedContext (the model only ever
sees grounded facts), calls the stable ReasoningModelGateway, and returns a typed Narrative
(text + cited sources — never a decision). It owns NO providers, routing, retries, or telemetry —
the gateway owns all of that. The only ML dependency is the gateway (+ GroundedContext, its input).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from navis.ml.gateway import Capability, GatewayRequest, gateway
from navis.ml.rag.grounding import GroundedContext


@dataclass(frozen=True, slots=True)
class Narrative:
    """Typed generation output: narrated text + the lineage of the facts it was grounded on."""

    text: str
    sources: list[str]
    prompt_id: str
    prompt_version: int
    model: str


async def _generate(
    org_id: uuid.UUID, *, capability: Capability, prompt_id: str, context: GroundedContext
) -> Narrative:
    """Build one product generation request and call the gateway. The gateway resolves the
    versioned prompt, routes the capability to a provider (with resilience), and records
    telemetry — generation never touches any of that."""
    response = await gateway.generate(
        GatewayRequest(
            capability=capability,
            user=context.render(),   # grounded facts ONLY (the model never sees raw data)
            prompt_id=prompt_id,
            org_id=org_id,
        )
    )
    # The gateway ALWAYS stamps the resolved prompt id + version for a prompt_id request (see
    # ReasoningModelGateway._resolve_prompt). Fail-fast if it did not, rather than fabricate
    # invalid metadata (e.g. version 0) that would corrupt telemetry / eval attribution.
    if response.prompt_id is None or response.prompt_version is None:
        raise RuntimeError(f"gateway returned no prompt metadata for prompt_id={prompt_id!r}")
    return Narrative(
        text=response.text,
        sources=context.sources(),            # lineage preserved from the grounding
        prompt_id=response.prompt_id,
        prompt_version=response.prompt_version,
        model=response.model,
    )


# Product task → (capability, versioned prompt id). The task→capability choice is a PRODUCT
# decision owned HERE (briefs/explanations want the strongest model; summaries/drafts the fast
# one), expressed as a CAPABILITY — never a vendor.
async def brief(org_id: uuid.UUID, context: GroundedContext) -> Narrative:
    return await _generate(org_id, capability=Capability.ACCURATE, prompt_id="brief_narrative", context=context)


async def summary(org_id: uuid.UUID, context: GroundedContext) -> Narrative:
    return await _generate(org_id, capability=Capability.FAST, prompt_id="summary", context=context)


async def draft(org_id: uuid.UUID, context: GroundedContext) -> Narrative:
    return await _generate(org_id, capability=Capability.FAST, prompt_id="draft", context=context)


async def explanation(org_id: uuid.UUID, context: GroundedContext) -> Narrative:
    return await _generate(org_id, capability=Capability.ACCURATE, prompt_id="explanation", context=context)
