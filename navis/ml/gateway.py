"""ReasoningModelGateway — the ONE stable seam between the AI platform (navis.ml) and every
layer above it (Reasoning, Decision, Execution, Observation, Learning, Intelligence, Adaptive).

Those layers import ONLY this module (the gateway class + GatewayRequest/GatewayResponse). They
never import a provider, a vendor SDK, the router, the prompt registry, or telemetry — the
gateway composes all of that behind two methods:

    async def generate(GatewayRequest) -> GatewayResponse
    async def embed(list[str])          -> EmbedResult

Design contract (kept EXTREMELY small and stable — this is a 10-year interface):
  * The two method SIGNATURES never change. New capabilities (structured output, tool calling,
    streaming) arrive as ADDITIVE optional fields on GatewayRequest/GatewayResponse, or as a new
    method — never by changing these two.
  * Callers pass a CAPABILITY, never a model/vendor name. Routing (deterministic, config-order)
    + provider resilience (timeout/retry/breaker) + prompt resolution + telemetry all live below
    this line, so a provider swap changes nothing for callers.
  * Determinism/replay: generation is non-deterministic; the `cached` field + a request
    fingerprint are the seam for the Phase-12 response cache that makes it replayable. Today
    `cached` is always False (no cache wired yet).
  * Telemetry is best-effort: a telemetry failure never fails a model call.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from navis.ml.providers.base import EmbedResult, GenRequest
from navis.ml.routing import Capability, route
from navis.ml.telemetry import estimate_cost, telemetry
from navis.platform.monitoring import get_logger

logger = get_logger("ml.gateway")


@dataclass(frozen=True, slots=True)
class GatewayRequest:
    """What a layer asks the AI platform for — capability + content, never a vendor.

    Either supply `system` inline, or reference a versioned prompt via `prompt_id` (resolved
    from the prompt registry; its version is stamped on the response for replay/eval). Future
    modes (tools, response_format, stream, seed) arrive as additional optional fields here."""

    capability: Capability
    user: str
    system: str = ""
    prompt_id: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.2
    org_id: uuid.UUID | None = None  # for per-org telemetry/cost attribution


@dataclass(frozen=True, slots=True)
class GatewayResponse:
    """The platform's answer — text + full accounting. Structured output / tool calls arrive as
    additive optional fields (Phases 7/8)."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False
    prompt_id: str | None = None
    prompt_version: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ReasoningModelGateway:
    """The single public entry to navis.ml for every layer above it. Stateless; a shared
    instance is safe. The router is injectable for tests; production composes the real stack."""

    def __init__(self, *, router=None) -> None:
        # `router` maps a Capability -> a (resilient) MLProvider; defaults to the real router.
        self._route = router or route

    async def generate(self, req: GatewayRequest) -> GatewayResponse:
        system, prompt_id, prompt_version = self._resolve_prompt(req)
        provider = self._route(req.capability)
        result = await provider.generate(
            GenRequest(
                system=system,
                user=req.user,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
            )
        )
        # Cost for the response is computed independently of telemetry so a telemetry outage
        # can never change or fail the returned answer (best-effort observability).
        cost = estimate_cost(result.model, result.input_tokens, result.output_tokens)
        if req.org_id is not None:
            try:
                telemetry.record(req.org_id, result, prompt_id=prompt_id, prompt_version=prompt_version)
            except Exception:  # noqa: BLE001 — telemetry is observability, never correctness
                logger.exception("gateway telemetry record failed (best-effort)")
        return GatewayResponse(
            text=result.text,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=cost,
            cached=False,  # Phase 12 response cache flips this
            prompt_id=prompt_id,
            prompt_version=prompt_version,
        )

    async def embed(self, texts: list[str]) -> EmbedResult:
        """Vectors for retrieval/similarity. Routes to the pinned embedding provider (via the
        EMBEDDING capability), so stored model/version can never drift through the gateway."""
        return await self._route(Capability.EMBEDDING).embed(texts)

    def _resolve_prompt(self, req: GatewayRequest) -> tuple[str, str | None, int | None]:
        if req.prompt_id is not None:
            # Lazy import: the prompt registry currently lives under navis.ml.generation, which
            # ALSO consumes the gateway (narratives) — importing it at module load would create a
            # cycle. Deferring to call time breaks it. (Proper future fix: relocate the prompt
            # registry to a platform-level home so the gateway never imports the generation pkg.)
            from navis.ml.generation.prompts import get_prompt

            prompt = get_prompt(req.prompt_id)
            return prompt.system, prompt.id, prompt.version
        return req.system, None, None


# Convenience shared instance (stateless). Callers may also construct their own.
gateway = ReasoningModelGateway()
