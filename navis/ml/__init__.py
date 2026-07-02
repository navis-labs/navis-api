"""ML layer — strictly ASSISTIVE. It never owns truth.

The deterministic pipeline (Signals→…→Adaptive) is the source of truth and contains
no ML. This layer adds: a vendor-agnostic provider abstraction, append-only versioned
embeddings over Memory, org-scoped semantic retrieval, RAG grounding (the LLM only
ever sees a GroundedContext, never the database), narrative generation (Decision→
narrative, never narrative→Decision), routing, telemetry, and evals.

INVARIANT (enforced by test): no pipeline domain imports navis.ml. ML may READ the
truth layers for retrieval; it may never inject itself into them.
"""
from navis.ml.gateway import Capability, GatewayRequest, GatewayResponse, ReasoningModelGateway
from navis.ml.providers import EmbedResult  # the return type of ReasoningModelGateway.embed

# The Gateway is the ONLY entrypoint: every layer above navis.ml imports ONLY these. Provider
# construction / SDKs / routing / telemetry are IMPLEMENTATION below the gateway and are NOT
# re-exported here (infra/tests import them from their own module paths). Keeping this surface
# minimal is what lets providers/routing/telemetry evolve without touching any caller.
__all__ = [
    "ReasoningModelGateway", "GatewayRequest", "GatewayResponse", "Capability", "EmbedResult",
]
