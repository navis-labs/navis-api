"""Phase 5 regression tests — ReasoningModelGateway (the permanent seam).

Asserts the two-method surface (generate + embed): capability-routed, prompt-registry-resolved,
cost-accounted, telemetry best-effort, and that callers never touch a provider/vendor/router.
The router is injected with a fake provider, so no network/model is involved.
"""
from __future__ import annotations

import uuid

import pytest

import navis.ml.gateway as gw_mod
from navis.ml import GatewayRequest, GatewayResponse, ReasoningModelGateway
from navis.ml.generation.prompts import get_prompt
from navis.ml.providers.base import EmbedResult, GenRequest, GenResult, MLProvider, ProviderInfo
from navis.ml.routing import Capability
from navis.ml.telemetry import estimate_cost


class _Fake(MLProvider):
    def __init__(self):
        self.info = ProviderInfo("fake", "gpt-4o-mini", "em", 8, online=False)
        self.seen: GenRequest | None = None

    async def generate(self, req: GenRequest) -> GenResult:
        self.seen = req
        return GenResult(text="narrative", model="gpt-4o-mini", input_tokens=1000, output_tokens=500)

    async def embed(self, texts: list[str]) -> EmbedResult:
        return EmbedResult([[0.0] * 8 for _ in texts], "em", 8)


def _gw(fake, seen_caps=None):
    def router(cap):
        if seen_caps is not None:
            seen_caps.append(cap)
        return fake
    return ReasoningModelGateway(router=router)


# --- generate ---------------------------------------------------------------

async def test_generate_returns_full_accounting():
    fake = _Fake()
    resp = await _gw(fake).generate(
        GatewayRequest(capability=Capability.REASONING, user="u", system="s",
                       max_tokens=10, temperature=0.0)
    )
    assert isinstance(resp, GatewayResponse)
    assert resp.text == "narrative" and resp.model == "gpt-4o-mini"
    assert (resp.input_tokens, resp.output_tokens, resp.total_tokens) == (1000, 500, 1500)
    assert resp.cached is False
    # the inline system/content were passed straight through to the provider
    assert fake.seen.system == "s" and fake.seen.user == "u" and fake.seen.max_tokens == 10


async def test_generate_resolves_prompt_from_registry():
    fake = _Fake()
    resp = await _gw(fake).generate(
        GatewayRequest(capability=Capability.ACCURATE, user="facts", prompt_id="summary")
    )
    prompt = get_prompt("summary")
    assert fake.seen.system == prompt.system          # resolved from the ONE prompt home
    assert resp.prompt_id == "summary" and resp.prompt_version == prompt.version


async def test_generate_routes_by_capability():
    caps: list[Capability] = []
    await _gw(_Fake(), caps).generate(GatewayRequest(capability=Capability.REASONING, user="u"))
    assert caps == [Capability.REASONING]


async def test_cost_is_computed_from_price_table():
    resp = await _gw(_Fake()).generate(GatewayRequest(capability=Capability.FAST, user="u"))
    assert resp.cost_usd == estimate_cost("gpt-4o-mini", 1000, 500)  # wired to the one cost home


async def test_telemetry_failure_never_breaks_generate(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("telemetry down")

    monkeypatch.setattr(gw_mod.telemetry, "record", boom)
    resp = await _gw(_Fake()).generate(
        GatewayRequest(capability=Capability.REASONING, user="u", org_id=uuid.uuid4())
    )
    assert resp.text == "narrative"  # survived a telemetry outage


# --- embed ------------------------------------------------------------------

async def test_embed_routes_to_embedding_capability():
    caps: list[Capability] = []
    result = await _gw(_Fake(), caps).embed(["a", "b"])
    assert isinstance(result, EmbedResult) and len(result.vectors) == 2
    assert caps == [Capability.EMBEDDING]  # never via generation routing


# --- surface ----------------------------------------------------------------

def test_public_seam_is_importable_from_package_root():
    from navis.ml import GatewayRequest as R, GatewayResponse as P, ReasoningModelGateway as G

    assert (R, P, G) == (GatewayRequest, GatewayResponse, ReasoningModelGateway)
    # Two methods only — the stable surface.
    assert {"generate", "embed"} <= set(vars(ReasoningModelGateway))


def test_package_surface_is_gateway_only_no_provider_leak():
    # The Gateway is the ONLY entrypoint: navis.ml must not re-export provider implementation.
    import navis.ml as ml

    assert set(ml.__all__) == {
        "ReasoningModelGateway", "GatewayRequest", "GatewayResponse", "Capability", "EmbedResult",
    }
    for leaked in ("get_provider", "get_embedding_provider", "MLProvider", "GenRequest", "GenResult"):
        assert not hasattr(ml, leaked), f"navis.ml must not re-export provider implementation: {leaked}"
