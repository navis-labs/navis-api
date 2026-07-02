"""Regression tests for navis/ml/generation — the gateway-bypass fix.

Generation now depends ONLY on the ReasoningModelGateway: it builds a GatewayRequest (capability
+ prompt_id + grounded content), calls the gateway, and maps the response to a typed Narrative.
It no longer imports providers / routing / telemetry, and it never names a vendor.
"""
from __future__ import annotations

import ast
import os
import types
import uuid

import pytest

from navis.ml.generation import narratives
from navis.ml.rag.grounding import Grounding, GroundedContext
from navis.ml.routing import Capability

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")


class _FakeGateway:
    def __init__(self):
        self.reqs: list = []

    async def generate(self, req):
        self.reqs.append(req)
        return types.SimpleNamespace(
            text="a grounded narrative", model="test-model",
            prompt_id=req.prompt_id, prompt_version=1,
            input_tokens=1, output_tokens=1, cost_usd=0.0, cached=False,
        )


def _ctx():
    return GroundedContext(org_id=str(ORG), question="q", facts=[Grounding("memory", "m1", "closed deal")])


async def test_brief_routes_through_gateway_only(monkeypatch):
    fake = _FakeGateway()
    monkeypatch.setattr(narratives, "gateway", fake)
    ctx = _ctx()
    out = await narratives.brief(ORG, ctx)

    assert len(fake.reqs) == 1
    req = fake.reqs[0]
    assert req.capability is Capability.ACCURATE     # capability, never a vendor
    assert req.prompt_id == "brief_narrative"         # versioned prompt by id
    assert req.org_id == ORG                          # telemetry attribution owned by gateway
    assert req.user == ctx.render()                   # grounded facts only
    # typed output maps the gateway response + preserves lineage
    assert out.text == "a grounded narrative" and out.model == "test-model"
    assert out.sources == ["m1"] and out.prompt_id == "brief_narrative" and out.prompt_version == 1


@pytest.mark.parametrize(
    "fn, capability, prompt_id",
    [
        (narratives.brief, Capability.ACCURATE, "brief_narrative"),
        (narratives.summary, Capability.FAST, "summary"),
        (narratives.draft, Capability.FAST, "draft"),
        (narratives.explanation, Capability.ACCURATE, "explanation"),
    ],
)
async def test_task_maps_to_capability_and_prompt(monkeypatch, fn, capability, prompt_id):
    fake = _FakeGateway()
    monkeypatch.setattr(narratives, "gateway", fake)
    await fn(ORG, _ctx())
    assert fake.reqs[0].capability is capability
    assert fake.reqs[0].prompt_id == prompt_id


def test_generation_depends_only_on_gateway_and_grounding():
    """Invariant lock: narratives must not import providers / routing / telemetry / SDKs."""
    src = os.path.join(os.path.dirname(narratives.__file__), "narratives.py")
    tree = ast.parse(open(src).read())
    navis_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("navis"):
            navis_modules.add(node.module)
    assert navis_modules <= {"navis.ml.gateway", "navis.ml.rag.grounding"}, navis_modules
    for forbidden in ("navis.ml.providers", "navis.ml.routing", "navis.ml.telemetry"):
        assert forbidden not in navis_modules


# --- hardening pass: prompt registry + fail-fast metadata --------------------

def test_prompt_validates_at_construction():
    from navis.ml.generation.prompts import Prompt

    Prompt("ok", 1, "instruction")  # valid
    with pytest.raises(ValueError):
        Prompt("", 1, "instruction")          # empty id
    with pytest.raises(ValueError):
        Prompt("ok", 0, "instruction")        # version < 1
    with pytest.raises(ValueError):
        Prompt("ok", 1, "")                   # empty system


def test_registry_is_immutable():
    from navis.ml.generation.prompts import REGISTRY

    with pytest.raises(TypeError):            # MappingProxyType rejects mutation
        REGISTRY["brief_narrative"] = None    # type: ignore[index]
    assert "brief_narrative" in REGISTRY      # still readable


def test_get_prompt_is_fail_fast():
    from navis.ml.generation.prompts import get_prompt

    assert get_prompt("summary").version >= 1
    with pytest.raises(KeyError):             # never invents a prompt, never returns None
        get_prompt("does-not-exist")


async def test_narrative_fails_fast_without_gateway_metadata(monkeypatch):
    class _NoMeta:
        async def generate(self, req):
            return types.SimpleNamespace(
                text="x", model="m", prompt_id=None, prompt_version=None,
                input_tokens=0, output_tokens=0, cost_usd=0.0, cached=False,
            )

    monkeypatch.setattr(narratives, "gateway", _NoMeta())
    with pytest.raises(RuntimeError, match="prompt metadata"):  # never fabricate version 0
        await narratives.brief(ORG, _ctx())
