"""AI reasoning source (navis.reasoning.ai): same ReasoningSignal contract, gateway-only
dependency, hallucination rejection, fail-open isolation, deterministic identity, and the
deterministic-first merge. The deterministic v2 pass is never modified — these tests prove the
AI source is purely additive."""
from __future__ import annotations

import ast
import json
import pathlib
import re
import types

from navis.memory.envelope import EntityType
from navis.reasoning import ai
from navis.reasoning.v2 import run_reasoning_v2
from navis.reasoning.v2.contract import REASONING_VERSION, SignalType
from navis.state.v2.builders import build_projection
from navis.state.v2.models import OrganizationalState
from tests.state.fakes import ORG_A, corpus, rec

AI_PY = pathlib.Path(ai.__file__)


def _state(org=ORG_A, *, stale: bool = True) -> OrganizationalState:
    records = corpus(org)
    if stale:
        # A late "heartbeat" pushes as_of ~40 days ahead so the deterministic risk engines fire.
        records = records + [
            rec(EntityType.PERSON, "heartbeat@acme.com",
                {"email": "heartbeat@acme.com", "name": "HB", "role": "from"},
                offset=40 * 86400, tag="heartbeat", org_id=org)
        ]
    return build_projection(org, records)


class _FakeGateway:
    """Mirrors GatewayResponse's shape. `payload` may be a list (serialized to JSON), a raw
    string, or an Exception to raise — one fake for every failure mode."""

    def __init__(self, payload):
        self.payload = payload
        self.reqs: list = []

    async def generate(self, req):
        self.reqs.append(req)
        if isinstance(self.payload, Exception):
            raise self.payload
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return types.SimpleNamespace(
            text=text, model="test-model", prompt_id=req.prompt_id, prompt_version=1,
            input_tokens=1, output_tokens=1, cost_usd=0.0, cached=False,
        )


def _item(state, **overrides):
    c = state.commitments()[0]
    base = {
        "signal_type": "commitment_slipping",
        "entity_type": "commitment",
        "entity_id": c.entity_id,
        "title": "Deck commitment is at risk",
        "why": "The only open commitment has aged with no follow-up in any thread.",
        "confidence": 0.6,
        "severity": 0.8,
        "evidence": {"age_days": 40},
    }
    base.update(overrides)
    return base


# ---- contract parity: AI signals ARE ReasoningSignals -------------------------------------

async def test_ai_signals_share_the_reasoning_contract(monkeypatch):
    state = _state()
    fake = _FakeGateway([_item(state)])
    monkeypatch.setattr(ai, "gateway", fake)
    signals = await ai.run_ai_reasoning(state)

    assert len(signals) == 1
    s = signals[0]
    # Full provenance, identical to the deterministic stamping law.
    assert s.org_id == state.org_id
    assert s.engine_name == "ai_reasoning" and s.engine_version == ai.AIReasoningEngine.version
    assert s.state_as_of == state.as_of and s.created_at == state.as_of  # logical clock, no now()
    assert s.state_watermark == str(state.watermark.last_memory_id)
    assert s.reasoning_version == REASONING_VERSION
    assert s.signal_type is SignalType.COMMITMENT_SLIPPING
    assert 0.0 <= s.confidence <= ai.AI_CONFIDENCE_CAP and 0.0 <= s.severity <= 1.0
    # Model/prompt attribution lands in evidence (eval + replay lineage).
    assert s.evidence["model"] == "test-model"
    assert s.evidence["prompt_id"] == "reasoning_signals" and s.evidence["prompt_version"] == 1
    assert s.evidence["age_days"] == 40  # model-cited evidence preserved

    # The one gateway call is capability-addressed, prompt-versioned, org-attributed, cold.
    req = fake.reqs[0]
    assert req.capability.value == "reasoning" and req.prompt_id == "reasoning_signals"
    assert req.org_id == state.org_id and req.temperature == 0.0
    assert req.user == ai.render_state(state)  # the model sees ONLY the normalized state


async def test_same_response_produces_identical_signals(monkeypatch):
    state = _state()
    monkeypatch.setattr(ai, "gateway", _FakeGateway([_item(state)]))
    a = [s.to_dict() for s in await ai.run_ai_reasoning(state)]
    b = [s.to_dict() for s in await ai.run_ai_reasoning(state)]
    assert a == b  # same state + same model output → byte-identical signals (incl. signal_id)


async def test_identity_survives_model_rephrasing(monkeypatch):
    # dedup_key is DERIVED (entity_id), never model-chosen: a rephrased title/why yields the
    # SAME signal_id, so re-runs converge instead of accumulating near-duplicates.
    state = _state()
    monkeypatch.setattr(ai, "gateway", _FakeGateway([_item(state, title="Slipping deck promise")]))
    first = await ai.run_ai_reasoning(state)
    monkeypatch.setattr(ai, "gateway", _FakeGateway([_item(state, title="Deck promise going stale")]))
    second = await ai.run_ai_reasoning(state)
    assert first[0].signal_id == second[0].signal_id


async def test_restatements_within_one_pass_collapse(monkeypatch):
    state = _state()
    monkeypatch.setattr(
        ai, "gateway", _FakeGateway([_item(state), _item(state, title="Same finding, reworded")])
    )
    signals = await ai.run_ai_reasoning(state)
    assert len(signals) == 1  # same (type, entity) → same id → collapsed, keep-first


# ---- grounding + validation: fail-closed per item ------------------------------------------

async def test_hallucinated_entity_is_rejected(monkeypatch):
    state = _state()
    monkeypatch.setattr(
        ai, "gateway",
        _FakeGateway([_item(state, entity_id="made-up-id"), _item(state)]),
    )
    signals = await ai.run_ai_reasoning(state)
    assert len(signals) == 1  # invented citation dropped, grounded item survives
    assert signals[0].entity_id == state.commitments()[0].entity_id


async def test_unknown_signal_type_is_rejected(monkeypatch):
    state = _state()
    monkeypatch.setattr(
        ai, "gateway", _FakeGateway([_item(state, signal_type="world_domination")])
    )
    assert await ai.run_ai_reasoning(state) == []


async def test_confidence_capped_and_severity_clamped(monkeypatch):
    state = _state()
    monkeypatch.setattr(
        ai, "gateway", _FakeGateway([_item(state, confidence=0.99, severity=1.7)])
    )
    s = (await ai.run_ai_reasoning(state))[0]
    assert s.confidence == ai.AI_CONFIDENCE_CAP  # AI never outranks deterministic confidence
    assert s.severity == 1.0


async def test_malformed_output_yields_no_signals(monkeypatch):
    state = _state()
    monkeypatch.setattr(ai, "gateway", _FakeGateway("I think everything looks fine!"))
    assert await ai.run_ai_reasoning(state) == []  # e.g. the offline deterministic provider


async def test_fenced_json_is_tolerated(monkeypatch):
    state = _state()
    fenced = "```json\n" + json.dumps([_item(state)]) + "\n```"
    monkeypatch.setattr(ai, "gateway", _FakeGateway(fenced))
    assert len(await ai.run_ai_reasoning(state)) == 1


# ---- failure isolation: deterministic reasoning is the floor --------------------------------

async def test_gateway_failure_is_isolated(monkeypatch):
    state = _state()
    monkeypatch.setattr(ai, "gateway", _FakeGateway(RuntimeError("provider down")))
    assert await ai.run_ai_reasoning(state) == []  # never raises, never blocks the pipeline


async def test_empty_state_never_spends_a_model_call(monkeypatch):
    fake = _FakeGateway([])
    monkeypatch.setattr(ai, "gateway", fake)
    assert await ai.run_ai_reasoning(build_projection(ORG_A, [])) == []
    assert fake.reqs == []


# ---- merge: deterministic-first, runner invariants preserved --------------------------------

async def test_merge_deterministic_wins_and_ai_only_adds(monkeypatch):
    state = _state()
    det = run_reasoning_v2(state)
    det_ids = {s.signal_id for s in det}
    slipping = next(s for s in det if s.signal_type is SignalType.COMMITMENT_SLIPPING)

    monkeypatch.setattr(
        ai, "gateway",
        _FakeGateway([
            # Restates a deterministic finding: same (signal_type, entity_id) → suppressed.
            _item(state, entity_id=slipping.entity_id),
            # Genuinely additive org-wide interpretation → kept.
            _item(state, signal_type="org_drift", entity_type=None, entity_id=None,
                  title="Compound drift", why="Slippage and cooling point the same direction."),
        ]),
    )
    merged = ai.merge_signals(det, await ai.run_ai_reasoning(state))

    ids = [s.signal_id for s in merged]
    assert len(ids) == len(set(ids))                       # runner invariant: unique ids
    assert det_ids <= set(ids)                             # every deterministic signal survives
    by_engine = [s for s in merged if s.engine_name == "ai_reasoning"]
    assert [s.signal_type for s in by_engine] == [SignalType.ORG_DRIFT]  # duplicate suppressed
    assert [(-s.severity, str(s.signal_id)) for s in merged] == sorted(
        (-s.severity, str(s.signal_id)) for s in merged
    )                                                      # runner invariant: ordering law


def test_merge_with_no_ai_signals_is_identity():
    det = run_reasoning_v2(_state())
    assert [s.to_dict() for s in ai.merge_signals(det, [])] == [s.to_dict() for s in det]


async def test_decision_consumes_ai_signals_identically(monkeypatch):
    # Q8 proof: Decision ranks the merged list with zero knowledge of the source.
    from navis.decisions.what_matters_today import WhatMattersTodayEngine

    state = _state()
    monkeypatch.setattr(
        ai, "gateway",
        _FakeGateway([_item(state, signal_type="org_drift", entity_type=None, entity_id=None,
                            confidence=0.6, severity=0.9)]),
    )
    merged = ai.merge_signals(run_reasoning_v2(state), await ai.run_ai_reasoning(state))
    decisions = WhatMattersTodayEngine().decide(
        merged, org_id=state.org_id, reasoning_version=REASONING_VERSION,
        state_watermark=str(state.watermark.last_memory_id), as_of=state.as_of,
    )
    ai_signal_ids = {str(s.signal_id) for s in merged if s.engine_name == "ai_reasoning"}
    assert any(set(d.reasoning_signal_ids) & ai_signal_ids for d in decisions)


# ---- architecture locks ----------------------------------------------------------------------

def test_ai_module_imports_only_the_gateway_from_ml():
    """The ONLY ML dependency is navis.ml.gateway — never providers/SDKs/routing/prompts/
    telemetry/embeddings/retrieval/grounding/cache. Everything else it touches is the frozen
    reasoning contract, state models, and platform logging."""
    allowed = {
        "navis.ml.gateway",
        "navis.platform.monitoring",
        "navis.reasoning.v2.base",
        "navis.reasoning.v2.contract",
        "navis.reasoning.v2.service",
        "navis.state.v2.models",
    }
    navis_modules: set[str] = set()
    for node in ast.walk(ast.parse(AI_PY.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("navis"):
            navis_modules.add(node.module)
        elif isinstance(node, ast.Import):
            navis_modules.update(a.name for a in node.names if a.name.startswith("navis"))
    assert navis_modules <= allowed, navis_modules
    ml_imports = {m for m in navis_modules if m.startswith("navis.ml")}
    assert ml_imports == {"navis.ml.gateway"}


def test_stamp_reuse_is_the_v2_function():
    # One home for draft→signal stamping: the AI source reuses v2's, so provenance can never
    # drift between sources. If v2 renames it, this fails loudly instead of silently forking.
    from navis.reasoning.v2 import service

    assert ai._stamp is service._stamp


def test_prompt_lists_every_signal_type():
    # prompts.py must never import reasoning (ml→pipeline isolation), so enum↔prompt sync is
    # enforced here instead of by import.
    from navis.ml.generation.prompts import get_prompt

    system = get_prompt(ai.PROMPT_ID).system
    for t in SignalType:
        assert t.value in system, f"prompt is missing signal_type {t.value}"


def test_no_connector_names_in_ai_module_or_prompt():
    # Reasoning sees organizational reality only — never applications/connectors.
    from navis.ml.generation.prompts import get_prompt

    text = (AI_PY.read_text() + " " + get_prompt(ai.PROMPT_ID).system).lower()
    for name in ("slack", "gmail", "calendar", "notion", "hubspot", "jira", "github", "crm", "drive"):
        assert not re.search(rf"\b{name}\b", text), f"connector name {name!r} leaked into reasoning"


def test_deterministic_v2_package_is_untouched_by_the_ai_source():
    # The AI source must not have introduced any navis.ml/navis.reasoning.ai import into the
    # frozen deterministic package (belt over the existing v2 AST test's suspenders).
    v2_root = AI_PY.parent / "v2"
    for py in v2_root.rglob("*.py"):
        assert "reasoning.ai" not in py.read_text()
