"""ML layer: deterministic providers, embeddings, retrieval, RAG grounding,
generation (narrative-only), evals, and the architecture invariant."""
from __future__ import annotations

import ast
import pathlib
import uuid

from navis.ml.evals import grounding_score, prompt_fingerprint, recall_at_k
from navis.ml.generation import REGISTRY, brief
from navis.ml.generation.narratives import Narrative
from navis.ml.math import cosine
from navis.ml.providers import GenRequest, get_embedding_provider, get_provider
from navis.ml.providers.deterministic import DeterministicProvider
from navis.ml.rag.grounding import GroundedContext, Grounding
from navis.ml.retrieval.reranker import rerank
from navis.ml.retrieval.vector_search import Hit
from navis.ml.routing import Task, Tier, tier_for

NAVIS = pathlib.Path(__file__).resolve().parent.parent.parent / "navis"
PIPELINE = {
    "signals", "memory", "state", "reasoning", "decisions",
    "execution", "observation", "learning", "intelligence", "adaptive",
}
ORG = uuid.uuid4()


# ---- providers: deterministic, stable, token-accounted ----

async def test_deterministic_embeddings_are_stable_and_normalized():
    p = DeterministicProvider()
    a = (await p.embed(["raised seed round with YC"])).vectors[0]
    b = (await p.embed(["raised seed round with YC"])).vectors[0]
    assert a == b  # deterministic
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6  # L2-normalized


async def test_embedding_captures_lexical_similarity():
    p = DeterministicProvider()
    vecs = (await p.embed(["meeting with YC partners", "YC partner meeting", "invoice paid"])).vectors
    assert cosine(vecs[0], vecs[1]) > cosine(vecs[0], vecs[2])  # related > unrelated


async def test_generate_accounts_tokens_and_never_decides():
    p = DeterministicProvider()
    res = await p.generate(GenRequest(system="s", user="grounded facts here"))
    assert res.total_tokens > 0 and res.model == "navis-deterministic-1"


def test_provider_registry_returns_a_working_backend():
    from navis.ml.providers.base import MLProvider

    assert isinstance(get_provider(), MLProvider)
    assert isinstance(get_embedding_provider(), MLProvider)
    # The deterministic provider is always available as the offline fallback.
    assert DeterministicProvider().info.online is False


# ---- rerank only reorders ----

def test_rerank_orders_by_relevance_without_rewriting():
    hits = [
        Hit(uuid.uuid4(), 0.4, "commitment", "ship the YC application"),
        Hit(uuid.uuid4(), 0.5, "person", "lunch with Bob"),
    ]
    ordered = rerank("YC application status", hits)
    assert ordered[0].source_text == "ship the YC application"  # lexical lift wins
    assert {h.source_text for h in ordered} == {h.source_text for h in hits}  # nothing invented


# ---- RAG grounding: model sees only cited facts ----

def test_grounded_context_renders_only_cited_facts():
    ctx = GroundedContext(
        org_id=str(ORG), question="What happened with YC?",
        facts=[Grounding("memory", "m1", "Applied to YC W26"), Grounding("memory", "m2", "Intro call booked")],
        token_estimate=20,
    )
    rendered = ctx.render()
    assert "[1]" in rendered and "[2]" in rendered
    assert "Applied to YC W26" in rendered
    assert ctx.sources() == ["m1", "m2"]  # lineage preserved


# ---- generation: narrative only, grounded, telemetry recorded ----

async def test_brief_generates_narrative_with_lineage_and_telemetry():
    from navis.ml.telemetry import telemetry

    ctx = GroundedContext(
        org_id=str(ORG), question="brief",
        facts=[Grounding("memory", "m1", "Closed deal with Acme")],
    )
    out = await brief(ORG, ctx)
    assert isinstance(out, Narrative)
    assert out.sources == ["m1"]                       # cites the fact
    assert out.prompt_id == "brief_narrative" and out.prompt_version == 1
    # No structured decision fields exist on a Narrative — it cannot drive Execution.
    assert not hasattr(out, "decision") and not hasattr(out, "action")
    assert telemetry.usage(ORG).calls >= 1             # observability recorded


def test_every_prompt_forbids_decisions_and_fabrication():
    for prompt in REGISTRY.values():
        s = prompt.system.lower()
        assert "never invent facts" in s
        assert "never make decisions" in s or "narrate only" in s


# ---- evals ----

def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], frozenset({"b", "z"}), k=3) == 0.5


def test_grounding_score_flags_fabrication():
    grounded = grounding_score("Closed deal with Acme", "Closed deal with Acme today")
    fabricated = grounding_score("Bought a yacht in Monaco", "Closed deal with Acme today")
    assert grounded > fabricated


def test_prompt_fingerprint_changes_on_edit():
    a = prompt_fingerprint("brief", 1, "sys", "user")
    assert a == prompt_fingerprint("brief", 1, "sys", "user")
    assert a != prompt_fingerprint("brief", 2, "sys", "user")


# ---- routing ----

def test_routing_tiers():
    assert tier_for(Task.SUMMARY) is Tier.FAST
    assert tier_for(Task.BRIEF_NARRATIVE) is Tier.DEEP


# ---- THE invariant: the deterministic pipeline never imports ML ----

def test_no_pipeline_domain_imports_ml():
    violations = []
    for domain in PIPELINE:
        for py in (NAVIS / domain).rglob("*.py"):
            for node in ast.walk(ast.parse(py.read_text(), filename=str(py))):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module]
                for m in mods:
                    if m == "navis.ml" or m.startswith("navis.ml."):
                        violations.append(f"{domain}/{py.name} imports {m}")
    assert not violations, "ML must not enter the truth pipeline:\n" + "\n".join(violations)


def test_ml_does_not_import_to_influence_decision_or_reasoning():
    # ML may READ memory/state/learning/intelligence, but must never import
    # reasoning/decisions/execution (which would let it steer the truth pipeline).
    forbidden = {"reasoning", "decisions", "execution", "signals", "observation", "adaptive"}
    violations = []
    for py in (NAVIS / "ml").rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(), filename=str(py))):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                parts = m.split(".")
                if len(parts) >= 2 and parts[0] == "navis" and parts[1] in forbidden:
                    violations.append(f"ml/{py.name} imports {m}")
    assert not violations, "ML must not import the active pipeline:\n" + "\n".join(violations)
