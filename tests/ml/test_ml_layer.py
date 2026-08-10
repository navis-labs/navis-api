"""ML layer: deterministic providers, embeddings, retrieval, RAG grounding,
generation (narrative-only), evals, and the architecture invariant."""
from __future__ import annotations

import ast
import pathlib
import uuid

from navis.ml.evals import grounding_score, prompt_fingerprint, recall_at_k
from navis.ml.generation import REGISTRY, brief, recommendation
from navis.ml.generation.narratives import Narrative
from navis.ml.math import cosine
from navis.ml.providers import GenRequest, get_embedding_provider, get_provider
from navis.ml.providers.deterministic import DeterministicProvider
from navis.ml.rag.grounding import GroundedContext, Grounding
from navis.ml.rag.time_window import parse_time_window
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


def test_rerank_drops_hits_below_min_score():
    # Without a floor, weakly-related memories always slip into "top K" and
    # get rendered as evidence on every turn (real-world: a generic ROCE
    # commitment kept surfacing on unrelated questions).
    hits = [
        Hit(uuid.uuid4(), 0.9, "commitment", "ship the YC application"),
        Hit(uuid.uuid4(), 0.05, "commitment", "unrelated fluff"),
    ]
    ordered = rerank("YC application status", hits, min_score=0.2)
    assert [h.source_text for h in ordered] == ["ship the YC application"]


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


def test_grounded_context_renders_event_dates_when_present():
    # The LLM needs the event date next to each fact to answer time-scoped
    # questions ("meetings this week"). Without it, retrieval can narrow to
    # the right window but the model still says "I can't tell dates."
    from datetime import datetime, timezone

    ctx = GroundedContext(
        org_id=str(ORG),
        question="which meetings are this week?",
        facts=[
            Grounding("memory", "m1", "Investor sync",
                      occurred_at=datetime(2026, 7, 29, 14, 0, tzinfo=timezone.utc)),
            Grounding("memory", "m2", "Team standup"),  # no timestamp — legacy path
        ],
    )
    rendered = ctx.render()
    assert "[2026-07-29]" in rendered      # date inlined so the LLM can see it
    assert "Investor sync" in rendered
    assert "Team standup" in rendered      # dateless facts still render cleanly


def test_grounded_context_renders_today_anchor_when_as_of_set():
    # Without an anchor, the LLM guesses what "today"/"tomorrow" mean from fact
    # dates or its training window — that's how "tomorrow" answered as yesterday.
    from datetime import datetime, timezone

    ctx = GroundedContext(
        org_id=str(ORG),
        question="any meeting tomorrow?",
        facts=[Grounding("memory", "m1", "Investor sync")],
        as_of=datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc),
    )
    assert "Today's date is: 2026-07-28" in ctx.render()


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


def test_every_prompt_forbids_fabrication_and_action():
    # Universal invariant: even the advisory `recommendation` prompt must ground on
    # real facts and must not take autonomous actions.
    for prompt in REGISTRY.values():
        s = prompt.system.lower()
        assert "never invent facts" in s, f"{prompt.id} does not forbid fabrication"
        assert "never take actions" in s, f"{prompt.id} does not forbid actions"


def test_narrative_prompts_stay_narrate_only():
    # Narrative generators must not author decisions/priorities — that is what the
    # dedicated `recommendation` prompt is for.
    narrate_only = {"brief_narrative", "summary", "draft", "explanation"}
    for pid in narrate_only:
        s = REGISTRY[pid].system.lower()
        assert "narrate only" in s or "never make decisions" in s


def test_recommendation_prompt_is_advisory_and_grounded():
    # The one prompt allowed to prioritize — but only over facts it was given,
    # framed as advisory, never as an autonomous decision.
    s = REGISTRY["recommendation"].system.lower()
    assert "advisor" in s or "advisory" in s
    assert "cite" in s
    assert "never invent facts" in s and "never take actions" in s


async def test_recommendation_generates_narrative_end_to_end():
    ctx = GroundedContext(
        org_id=str(ORG),
        question="what should I do first?",
        facts=[Grounding("memory", "m1", "Investor intro reply pending 3 days")],
    )
    out = await recommendation(ORG, ctx)
    assert isinstance(out, Narrative)
    assert out.prompt_id == "recommendation" and out.prompt_version == 1
    assert out.sources == ["m1"]  # lineage still preserved on advisory prompts


def test_chat_router_routes_prioritization_to_recommendation_prompt():
    # Wire intent stays "question", but "what should I do first?" must be
    # detected so the chat pipeline picks the advisory generator. Without this
    # detector, the request falls back to `explanation`, whose prompt refuses
    # to prioritize — exactly the bug we saw in production traces.
    from navis.api.chat_router import _is_prioritization

    assert _is_prioritization("what should I do first?")
    assert _is_prioritization("What are my top priorities today?")
    assert _is_prioritization("what's most urgent right now")
    assert _is_prioritization("where should I start")
    # Non-prioritization questions must NOT route to `recommendation`.
    assert not _is_prioritization("who am I keeping waiting")
    assert not _is_prioritization("summarise this week")
    assert not _is_prioritization("why did we push the launch")


def test_chat_router_routes_summarize_to_summary_prompt():
    # "Summarise this week" was going through `explanation`, which produced a
    # raw fact dump instead of the What happened / Why it matters / What next
    # narrative the product spec requires.
    from navis.api.chat_router import _is_summarize

    assert _is_summarize("summarise this week")
    assert _is_summarize("summarize yesterday")
    assert _is_summarize("give me a recap")
    assert _is_summarize("tl;dr on the deal")
    assert _is_summarize("catch me up")
    assert not _is_summarize("what should I do first?")
    assert not _is_summarize("who am I keeping waiting")


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

# Narrowly-scoped, deliberate exceptions. The invariant protects TRUTH: no fact
# about the organisation may be authored by a model. A module earns a place here
# only when it uses an LLM to interpret the USER'S OWN words — never to produce
# a fact — and when everything it extracts is either regex-derivable or shown to
# the user for confirmation before it can cause a side effect.
#
# `reasoning/scheduling/extract.py`: slot-fills a scheduling utterance
# ("fix a call with Rahul at 3") into date/time/attendee fields. Every slot has
# a deterministic regex path that WINS over the model's answer (see
# `extract_constraints`'s merge), the LLM only fills what regex left null, and
# nothing reaches Google without an explicit human approval of a rendered card.
_ML_IMPORT_EXCEPTIONS = {
    "reasoning/extract.py",
}


def test_no_pipeline_domain_imports_ml():
    violations = []
    for domain in PIPELINE:
        for py in (NAVIS / domain).rglob("*.py"):
            # Python source is UTF-8 by definition (PEP 3120). `read_text()`
            # with no encoding uses the platform default, which is cp1252 on
            # Windows — so any source file containing a character outside that
            # codepage failed this test with a UnicodeDecodeError that had
            # nothing to do with the invariant being checked.
            source = py.read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source, filename=str(py))):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module]
                for m in mods:
                    if m == "navis.ml" or m.startswith("navis.ml."):
                        where = f"{domain}/{py.name}"
                        if where in _ML_IMPORT_EXCEPTIONS:
                            continue
                        violations.append(f"{where} imports {m}")
    assert not violations, "ML must not enter the truth pipeline:\n" + "\n".join(violations)


def test_ml_import_exceptions_are_all_still_real():
    """An exception that no longer applies is a hole in the invariant.

    If a listed module stops importing navis.ml (or is deleted), this fails so
    the entry gets removed rather than quietly widening what's allowed.
    """
    stale = []
    for entry in _ML_IMPORT_EXCEPTIONS:
        domain, _, name = entry.partition("/")
        matches = list((NAVIS / domain).rglob(name))
        if not matches:
            stale.append(f"{entry}: no such file")
            continue
        if not any(
            "navis.ml" in m.read_text(encoding="utf-8") for m in matches
        ):
            stale.append(f"{entry}: no longer imports navis.ml")
    assert not stale, "stale ML-import exceptions:\n" + "\n".join(stale)


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


# ---- embedding subscriber: memory-layer events → memory_embeddings ----

def test_embedding_subscriber_registered_for_every_memory_event():
    """The chat endpoint retrieves from memory_embeddings only — if any of the
    five memory-layer pointer events lacks the embedding subscriber, new records
    of that entity type would land in memory but never in the search index, and
    /chat would silently miss them.
    """
    import navis.registry  # composition-root — registers the subscriber  # noqa: F401
    from navis.platform.events import EventType, event_bus

    memory_events = (
        EventType.PERSON_DETECTED,
        EventType.PROJECT_DETECTED,
        EventType.COMMITMENT_DETECTED,
        EventType.DECISION_DETECTED,
        EventType.RELATIONSHIP_UPDATED,
    )
    for evt in memory_events:
        handler_names = {h.__name__ for h in event_bus._handlers.get(evt, [])}
        assert "_embed_memory_pointer" in handler_names, (
            f"{evt.value} has no embedding subscriber — chat won't find records "
            f"of this entity type"
        )


def test_azure_provider_gates_embedding_on_separate_deployment():
    """Azure routes by deployment name; chat and embeddings need SEPARATE
    deployments. If only chat is configured, the registry must not send embed
    calls to Azure (they will 404) — it must fall back to the deterministic
    embedder so /chat's retrieval still populates memory_embeddings.
    """
    from navis.ml.providers.azure import AzureProvider
    from navis.ml.providers.deterministic import DeterministicProvider
    from navis.ml.providers.registry import get_embedding_provider
    from navis.platform.config import settings

    # `embed_deployment=None` means "fall back to configuration", so the
    # chat-only case must be built with the setting cleared — otherwise this
    # assertion silently depends on the deploying environment and starts failing
    # the moment a real embed deployment is configured.
    get_embedding_provider.cache_clear()
    configured = settings.azure_openai_embed_deployment
    try:
        settings.azure_openai_embed_deployment = None
        az_chat_only = AzureProvider(deployment="gpt", embed_deployment=None)
        assert az_chat_only.has_embedding() is False
    finally:
        settings.azure_openai_embed_deployment = configured
        get_embedding_provider.cache_clear()

    az_both = AzureProvider(deployment="gpt", embed_deployment="my-embed")
    assert az_both.has_embedding() is True
    assert az_both.info.embed_model == "my-embed"  # deployment name recorded

    # Registry-level gate — Azure key + endpoint alone are NOT enough; the
    # embed deployment must ALSO be set for the embedding provider to route
    # to Azure. Without it, deterministic keeps the retrieval path alive.
    get_embedding_provider.cache_clear()
    original = settings.azure_openai_embed_deployment
    try:
        settings.azure_openai_embed_deployment = None
        assert isinstance(get_embedding_provider(), DeterministicProvider)
    finally:
        settings.azure_openai_embed_deployment = original
        get_embedding_provider.cache_clear()


# ---- time-window parser: scopes retrieval to "this week" etc. ----


def test_time_window_returns_none_when_no_time_phrase():
    assert parse_time_window("what should I do first?") is None
    assert parse_time_window("who am I keeping waiting") is None


def test_time_window_this_week_covers_monday_through_next_monday():
    from datetime import datetime, timezone

    # Wednesday, 2026-03-04 UTC
    now = datetime(2026, 3, 4, 15, 30, tzinfo=timezone.utc)
    since, until = parse_time_window("how many meetings this week", now=now)
    assert since == datetime(2026, 3, 2, tzinfo=timezone.utc)   # Mon 00:00
    assert until == datetime(2026, 3, 9, tzinfo=timezone.utc)   # next Mon 00:00 (exclusive)


def test_time_window_today_is_a_single_utc_day():
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
    since, until = parse_time_window("what happened today", now=now)
    assert since == datetime(2026, 7, 28, tzinfo=timezone.utc)
    assert until - since == timedelta(days=1)


def test_time_window_last_n_days_is_inclusive_of_today():
    from datetime import datetime, timezone

    now = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
    since, until = parse_time_window("summarise the past 7 days", now=now)
    assert since == datetime(2026, 7, 21, tzinfo=timezone.utc)
    assert until == datetime(2026, 7, 29, tzinfo=timezone.utc)  # end of today, exclusive


def test_time_window_last_month_shifts_to_prior_month():
    from datetime import datetime, timezone

    now = datetime(2026, 3, 4, 15, 30, tzinfo=timezone.utc)
    since, until = parse_time_window("what closed last month", now=now)
    assert since == datetime(2026, 2, 1, tzinfo=timezone.utc)
    assert until == datetime(2026, 3, 1, tzinfo=timezone.utc)


def test_time_window_tomorrow_shorthand_matches():
    # Users type "tomm" and "tmrw" — without these, questions fall through to
    # unfiltered semantic search and past events get returned as upcoming.
    from datetime import datetime, timezone

    now = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
    expected_since = datetime(2026, 7, 29, tzinfo=timezone.utc)
    for phrasing in ("any meeting tomorrow?", "meeting tomm?", "tmrw sync?", "tmr call?"):
        since, until = parse_time_window(phrasing, now=now)
        assert since == expected_since, phrasing


def test_time_window_open_ended_coming_days_uses_forward_window():
    # "In coming days" with no number previously fell through and returned past events.
    from datetime import datetime, timezone

    now = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
    since, until = parse_time_window("any meeting in coming days?", now=now)
    assert since == datetime(2026, 7, 28, tzinfo=timezone.utc)  # today, not before
    assert until > since


def test_embedding_backfill_task_is_celery_registered():
    """Existing orgs (pre-subscriber) need a one-shot backfill; the sweeper also
    catches entity types the pointer bus doesn't route (COMPANY / MEETING / …).
    A missing task registration means the Celery worker can't run either."""
    import navis.registry  # noqa: F401
    from navis.platform.queue import celery_app

    for name in (
        "navis.ml.embeddings.jobs.embed_org",
        "navis.ml.embeddings.jobs.embed_all_orgs",
    ):
        assert name in celery_app.tasks, f"Celery task {name} not registered"
