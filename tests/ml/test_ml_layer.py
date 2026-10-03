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
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"), filename=str(py))):
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


async def test_the_azure_client_is_bounded_and_reused_within_a_loop(monkeypatch):
    """Two production faults, pinned together because they share one line.

    THE TIMEOUT. The client used to be built with no `timeout=`, so the SDK
    default applied — `read=600.0`. A hung Azure response could hold a context
    build for ten minutes. An embedding of one short string returns in well
    under two seconds, so anything near the default is a stall, not a slow
    answer.

    THE REUSE. It also built a NEW client per call, so every embedding paid a
    fresh TLS handshake and no connection was ever reused; nothing closed them
    either, so each leaked its httpx pool. Measured, the fix took the median
    embedding from ~1.55s to ~0.40s.

    Asserted on the CLIENT rather than by timing, so it holds on a machine
    with a different link to Azure and needs no network at all.
    """
    from navis.ml.providers.azure import AzureProvider
    from navis.platform.config import settings

    # `_client()` builds the SDK client for real, and `AsyncAzureOpenAI` refuses
    # to construct without a credential. Nothing below is about authentication
    # and no request is ever made, so a placeholder is enough — but it has to be
    # supplied, because read from the environment this passed on a laptop with a
    # populated `.env` and failed in CI, which has none. Same environment
    # coupling `test_azure_provider_gates_embedding_on_separate_deployment`
    # guards against below. `or` keeps a real configured key in play where one
    # exists, so a developer with Azure set up still exercises their own value.
    monkeypatch.setattr(
        settings,
        "azure_openai_api_key",
        settings.azure_openai_api_key or "not-a-real-key-no-request-is-made",
    )

    az = AzureProvider(deployment="gpt", embed_deployment="my-embed")
    client = az._client()

    assert client.timeout.read == settings.azure_openai_timeout_seconds
    assert client.timeout.read < 600, "the 600s SDK default is back"
    assert client.timeout.connect == settings.azure_openai_connect_timeout_seconds
    # Bounded, not disabled: one retry through the SDK's backoff+jitter is the
    # right answer to a single 429, and the ceiling stays (1 + retries) x timeout.
    assert client.max_retries == settings.azure_openai_max_retries

    # Same loop, same client — including across provider instances, because the
    # cache is keyed by the loop and not by `self`.
    assert az._client() is client
    assert AzureProvider(deployment="gpt", embed_deployment="my-embed")._client() is client


async def test_the_azure_client_keeps_idle_connections_for_the_configured_window(monkeypatch):
    """KEEP-ALIVE is not a timeout, and httpx's 5s default cost every call.

    Agents call context tens of seconds apart. With pooled connections expiring
    after 5s idle, nearly every embedding opened a new TCP + TLS connection to
    the East US endpoint: measured 1.6-2.1s per call against ~0.4-0.6s on a
    reused one.

    The window is BOUNDED, not "as long as possible": a connection a middlebox
    dropped silently fails on the write, so it must be discarded before the path
    forgets it. Measured, Azure closed an idle connection somewhere between 120s
    and 240s; AWS NAT gateways drop idle flows at 350s. The default stays under
    both, and passing our own client must not quietly change anything else —
    the SDK's pool sizes and our bounded timeout are asserted beside it.
    """
    from navis.ml.providers.azure import AzureProvider
    from navis.platform.config import settings

    monkeypatch.setattr(
        settings,
        "azure_openai_api_key",
        settings.azure_openai_api_key or "not-a-real-key-no-request-is-made",
    )
    monkeypatch.setattr(settings, "azure_openai_keepalive_expiry_seconds", 42.0)

    client = AzureProvider(deployment="gpt", embed_deployment="my-embed")._client()
    pool = client._client._transport._pool

    assert pool._keepalive_expiry == 42.0
    assert pool._max_connections == 1000          # the SDK's own default
    assert pool._max_keepalive_connections == 100  # the SDK's own default
    assert client.timeout.read == settings.azure_openai_timeout_seconds
    assert client.timeout.connect == settings.azure_openai_connect_timeout_seconds


def test_the_default_keepalive_window_stays_under_what_the_path_holds():
    from navis.platform.config.settings import Settings

    default = Settings(_env_file=None).azure_openai_keepalive_expiry_seconds
    assert 5.0 < default < 120.0, "above 5s to reuse at all; under 120s, the longest idle Azure was seen to hold"


def test_the_azure_client_cache_does_not_leak_across_event_loops(monkeypatch):
    """httpx pools are not shareable between asyncio loops, so a plain module
    global would be reused from the wrong loop and fail at the transport. Each
    loop must get its own."""
    import asyncio

    from navis.ml.providers.azure import AzureProvider
    from navis.platform.config import settings

    # Same reason as the test above: the SDK will not construct without a
    # credential, and this test is about loop affinity, not authentication.
    monkeypatch.setattr(
        settings,
        "azure_openai_api_key",
        settings.azure_openai_api_key or "not-a-real-key-no-request-is-made",
    )

    async def _grab():
        return AzureProvider(deployment="gpt", embed_deployment="my-embed")._client()

    first = asyncio.run(_grab())
    second = asyncio.run(_grab())
    assert first is not second, "a client crossed event loops"


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


# ---- context composition (Phase 3E.1 / 3E.2) -------------------------------
#
# 3E.1: `build()` appended caller facts AFTER retrieval hits, so the org's own
# conclusions ranked below raw recall and the model answered "cannot be
# determined" to questions the reasoning signals already answered.
#
# 3E.2: the fix over-corrected. Ordering purely by kind put 8 conclusions + 22
# State entities ahead of ALL retrieved evidence, and six questions that need
# retrieval detail regressed to unanswerable — not from evidence loss (8/8 were
# retained, ~900 of 1500 tokens) but because the model anchored on 30 derived
# facts before reaching any. Both properties are pinned here.


def _g(kind, source, text=""):
    from navis.ml.rag.grounding import Grounding
    return Grounding(kind=kind, source=source, text=text or source)


def test_conclusions_lead_and_evidence_is_not_buried_under_state():
    from navis.ml.rag.context_builder import _LEADING_STATE_FACTS, _compose

    facts = (
        [_g("memory", f"m{i}") for i in range(8)]
        + [_g("reasoning", f"r{i}") for i in range(3)]
        + [_g("state", f"s{i}") for i in range(22)]
    )
    out = [f.kind for f in _compose(facts, "which documents are being worked on?")]

    assert out[:3] == ["reasoning"] * 3, "conclusions must lead"
    # Evidence must appear before the State tail, and no later than
    # conclusions + the bounded State slice.
    first_memory = out.index("memory")
    assert first_memory <= 3 + _LEADING_STATE_FACTS
    # ALL retrieval evidence is kept — that is the property that regressed.
    assert out.count("memory") == 8, "retrieval evidence must never be dropped"
    # The LEADING band is what is bounded: thin pointers must not out-volume the
    # detailed evidence the model reads first.
    assert out[: 3 + _LEADING_STATE_FACTS].count("state") == _LEADING_STATE_FACTS
    # 3E.2.1: the rest of State is APPENDED, not discarded. Ordering decides what
    # is read first and — only if the budget binds — who is dropped first; it
    # does not decide membership. Truncating here threw away 16 of 22 entities on
    # every turn while a third of the token budget went unspent.
    assert out.count("state") == 22, "State must not be silently truncated"
    # ...and every retrieved fact still precedes the tail, so the tail can only
    # spend budget the evidence has already left over.
    last_memory = len(out) - 1 - out[::-1].index("memory")
    assert out.index("state", 3 + _LEADING_STATE_FACTS) > last_memory


def test_state_slice_is_question_ranked_and_deterministic():
    from navis.ml.rag.context_builder import _compose

    facts = [
        _g("state", "s-irrelevant", "relationship with a vendor"),
        _g("state", "s-match", "documents Product Spec owned by alice"),
        _g("memory", "m1", "an email"),
    ]
    # NOTE: `_overlap` is exact-token (no stemming) — "documents" matches
    # "documents", not "document". That is the reranker's existing definition,
    # reused deliberately so retrieval and State share one notion of relevance.
    out = _compose(facts, "which documents are being worked on?")
    assert out[0].source == "s-match", "the question-relevant State fact leads"
    # Deterministic: same inputs, same order, no clock and no model.
    assert [f.source for f in _compose(facts, "which documents are being worked on?")] == [
        f.source for f in out
    ]


def test_unknown_kinds_are_treated_as_evidence_not_conclusions():
    """A connector-named fact from the workspace fallback must never outrank the
    organization's own conclusions."""
    from navis.ml.rag.context_builder import _compose

    out = _compose([_g("gmail", "g1"), _g("reasoning", "r1")], "what changed?")
    assert [f.kind for f in out] == ["reasoning", "gmail"]


# ══ LEARNING → ML: the organization's own outcomes as grounded facts ══════════
#
# `learning_facts` is a pure function of a LearningSnapshot, so everything about
# WHAT the model is told can be pinned without a database. The Observation →
# Learning → ML chain on real Postgres is proved in
# tests/integration/test_observation_contract.py.


def _snapshot(**over):
    """A LearningSnapshot built straight from primitives — no ORM, no session."""
    from datetime import datetime, timezone

    from navis.learning.metrics import AggregateFact, LearningSnapshot

    fields = {
        "org_id": ORG,
        "learning_version": 4,
        "as_of": datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
        "state_watermark": "wm-9",
        "reasoning_version": 7,
        "aggregates": (
            AggregateFact("outcome", "action_executed", 0, 0, 9, 9),
            AggregateFact("outcome", "action_failed", 0, 3, 0, 3),
            AggregateFact("outcome", "action_rejected", 0, 2, 0, 2),
            AggregateFact("outcome", "reply_received", 5, 0, 0, 5),
        ),
        "patterns": (),
        "lineage": {},
        "signal_lineage": {},
    }
    fields.update(over)
    return LearningSnapshot(**fields)


def _texts(facts):
    return " ".join(f.text for f in facts)


def test_learning_facts_are_pure_and_reproducible():
    """Same org + same learning_version => byte-identical facts, in order.

    Reproducibility is the whole claim: a learned fact the model cites must be
    re-derivable, not something that happened to be true during one request.
    """
    from navis.ml.rag.context_builder import learning_facts

    snap = _snapshot()
    a = learning_facts(snap)
    b = learning_facts(snap)
    assert [(f.source, f.text, f.lineage) for f in a] == [
        (f.source, f.text, f.lineage) for f in b
    ]
    assert a, "a snapshot with outcomes produced no facts at all"


def test_delivery_judgement_and_world_stay_three_separate_facts():
    """A channel fault must never reach the model as a reasoning failure.

    The three surfaces are rendered as three separately-labelled facts; folding
    them into one "3 failures" line would recreate, inside the prompt, exactly
    the conflation Learning's two-axis classification removed.
    """
    from navis.ml.rag.context_builder import learning_facts

    facts = learning_facts(_snapshot())
    by_source = {f.source: f for f in facts}
    assert "outcome_surface:delivery" in by_source
    assert "outcome_surface:judgement" in by_source
    assert "outcome_surface:world" in by_source

    delivery = by_source["outcome_surface:delivery"].text
    assert "action_failed" in delivery
    assert "NOT a judgement" in delivery, "a delivery fault was not labelled as one"
    # The delivery failure must not appear inside the judgement or world facts.
    assert "action_failed" not in by_source["outcome_surface:judgement"].text
    assert "action_failed" not in by_source["outcome_surface:world"].text
    # And the human's rejection is named as a verdict, not as an error.
    assert "action_rejected" in by_source["outcome_surface:judgement"].text
    assert "reply_received" in by_source["outcome_surface:world"].text


def test_an_organization_with_no_outcomes_gets_no_fabricated_feature():
    """Absence of evidence must not become evidence.

    No zeroes, no "insufficient data" placeholder — nothing, so the model has
    nothing to cite and cannot report a learned fact that was never learned.
    """
    from navis.ml.rag.context_builder import learning_facts

    assert learning_facts(_snapshot(learning_version=0, aggregates=())) == []


def test_a_surface_with_no_outcomes_is_omitted_not_zeroed():
    from navis.ml.rag.context_builder import learning_facts
    from navis.learning.metrics import AggregateFact

    facts = learning_facts(
        _snapshot(aggregates=(AggregateFact("outcome", "action_executed", 0, 0, 2, 2),))
    )
    sources = {f.source for f in facts}
    assert sources == {"outcome_surface:delivery"}
    assert "outcome_surface:judgement" not in sources
    assert "outcome_surface:world" not in sources


def test_signal_reliability_is_rendered_not_recomputed():
    """Learning already decided what "reliable" means and gated it on samples.

    ML renders that verdict; if it re-thresholded here there would be two
    definitions of signal reliability in the codebase, free to drift.
    """
    from navis.learning.metrics import SignalPattern
    from navis.ml.rag.context_builder import learning_facts

    facts = learning_facts(
        _snapshot(
            patterns=(SignalPattern("sig-A", "reliable", 0.87, 120),),
            lineage={"sig-A": ["le-1", "le-2"]},
        )
    )
    pattern_fact = next(f for f in facts if f.source == "signal_pattern:sig-A")
    assert "87%" in pattern_fact.text and "120" in pattern_fact.text
    assert "reliable" in pattern_fact.text
    # Lineage travels with the fact: which signal, which evidence, which version.
    assert "signal:sig-A" in pattern_fact.lineage
    assert "learning_event:le-1" in pattern_fact.lineage
    assert "learning_version:4" in pattern_fact.lineage


def test_signal_facts_are_capped_deterministically_by_sample_weight():
    """Most-evidenced first, signal_id to break ties — never arbitrary order."""
    from navis.learning.metrics import SignalPattern
    from navis.ml.rag.context_builder import _MAX_SIGNAL_FACTS, learning_facts

    patterns = tuple(
        SignalPattern(f"sig-{i:02d}", "reliable", 0.9, samples)
        for i, samples in enumerate([10, 90, 30, 70, 50, 20, 80])
    )
    facts = [f for f in learning_facts(_snapshot(patterns=patterns))
             if f.source.startswith("signal_pattern:")]
    assert len(facts) == _MAX_SIGNAL_FACTS
    assert [f.source for f in facts] == [
        "signal_pattern:sig-01",  # 90
        "signal_pattern:sig-06",  # 80
        "signal_pattern:sig-03",  # 70
        "signal_pattern:sig-04",  # 50
        "signal_pattern:sig-02",  # 30
    ]


def test_learning_facts_land_in_the_state_band_not_the_conclusion_band():
    """`_STATE_KINDS` already reserved "learning"; these facts must fill it.

    A learned fact describes the organization, so it ranks with State — it must
    not outrank the reasoning conclusions a founder actually asked about.
    """
    from navis.ml.rag.context_builder import LEARNING_KIND, _compose

    out = _compose([_g(LEARNING_KIND, "l1"), _g("reasoning", "r1"), _g("memory", "m1")], "q")
    assert [f.kind for f in out] == ["reasoning", LEARNING_KIND, "memory"]


def test_ml_reads_learning_through_the_contract_not_its_tables():
    """The whole point of the seam. ML may import navis.learning (its own AST
    invariant says so), but only the read contract — never the ORM."""
    offenders = []
    for py in (NAVIS / "ml").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for orm in ("LearningEvent", "LearningAggregate", "LearningPattern", "learning_events"):
            if orm in text:
                offenders.append(f"ml/{py.name} references {orm}")
    assert not offenders, "ML reached into Learning's storage:\n" + "\n".join(offenders)

    builder = (NAVIS / "ml" / "rag" / "context_builder.py").read_text(encoding="utf-8")
    assert "from navis.learning.metrics import" in builder, (
        "ML stopped consuming Learning through its contract"
    )


# ── ContextBuilder reads Intelligence (TASK 7.5) ───────────────────────────
#
# `_CONCLUSION_KINDS` has reserved a band for "intelligence" since it was
# written, but nothing ever produced a fact for it: Ask was asking the LLM to
# re-derive, from raw memory, conclusions the organization had already reached
# and persisted in `intelligence_insights`. These pin the wiring — without them a
# refactor can drop the call and every other test still passes.


async def test_context_builder_reads_intelligence_and_leads_with_it(monkeypatch):
    """Persisted insights reach the model, AHEAD of the evidence behind them."""
    import uuid as _uuid

    from navis.ml.rag import context_builder as CB
    from navis.ml.rag.grounding import Grounding

    org = _uuid.UUID("11111111-1111-1111-1111-111111111111")
    seen: list = []

    class _Retriever:
        def __init__(self, db):
            pass

        async def search(self, org_id, query, **kw):
            return []

    async def _no_learning(db, org_id):
        return []

    async def _insights(db, org_id):
        seen.append(org_id)
        return [
            Grounding(
                kind=CB.INTELLIGENCE_KIND,
                source="insight:i-1",
                text="Investor threads stall after Thursday.",
                lineage=("reasoning_signal:rs-1",),
            )
        ]

    monkeypatch.setattr(CB, "MemoryRetriever", _Retriever)
    monkeypatch.setattr(CB, "learning_groundings", _no_learning)
    monkeypatch.setattr(CB, "intelligence_groundings", _insights)

    ctx = await CB.ContextBuilder(object()).build(
        org,
        "why do investor threads stall?",
        extra=[Grounding(kind="memory", source="mem-1", text="A thread.")],
    )

    # Asked for the caller's org, and nothing else.
    assert seen == [org]
    kinds = [f.kind for f in ctx.facts]
    assert kinds[0] == "intelligence", "a conclusion must precede its evidence"
    assert "memory" in kinds
    # Lineage survives assembly.
    insight = next(f for f in ctx.facts if f.kind == "intelligence")
    assert insight.lineage == ("reasoning_signal:rs-1",)


async def test_intelligence_outage_never_fails_the_turn(monkeypatch):
    """Intelligence is enrichment. Losing it costs the answer its conclusions,
    never the answer — the same contract Learning already keeps."""
    import uuid as _uuid

    from navis.ml.rag import context_builder as CB
    from navis.ml.rag.grounding import Grounding

    class _Retriever:
        def __init__(self, db):
            pass

        async def search(self, org_id, query, **kw):
            return []

    async def _no_learning(db, org_id):
        return []

    async def _broken(db, org_id):
        raise RuntimeError("intelligence repository down")

    monkeypatch.setattr(CB, "MemoryRetriever", _Retriever)
    monkeypatch.setattr(CB, "learning_groundings", _no_learning)
    monkeypatch.setattr(CB, "intelligence_groundings", _broken)

    ctx = await CB.ContextBuilder(object()).build(
        _uuid.uuid4(),
        "anything",
        extra=[Grounding(kind="memory", source="mem-1", text="A thread.")],
    )

    assert [f.kind for f in ctx.facts] == ["memory"]


async def test_retrieval_query_is_what_gets_embedded_not_the_question(monkeypatch):
    """The model is shown the founder's words; retrieval searches for the
    resolved person. Memory holds a person's ADDRESS far more often than their
    display name, so the two must be allowed to differ."""
    import uuid as _uuid

    from navis.ml.rag import context_builder as CB

    embedded: list = []

    class _Retriever:
        def __init__(self, db):
            pass

        async def search(self, org_id, query, **kw):
            embedded.append(query)
            return []

    async def _none(db, org_id):
        return []

    monkeypatch.setattr(CB, "MemoryRetriever", _Retriever)
    monkeypatch.setattr(CB, "learning_groundings", _none)
    monkeypatch.setattr(CB, "intelligence_groundings", _none)

    ctx = await CB.ContextBuilder(object()).build(
        _uuid.uuid4(),
        "Why does Abhishek matter?",
        retrieval_query="Why does Abhishek matter? abhishek@acme.com",
    )

    assert embedded == ["Why does Abhishek matter? abhishek@acme.com"]
    assert ctx.question == "Why does Abhishek matter?"


# ── Subject-aware composition + the one expansion hop (TASK 7.5.1) ─────────
#
# MEASURED ON THE LIVE ORG: all eight entity-centric probe questions produced a
# byte-identical 53-fact bundle at ~1450/1500 tokens whose first SEVEN facts were
# org-wide insights naming nobody. Only 9-11 of 53 facts mentioned the person
# asked about. The bundle was, in effect, question-independent.


def test_subject_facts_lead_and_off_subject_conclusions_are_capped():
    """On an entity question, facts naming the subject precede org-wide ones and
    the generic conclusions cannot own the opening."""
    from navis.ml.rag.context_builder import _compose
    from navis.ml.rag.grounding import Grounding

    def g(kind, text):
        return Grounding(kind=kind, source=f"{kind}:{text[:6]}", text=text)

    cands = [
        g("intelligence", "3 hub companies connect multiple people"),
        g("intelligence", "Tracking 113 organizational entities"),
        g("intelligence", "1 warm relationship this window"),
        g("intelligence", "2 open commitments at risk"),
        g("intelligence", "Priya is the counterparty on the stalled deal"),
        g("memory", "Person Priya, sender"),
        g("memory", "Unrelated invoice thread"),
        g("state", "relationship: Priya — open"),
        g("state", "company: Acme — active"),
    ]
    out = _compose(cands, "why does priya matter?", ("priya",))
    texts = [f.text for f in out]

    # The conclusion ABOUT the subject leads.
    assert texts[0] == "Priya is the counterparty on the stalled deal"
    # At most three off-subject conclusions may precede the subject's evidence.
    first_mem = next(i for i, t in enumerate(texts) if t == "Person Priya, sender")
    off_before = sum(
        1 for t in texts[:first_mem]
        if "priya" not in t.lower()
    )
    assert off_before <= 3, f"generic facts owned the opening: {texts[:first_mem]}"
    # Nothing is dropped — this decides order only.
    assert len(out) == len(cands)


def test_composition_is_unchanged_without_a_subject():
    """An org-wide question keeps the previous ordering exactly: every conclusion
    IS the subject there, so the cap must not bind."""
    from navis.ml.rag.context_builder import _compose
    from navis.ml.rag.grounding import Grounding

    cands = [
        Grounding(kind="intelligence", source=f"i{i}", text=f"insight {i}")
        for i in range(6)
    ] + [Grounding(kind="memory", source="m", text="evidence")]

    out = _compose(cands, "how are we doing?")
    assert [f.kind for f in out[:6]] == ["intelligence"] * 6


async def test_expand_runs_once_and_never_fails_the_turn(monkeypatch):
    """The second hop is called exactly once with the retrieved facts, and an
    expansion that raises costs depth, not the answer."""
    import uuid as _uuid

    from navis.ml.rag import context_builder as CB
    from navis.ml.rag.grounding import Grounding

    class _Retriever:
        def __init__(self, db):
            pass

        async def search(self, org_id, query, **kw):
            from navis.ml.retrieval.vector_search import Hit
            return [Hit(_uuid.uuid4(), 0.9, "person", "A retrieved fact", None)]

    async def _none(db, org_id):
        return []

    monkeypatch.setattr(CB, "MemoryRetriever", _Retriever)
    monkeypatch.setattr(CB, "learning_groundings", _none)
    monkeypatch.setattr(CB, "intelligence_groundings", _none)

    calls: list = []

    async def _expand(retrieved):
        calls.append([f.text for f in retrieved])
        return [Grounding(kind="memory", source="expanded-1", text="An expanded fact")]

    ctx = await CB.ContextBuilder(object()).build(
        _uuid.uuid4(), "q", expand=_expand
    )
    assert len(calls) == 1, "the hop must run exactly once, never in a loop"
    assert calls[0] == ["A retrieved fact"], "the callback sees what retrieval found"
    assert "An expanded fact" in [f.text for f in ctx.facts]

    async def _boom(retrieved):
        raise RuntimeError("expansion backend down")

    ctx2 = await CB.ContextBuilder(object()).build(
        _uuid.uuid4(), "q", expand=_boom
    )
    assert [f.text for f in ctx2.facts] == ["A retrieved fact"]
