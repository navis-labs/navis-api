"""An empty index answers without an embedding — and nothing else moves.

`MemoryRetriever.search` embeds the query and ranks the organization's vectors.
With no vector for the read pin's (model, version) the ranking is over nothing,
so the answer is `[]` whatever the query vector would have been — and the
remote embedding call (0.4s warm, 1.6s cold, measured) bought nothing. The
retriever now asks the index first.

What these tests hold in place, because each was a way this could go wrong:

  * an empty index returns exactly `[]`, and the provider is never called;
  * a populated index embeds and searches EXACTLY as before — same query, same
    read-pin model and version, same candidate width, same window;
  * failure is still failure: an embedding that fails for an organization with
    an index raises, and so does the index check itself — neither is ever
    turned into an empty answer;
  * `ContextBuilder` still treats a retrieval failure as fatal on BOTH of its
    paths (sequential for /chat, concurrent for the Developer boundary);
  * the one deliberate difference: an organization with nothing to search no
    longer depends on the embedding provider being up, and gets the same
    answer it gets when the provider is healthy.

The database half — that `index_has_rows` is False exactly when `search` has
nothing to return — is proved against Postgres in
`tests/integration/test_retriever_empty_index_postgres.py`.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import pytest

import navis.ml.retrieval.retriever as R
from navis.ml.embeddings.pins import EmbeddingPin
from navis.ml.providers.base import EmbedResult
from navis.ml.rag.grounding import Grounding
from navis.ml.retrieval.retriever import CANDIDATE_MULTIPLIER, MemoryRetriever

MODEL = "pinned-model"
VERSION = 3


class _Provider:
    """Counts calls; optionally behaves like an Azure outage."""

    def __init__(self, *, fail: bool = False):
        self.calls: list[list[str]] = []
        self.fail = fail

    async def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise ConnectionError("embedding provider unavailable")
        return EmbedResult(vectors=[[1.0, 0.0, 0.0]], model=MODEL, dim=3)


@pytest.fixture
def io(monkeypatch):
    """Fake every I/O call the retriever makes and record what it was asked."""
    seen: dict = {"index": [], "search": [], "lineage": 0, "has_rows": True, "index_error": None}

    async def index_has_rows(db, org_id, *, model, version):
        seen["index"].append((org_id, model, version))
        if seen["index_error"]:
            raise seen["index_error"]
        return seen["has_rows"]

    async def search(db, org_id, qvec, *, model, version, k, since, until):
        seen["search"].append({"qvec": qvec, "model": model, "version": version,
                               "k": k, "since": since, "until": until})
        return []

    async def lineage(db, org_id, ids):
        seen["lineage"] += 1
        return {}

    monkeypatch.setattr(R, "index_has_rows", index_has_rows)
    monkeypatch.setattr(R, "search", search)
    monkeypatch.setattr(R, "lineage_by_memory_id", lineage)
    return seen


def _retriever(provider) -> MemoryRetriever:
    return MemoryRetriever(object(), pin=EmbeddingPin(provider, MODEL, VERSION))


# ── the retriever ───────────────────────────────────────────────────────────

async def test_an_empty_index_answers_empty_without_embedding(io):
    io["has_rows"] = False
    provider = _Provider()

    hits = await _retriever(provider).search(uuid.uuid4(), "what is going on", k=8)

    assert hits == []
    assert provider.calls == [], "the query was embedded for an index with nothing in it"
    assert io["search"] == [] and io["lineage"] == 0


async def test_a_populated_index_embeds_and_searches_exactly_as_before(io):
    provider = _Provider()
    org = uuid.uuid4()
    since, until = object(), object()

    await _retriever(provider).search(org, "who owes us a reply", k=8, since=since, until=until)

    assert provider.calls == [["who owes us a reply"]]
    (call,) = io["search"]
    assert call == {"qvec": [1.0, 0.0, 0.0], "model": MODEL, "version": VERSION,
                    "k": 8 * CANDIDATE_MULTIPLIER, "since": since, "until": until}


async def test_the_check_reads_the_same_read_pin_index_the_search_reads(io):
    org = uuid.uuid4()

    await _retriever(_Provider()).search(org, "q")
    await _retriever(_Provider()).search(org, "q", version=VERSION + 1)

    assert io["index"] == [(org, MODEL, VERSION), (org, MODEL, VERSION + 1)]
    assert [c["version"] for c in io["search"]] == [VERSION, VERSION + 1]


async def test_an_embedding_failure_is_still_fatal_for_an_org_with_an_index(io):
    with pytest.raises(ConnectionError):
        await _retriever(_Provider(fail=True)).search(uuid.uuid4(), "q")
    assert io["search"] == []


async def test_a_failing_index_check_is_fatal_never_an_empty_answer(io):
    io["index_error"] = RuntimeError("database unavailable")
    provider = _Provider()

    with pytest.raises(RuntimeError, match="database unavailable"):
        await _retriever(provider).search(uuid.uuid4(), "q")
    assert provider.calls == []


async def test_an_empty_index_gets_the_same_answer_whether_or_not_the_provider_is_up(io):
    """THE ONE DELIBERATE BEHAVIOUR CHANGE, stated as a test.

    Before, an organization with nothing to search still failed when the
    embedding provider did, because the embedding came first. Its correct answer
    is `[]` either way; now that is the answer it gets either way."""
    io["has_rows"] = False

    healthy = await _retriever(_Provider()).search(uuid.uuid4(), "q")
    down = await _retriever(_Provider(fail=True)).search(uuid.uuid4(), "q")

    assert healthy == down == []


# ── the builder: retrieval failure stays fatal on both paths ────────────────

@asynccontextmanager
async def _session():
    yield object()


@pytest.fixture
def builder_io(monkeypatch, io):
    """A real `MemoryRetriever` inside a real `ContextBuilder`, provider pinned."""
    from navis.ml.rag import context_builder as CB

    async def none(db, org_id):
        return []

    monkeypatch.setattr(CB, "learning_groundings", none)
    monkeypatch.setattr(CB, "intelligence_groundings", none)

    def pin_with(provider):
        monkeypatch.setattr(R, "read_pin", lambda: EmbeddingPin(provider, MODEL, VERSION))

    return pin_with


@pytest.mark.parametrize("concurrent", [False, True], ids=["sequential(/chat)", "concurrent(developer)"])
async def test_the_builder_still_raises_a_retrieval_failure(builder_io, io, concurrent):
    from navis.ml.rag.context_builder import ContextBuilder

    builder_io(_Provider(fail=True))

    with pytest.raises(ConnectionError):
        await ContextBuilder(object()).build(
            uuid.uuid4(), "q", sessions=_session if concurrent else None,
        )


@pytest.mark.parametrize("concurrent", [False, True], ids=["sequential(/chat)", "concurrent(developer)"])
async def test_an_empty_index_builds_the_same_context_with_the_provider_up_or_down(
    builder_io, io, concurrent
):
    from navis.ml.rag.context_builder import ContextBuilder

    io["has_rows"] = False
    extra = [Grounding(kind="state", source="state:1", text="Org-wide state fact.")]
    built = {}
    for label, provider in (("up", _Provider()), ("down", _Provider(fail=True))):
        builder_io(provider)
        ctx = await ContextBuilder(object()).build(
            uuid.UUID(int=7), "q", extra=list(extra), sessions=_session if concurrent else None,
        )
        built[label] = [(f.kind, f.source, f.text) for f in ctx.facts]
        assert provider.calls == []

    assert built["up"] == built["down"]
    assert all(kind != "memory" for kind, _, _ in built["up"])
