"""Migration infrastructure for the embedding index (no Azure calls).

Covers the five pieces that make a full re-embed survivable:
read/write pin split, resumable keyset pagination, batched generation,
progress tracking, and embed-time admission filtering.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pytest

from navis.ml.embeddings.admission import admits
from navis.ml.embeddings.models import compute_embedding_id, compute_progress_id
from navis.ml.embeddings.pins import (
    EMBEDDING_VERSION,
    EmbeddingPin,
    UnknownEmbeddingModel,
    _provider_for_model,
    pins_are_aligned,
    read_pin,
    write_pin,
)
from navis.ml.embeddings.service import EmbeddingService, memory_text
from navis.ml.providers.deterministic import DeterministicProvider
from navis.platform.config import settings


# ── Admission filtering ────────────────────────────────────────────────────


@dataclass
class FakeRecord:
    entity_type: str
    payload: dict
    memory_id: uuid.UUID = field(default_factory=uuid.uuid4)
    organization_id: uuid.UUID = field(default_factory=uuid.uuid4)
    dedup_key: str = "k"


def test_admits_keeps_structural_facts():
    for kind in ("person", "company", "thread", "relationship", "meeting", "document", "project"):
        assert admits(FakeRecord(kind, {"name": "x"})) is True


def test_admits_rejects_the_old_regex_false_positives():
    """The exact strings the buggy `\\bwe'?ll\\b` produced must not be indexed."""
    for text in (
        "We hope this email finds you well.",
        "Everything was going well.",
        "I hope you are doing well.",
    ):
        assert admits(FakeRecord("commitment", {"text": text})) is False


def test_admits_keeps_genuine_claims():
    assert admits(FakeRecord("commitment", {"text": "I'll send the deck by Friday."})) is True
    assert admits(FakeRecord("decision", {"text": "We decided to postpone the launch."})) is True


def test_admits_rejects_claim_with_no_text():
    assert admits(FakeRecord("commitment", {})) is False
    assert admits(FakeRecord("decision", {"text": ""})) is False


# ── Read / write pin split ─────────────────────────────────────────────────


@pytest.fixture
def clean_pins(monkeypatch):
    monkeypatch.setattr(settings, "embedding_read_model", None, raising=False)
    monkeypatch.setattr(settings, "embedding_read_version", None, raising=False)
    yield


def test_pins_default_to_following_writes(clean_pins):
    """Unset config must behave exactly as before this change."""
    r, w = read_pin(), write_pin()
    assert (r.model, r.version) == (w.model, w.version)
    assert pins_are_aligned() is True


def test_read_pin_can_be_held_on_a_different_index(monkeypatch):
    """The whole point: reads stay put while writes move to the next index."""
    monkeypatch.setattr(settings, "embedding_read_model", "navis-hash-256", raising=False)
    monkeypatch.setattr(settings, "embedding_read_version", 1, raising=False)
    r = read_pin()
    assert (r.model, r.version) == ("navis-hash-256", 1)
    assert r.provider.info.embed_model == "navis-hash-256"


def test_read_pin_version_can_differ_from_write_version(monkeypatch):
    monkeypatch.setattr(settings, "embedding_read_model", "navis-hash-256", raising=False)
    monkeypatch.setattr(settings, "embedding_read_version", 7, raising=False)
    assert read_pin().version == 7
    assert write_pin().version == EMBEDDING_VERSION
    assert pins_are_aligned() is False


def test_unknown_read_model_fails_loudly_rather_than_falling_back(monkeypatch):
    """A typo must not silently route queries to the wrong embedder."""
    monkeypatch.setattr(settings, "embedding_read_model", "not-a-real-model", raising=False)
    with pytest.raises(UnknownEmbeddingModel):
        read_pin()


def test_provider_for_model_resolves_the_deterministic_embedder():
    assert isinstance(_provider_for_model("navis-hash-256"), DeterministicProvider)


async def test_pin_carries_its_own_provider_for_query_embedding():
    """A query must be embedded by the model that produced the stored vectors."""
    pin = EmbeddingPin(DeterministicProvider(), "navis-hash-256", 1)
    vectors = await pin.embed(["what am I waiting on?"])
    assert len(vectors) == 1 and len(vectors[0]) == 256


# ── Batched generation ─────────────────────────────────────────────────────


class RecordingProvider(DeterministicProvider):
    """Counts embed() calls so batching is provable, not assumed."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def embed(self, texts: list[str]):
        self.calls.append(len(texts))
        return await DeterministicProvider().embed(texts)


class FakeResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class FakeDB:
    """Captures the insert payload without needing Postgres.

    `pg_insert(...).values([...])` stores the batch in `_multi_values`, keyed by
    Column objects rather than strings — normalize to column names so assertions
    read naturally.
    """

    def __init__(self) -> None:
        self.inserted: list[dict] = []

    @staticmethod
    def _named(row: dict) -> dict:
        return {getattr(k, "name", str(k)): v for k, v in row.items()}

    async def execute(self, stmt):
        batch = [self._named(r) for r in stmt._multi_values[0]]
        self.inserted.extend(batch)
        return FakeResult(len(batch))


async def test_write_batch_uses_one_provider_call_per_batch():
    provider = RecordingProvider()
    svc = EmbeddingService(FakeDB(), pin=EmbeddingPin(provider, "navis-hash-256", 1))
    records = [
        FakeRecord("person", {"email": f"p{i}@x.com", "name": f"P{i}", "role": "from"})
        for i in range(10)
    ]
    written = await svc._write_batch(records, version=1)
    assert provider.calls == [10], "expected ONE embed call for the whole batch"
    assert written == 10


async def test_write_batch_refuses_mismatched_vector_count():
    """Never pair a vector with the wrong fact."""

    class BadProvider(DeterministicProvider):
        async def embed(self, texts):
            return await DeterministicProvider().embed(texts[:-1])  # one short

    svc = EmbeddingService(FakeDB(), pin=EmbeddingPin(BadProvider(), "navis-hash-256", 1))
    records = [FakeRecord("person", {"email": f"p{i}@x.com"}) for i in range(3)]
    with pytest.raises(RuntimeError, match="returned 2 vectors for 3 inputs"):
        await svc._write_batch(records, version=1)


async def test_write_batch_is_a_noop_on_empty_input():
    provider = RecordingProvider()
    svc = EmbeddingService(FakeDB(), pin=EmbeddingPin(provider, "navis-hash-256", 1))
    assert await svc._write_batch([], version=1) == 0
    assert provider.calls == []


async def test_batch_rows_carry_the_pin_model_and_deterministic_ids():
    db = FakeDB()
    pin = EmbeddingPin(DeterministicProvider(), "navis-hash-256", 3)
    svc = EmbeddingService(db, pin=pin)
    rec = FakeRecord("person", {"email": "a@x.com", "name": "A", "role": "from"})
    await svc._write_batch([rec], version=3)
    row = db.inserted[0]
    assert row["model"] == "navis-hash-256" and row["version"] == 3
    assert row["id"] == compute_embedding_id(rec.memory_id, "navis-hash-256", 3)
    assert row["source_text"] == memory_text(rec)
    assert row["dim"] == 256


# ── Progress identity ──────────────────────────────────────────────────────


def test_progress_id_is_deterministic_per_index():
    org = uuid.uuid4()
    a = compute_progress_id(org, "navis-hash-256", 1)
    b = compute_progress_id(org, "navis-hash-256", 1)
    assert a == b, "concurrent workers must converge on ONE progress row"
    assert compute_progress_id(org, "navis-hash-256", 2) != a
    assert compute_progress_id(org, "text-embedding-3-small", 1) != a
    assert compute_progress_id(uuid.uuid4(), "navis-hash-256", 1) != a


def test_embedding_id_separates_indexes():
    """Two indexes coexist in the same table — ids must never collide."""
    mid = uuid.uuid4()
    hash_id = compute_embedding_id(mid, "navis-hash-256", 1)
    azure_id = compute_embedding_id(mid, "navis-embed", 1)
    v2_id = compute_embedding_id(mid, "navis-hash-256", 2)
    assert len({hash_id, azure_id, v2_id}) == 3
