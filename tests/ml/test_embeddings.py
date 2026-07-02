"""Regression tests for navis/ml/embeddings — the production bugs fixed in the embeddings audit:

  * embed_org forward-progress: selects NOT-yet-embedded records via a NOT EXISTS anti-join,
    ordered + limited (no more "arbitrary 1000, never converges").
  * batching: one provider.embed() round-trip per batch, not per record.
  * fail-closed: a provider returning a mismatched vector count never writes misaligned vectors.
  * job lifecycle: the Celery job closes the per-loop provider client even on error (no leak).

No Postgres required: a fake session captures the SQL / inserts and a fake provider records the
batch sizes. compute_embedding_id / memory_text are pure and tested directly.
"""
from __future__ import annotations

import types
import uuid

import pytest
from sqlalchemy.dialects import postgresql

from navis.ml.embeddings import service as svc_mod
from navis.ml.embeddings.models import compute_embedding_id
from navis.ml.embeddings.service import EmbeddingService, memory_text

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")


def _rec(i: int):
    return types.SimpleNamespace(
        memory_id=uuid.UUID(int=i),
        organization_id=ORG,
        entity_type="person",
        payload={"title": f"fact {i}"},
        dedup_key=f"d{i}",
    )


class _Provider:
    info = types.SimpleNamespace(embed_model="navis-hash-256")

    def __init__(self, *, short_by: int = 0):
        self.batches: list[int] = []
        self._short_by = short_by

    async def embed(self, texts):
        self.batches.append(len(texts))
        n = len(texts) - self._short_by
        return types.SimpleNamespace(vectors=[[0.0] * 8 for _ in range(n)], dim=8, model="navis-hash-256")


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Session:
    def __init__(self, records, provider):
        self._records = records
        self._provider = provider
        self.selects: list = []
        self.inserts: list = []

    async def scalars(self, stmt):
        self.selects.append(stmt)
        return _Result(self._records)

    async def execute(self, stmt):
        self.inserts.append(stmt)
        return types.SimpleNamespace(rowcount=self._provider.batches[-1])  # rows in the last batch

    async def flush(self):
        return None


def _service(records, provider, monkeypatch) -> EmbeddingService:
    monkeypatch.setattr(svc_mod, "get_embedding_provider", lambda: provider)
    return EmbeddingService(_Session(records, provider))  # type: ignore[arg-type]


def _sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect())).upper()


# --- forward progress: the anti-join query shape -----------------------------

async def test_embed_org_selects_only_unembedded_ordered(monkeypatch):
    prov = _Provider()
    svc = _service([_rec(1)], prov, monkeypatch)
    await svc.embed_org(ORG)
    sql = _sql(svc.db.selects[0])  # type: ignore[attr-defined]
    assert "NOT (EXISTS" in sql                       # anti-join: skip already-embedded
    assert "MEMORY_EMBEDDINGS.MEMORY_ID = MEMORY_RECORDS.MEMORY_ID" in sql
    assert "ORDER BY MEMORY_RECORDS.MEMORY_ID" in sql  # stable forward progress
    assert "LIMIT" in sql and "ORGANIZATION_ID" in sql


# --- batching: one embed() per batch, not per record -------------------------

async def test_embed_org_batches_provider_calls(monkeypatch):
    prov = _Provider()
    svc = _service([_rec(i) for i in range(300)], prov, monkeypatch)
    written = await svc.embed_org(ORG)
    assert prov.batches == [128, 128, 44]                 # 300 rows -> 3 batches, not 300 calls
    assert len(svc.db.inserts) == 3                        # one bulk insert per batch  # type: ignore[attr-defined]
    assert written == 300


async def test_embed_org_empty_is_noop(monkeypatch):
    prov = _Provider()
    svc = _service([], prov, monkeypatch)
    assert await svc.embed_org(ORG) == 0
    assert prov.batches == [] and svc.db.inserts == []    # type: ignore[attr-defined]


# --- fail-closed: never store misaligned vectors -----------------------------

async def test_batch_vector_count_mismatch_raises(monkeypatch):
    prov = _Provider(short_by=1)  # returns one fewer vector than inputs
    svc = _service([_rec(i) for i in range(5)], prov, monkeypatch)
    with pytest.raises(ValueError, match="vectors"):
        await svc.embed_org(ORG)


# --- pure helpers ------------------------------------------------------------

def test_memory_text_is_stable_and_bounded():
    assert memory_text(_rec(7)) == "person — fact 7"
    long = types.SimpleNamespace(entity_type="x", payload={"text": "y" * 5000}, dedup_key="d")
    assert len(memory_text(long)) <= 2000  # matches source_text column bound


def test_compute_embedding_id_is_deterministic_and_versioned():
    mid = uuid.uuid4()
    assert compute_embedding_id(mid, "m", 1) == compute_embedding_id(mid, "m", 1)
    assert compute_embedding_id(mid, "m", 1) != compute_embedding_id(mid, "m", 2)
    assert compute_embedding_id(mid, "m", 1) != compute_embedding_id(mid, "n", 1)


# --- job lifecycle: close the pooled client even on error --------------------

class _Closable:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


class _Ctx:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *a):
        return False


async def test_job_closes_embedding_client_even_on_error(monkeypatch):
    import navis.ml.embeddings.jobs as jobs

    prov = _Closable()
    monkeypatch.setattr("navis.ml.providers.get_embedding_provider", lambda: prov)
    monkeypatch.setattr(jobs, "session_scope", lambda: _Ctx())

    class _Boom:
        def __init__(self, db):
            pass

        async def embed_org(self, org_id, *, version):
            raise RuntimeError("provider down")

    monkeypatch.setattr(jobs, "EmbeddingService", _Boom)
    with pytest.raises(RuntimeError):
        await jobs._embed_org(uuid.uuid4(), 1)
    assert prov.closed is True  # finally closed the per-loop client despite the error
