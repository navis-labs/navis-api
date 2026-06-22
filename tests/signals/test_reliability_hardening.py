"""Reliability hardening: DLQ poison-loop guard + full-backlog deterministic replay."""
from __future__ import annotations

import uuid

from navis.signals.reliability import replay as RP
from navis.signals.reliability.replay import ReplayService
from tests.signals.fakes import FakeIngestion, make_ctx
from tests.signals.test_dead_letter_pg import FakeDLQRepo, _event


class _Row:
    def __init__(self, payload, attempts=0):
        self.id = uuid.uuid4()
        self.payload = payload
        self.attempts = attempts


# ---- P1: DLQ poison-loop guard ----

async def test_poison_is_permanently_dead_lettered_after_max_attempts():
    ctx = make_ctx()
    poison = _event(ctx, "poison")
    repo = FakeDLQRepo(unresolved=[_Row(poison.to_dict())])
    rs = ReplayService(db=None)  # type: ignore[arg-type]
    rs.dlq_repo = repo  # type: ignore[assignment]
    rs.ingestion = FakeIngestion(fail_on={"poison"})  # type: ignore[assignment]

    # max_attempts=1 -> the first failure exhausts it (no infinite retry loop).
    result = await rs.replay_dead_letters(ctx.org_id, max_attempts=1)
    assert result["exhausted"] == 1
    assert result["still_failing"] == 0
    assert result["recovered"] == 0


async def test_poison_retries_until_exhausted_then_stops():
    ctx = make_ctx()
    poison = _event(ctx, "poison")
    row = _Row(poison.to_dict())
    repo = FakeDLQRepo(unresolved=[row])
    rs = ReplayService(db=None)  # type: ignore[arg-type]
    rs.dlq_repo = repo  # type: ignore[assignment]
    rs.ingestion = FakeIngestion(fail_on={"poison"})  # type: ignore[assignment]

    # 2 passes under max=3 -> still failing; 3rd -> exhausted.
    assert (await rs.replay_dead_letters(ctx.org_id, max_attempts=3))["still_failing"] == 1
    assert (await rs.replay_dead_letters(ctx.org_id, max_attempts=3))["still_failing"] == 1
    assert (await rs.replay_dead_letters(ctx.org_id, max_attempts=3))["exhausted"] == 1
    assert row.attempts == 3


# ---- P1: full-backlog deterministic replay with capture-origin lineage ----

class _FakeBus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


class _Rec:
    def __init__(self, i):
        self.id = uuid.uuid4()
        self.organization_id = uuid.uuid4()
        self.source = "gmail"
        self.event_type = "email.received"
        self.external_id = f"e{i}"
        self.external_version = "1"
        self.user_id = None
        self.integration_id = None


class _FakeSignalRepo:
    def __init__(self, n):
        self.n = n

    async def stream_replay(self, org_id, *, source=None, since=None, batch_size=1000):
        for i in range(self.n):
            yield _Rec(i)


async def test_replay_paginates_full_backlog(monkeypatch):
    bus = _FakeBus()
    monkeypatch.setattr(RP, "event_bus", bus)
    rs = ReplayService(db=None)  # type: ignore[arg-type]
    rs.repo = _FakeSignalRepo(5)  # type: ignore[assignment]

    # Unbounded (limit=None) replays the WHOLE backlog (not just one batch).
    n = await rs.replay(uuid.uuid4(), limit=None, batch_size=2)
    assert n == 5 and len(bus.events) == 5
    # Capture-origin lineage stamped on every replayed pointer.
    assert all(e.payload["origin"] == "replay" for e in bus.events)


async def test_replay_respects_safety_cap(monkeypatch):
    bus = _FakeBus()
    monkeypatch.setattr(RP, "event_bus", bus)
    rs = ReplayService(db=None)  # type: ignore[arg-type]
    rs.repo = _FakeSignalRepo(10)  # type: ignore[assignment]

    n = await rs.replay(uuid.uuid4(), limit=3, batch_size=2)
    assert n == 3 and len(bus.events) == 3  # bounded against accidental fleet-wide flood
