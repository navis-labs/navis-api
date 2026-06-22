"""P0-1: DLQ persists to PostgreSQL (primary) and replays from it."""
from __future__ import annotations

import uuid
from typing import Any

from navis.signals.base import SignalEvent, SignalEventType, SignalSource
from navis.signals.reliability.dead_letter import DeadLetterQueue
from navis.signals.reliability.replay import ReplayService
from tests.signals.fakes import FakeIngestion, make_ctx


class FakeDLQRepo:
    """Captures DLQ writes (stands in for the PostgreSQL DeadLetterRepository)."""

    def __init__(self, unresolved: list | None = None) -> None:
        self.records: list[dict[str, Any]] = []
        self._unresolved = unresolved or []
        self.replayed: list[uuid.UUID] = []

    async def record(self, **kwargs: Any):
        self.records.append(kwargs)
        return kwargs

    async def list_unresolved(self, org_id, *, limit=100):
        return self._unresolved

    async def mark_replayed(self, ids, *, resolved):
        self.replayed.extend(ids)

    async def record_replay_failure(self, dlq_id, *, error, max_attempts):
        for row in self._unresolved:
            if row.id == dlq_id:
                row.attempts = getattr(row, "attempts", 0) + 1
                row.last_error = error
                return row.attempts >= max_attempts
        return False


def _event(ctx, ext="m1"):
    return SignalEvent.create(
        ctx,
        source=SignalSource.GMAIL,
        event_type=SignalEventType.EMAIL_RECEIVED,
        external_id=ext,
        external_version="1",
    )


async def test_poison_event_persisted_to_postgres_primary():
    ctx = make_ctx()
    ev = _event(ctx)
    repo = FakeDLQRepo()
    dlq = DeadLetterQueue(repository=repo, mirror=False)  # no Redis in test

    await dlq.send(
        stage="ingest",
        error=RuntimeError("boom"),
        context={"org_id": str(ctx.org_id), "source": "gmail"},
        item=ev.to_dict(),
    )

    assert len(repo.records) == 1  # written to PG, not just Redis
    rec = repo.records[0]
    assert rec["stage"] == "ingest"
    assert rec["event_id"] == ev.event_id
    assert rec["dedupe_key"] == ev.dedupe_key
    assert rec["payload"]["external_id"] == "m1"  # full payload retained in PG


async def test_dead_letter_replay_recovers_from_postgres():
    ctx = make_ctx()
    good = _event(ctx, "good")
    poison = _event(ctx, "poison")

    class Row:
        def __init__(self, payload):
            self.id = uuid.uuid4()
            self.payload = payload

    repo = FakeDLQRepo(unresolved=[Row(good.to_dict()), Row(poison.to_dict())])
    ingestion = FakeIngestion(fail_on={"poison"})

    rs = ReplayService(db=None)  # repos built lazily; override with fakes
    rs.dlq_repo = repo
    rs.ingestion = ingestion

    result = await rs.replay_dead_letters(ctx.org_id)

    assert result["recovered"] == 1
    assert result["still_failing"] == 1
    assert len(repo.replayed) == 1  # only the recovered one marked resolved
    assert {e.external_id for e in ingestion.ingested} == {"good"}
