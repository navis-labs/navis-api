"""Outbox delivery semantics — `published_at` means HANDLED, not merely "published".

Regression cover for the stall that left 500+ signal_outbox rows unpublished since
July while new Gmail messages silently never became memory. Two distinct defects:

  1. `EventBus.publish()` returned None whether handlers succeeded, failed, or were
     never registered — so the relays marked every row published regardless.
  2. Subscriber registration rode on a Celery signal that does not fire for every
     pool type, so the relay could publish into an empty bus and lose events.
"""
from __future__ import annotations

import json
import uuid

import pytest

from navis.memory.outbox import relay_memory_pass
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.signals.outbox import relay_signals_pass


class FakeOutboxRow:
    def __init__(self, event_type: str, org_id: uuid.UUID | None = None, payload=None):
        self.id = uuid.uuid4()
        self.organization_id = org_id or uuid.uuid4()
        self.event_type = event_type
        self.payload = payload if payload is not None else {"event_id": str(uuid.uuid4())}


class FakeOutboxRepo:
    """Minimal stand-in for Signal/MemoryRepository's outbox surface."""

    def __init__(self, rows):
        self._rows = list(rows)
        self.published_ids: list[uuid.UUID] = []

    async def pending_outbox(self, *, limit: int = 500):
        return self._rows[:limit]

    async def mark_outbox_published(self, ids):
        self.published_ids.extend(ids)


@pytest.fixture(autouse=True)
def _isolate_bus():
    """Run each test against a clean handler table and no Redis mirror.

    The composition root is loaded FIRST and deliberately left memoized: the
    relays call `ensure_subscribers()` on every pass, so if the guard were still
    un-memoized here it would re-import the registry mid-test and re-attach every
    real handler onto the bus we just cleared (they then hit the real DB and
    fail). Loading it up front makes the subsequent clear stick.
    """
    from navis.platform.composition import ensure_subscribers

    ensure_subscribers()
    saved_handlers = {k: list(v) for k, v in event_bus._handlers.items()}
    saved_mirror = event_bus._mirror_to_redis
    event_bus._handlers.clear()
    event_bus._mirror_to_redis = False
    yield
    event_bus._handlers.clear()
    event_bus._handlers.update(saved_handlers)
    event_bus._mirror_to_redis = saved_mirror


# ---- EventBus.publish reports delivery outcome -----------------------------


async def test_publish_reports_success_for_every_handler():
    seen = []

    async def ok(event: DomainEvent) -> None:
        seen.append(event)

    event_bus.subscribe(EventType.GMAIL_RECEIVED, ok)
    delivery = await event_bus.publish(
        DomainEvent(type=EventType.GMAIL_RECEIVED, org_id=uuid.uuid4(), source="t", payload={})
    )
    assert len(seen) == 1
    assert (delivery.handlers, delivery.handled, delivery.failed) == (1, 1, 0)
    assert delivery.ok


async def test_publish_reports_failure_when_a_handler_raises():
    async def boom(event: DomainEvent) -> None:
        raise RuntimeError("db timeout")

    async def ok(event: DomainEvent) -> None:
        return None

    event_bus.subscribe(EventType.GMAIL_RECEIVED, boom)
    event_bus.subscribe(EventType.GMAIL_RECEIVED, ok)
    delivery = await event_bus.publish(
        DomainEvent(type=EventType.GMAIL_RECEIVED, org_id=uuid.uuid4(), source="t", payload={})
    )
    # One failed, one succeeded — and the pass is NOT ok, so the row stays retryable.
    assert (delivery.handlers, delivery.handled, delivery.failed) == (2, 1, 1)
    assert not delivery.ok


async def test_publish_with_no_handlers_is_ok_but_reports_zero():
    delivery = await event_bus.publish(
        DomainEvent(type=EventType.GMAIL_RECEIVED, org_id=uuid.uuid4(), source="t", payload={})
    )
    assert (delivery.handlers, delivery.failed) == (0, 0)
    assert delivery.ok


# ---- Relay reaches the REAL consumer ---------------------------------------


async def test_signal_relay_invokes_subscriber_and_marks_published():
    delivered = []

    async def consumer(event: DomainEvent) -> None:
        delivered.append(event.payload["event_id"])

    event_bus.subscribe(EventType.GMAIL_RECEIVED, consumer)
    row = FakeOutboxRow(EventType.GMAIL_RECEIVED.value)
    repo = FakeOutboxRepo([row])

    published = await relay_signals_pass(repo)

    assert delivered == [row.payload["event_id"]]   # the consumer actually ran
    assert published == 1
    assert repo.published_ids == [row.id]


async def test_memory_relay_invokes_subscriber_and_marks_published():
    delivered = []

    async def consumer(event: DomainEvent) -> None:
        delivered.append(event.payload["memory_id"])

    event_bus.subscribe(EventType.COMMITMENT_DETECTED, consumer)
    mem_id = str(uuid.uuid4())
    row = FakeOutboxRow(EventType.COMMITMENT_DETECTED.value, payload={"memory_id": mem_id})
    repo = FakeOutboxRepo([row])

    published = await relay_memory_pass(repo)

    assert delivered == [mem_id]
    assert published == 1
    assert repo.published_ids == [row.id]


# ---- Failed processing must NOT mark the row published ---------------------


async def test_signal_relay_keeps_failed_row_unpublished():
    async def boom(event: DomainEvent) -> None:
        raise RuntimeError("MemoryIngestionService exploded")

    event_bus.subscribe(EventType.GMAIL_RECEIVED, boom)
    row = FakeOutboxRow(EventType.GMAIL_RECEIVED.value)
    repo = FakeOutboxRepo([row])

    published = await relay_signals_pass(repo)

    assert published == 0
    assert repo.published_ids == []  # stays claimable — retried next pass


async def test_memory_relay_keeps_failed_row_unpublished():
    async def boom(event: DomainEvent) -> None:
        raise RuntimeError("embedding subscriber exploded")

    event_bus.subscribe(EventType.PERSON_DETECTED, boom)
    row = FakeOutboxRow(EventType.PERSON_DETECTED.value, payload={"memory_id": str(uuid.uuid4())})
    repo = FakeOutboxRepo([row])

    assert await relay_memory_pass(repo) == 0
    assert repo.published_ids == []


async def test_one_failure_does_not_block_healthy_rows():
    """A poison row must not stall the rest of the batch."""
    async def selective(event: DomainEvent) -> None:
        if event.payload.get("event_id") == "bad":
            raise RuntimeError("nope")

    event_bus.subscribe(EventType.GMAIL_RECEIVED, selective)
    good1 = FakeOutboxRow(EventType.GMAIL_RECEIVED.value, payload={"event_id": "good1"})
    bad = FakeOutboxRow(EventType.GMAIL_RECEIVED.value, payload={"event_id": "bad"})
    good2 = FakeOutboxRow(EventType.GMAIL_RECEIVED.value, payload={"event_id": "good2"})
    repo = FakeOutboxRepo([good1, bad, good2])

    published = await relay_signals_pass(repo)

    assert published == 2
    assert set(repo.published_ids) == {good1.id, good2.id}
    assert bad.id not in repo.published_ids


async def test_unknown_event_type_stays_unpublished():
    row = FakeOutboxRow("NoSuchEventTypeWasEverDefined")
    repo = FakeOutboxRepo([row])
    assert await relay_signals_pass(repo) == 0
    assert repo.published_ids == []


# ---- Replay is idempotent --------------------------------------------------


async def test_replaying_the_same_row_is_idempotent_for_the_consumer():
    """Re-delivery is expected (at-least-once). The consumer dedupes; the relay
    must publish the same row cleanly both times without erroring."""
    calls = []

    async def consumer(event: DomainEvent) -> None:
        calls.append(event.payload["event_id"])

    event_bus.subscribe(EventType.GMAIL_RECEIVED, consumer)
    row = FakeOutboxRow(EventType.GMAIL_RECEIVED.value)

    first = await relay_signals_pass(FakeOutboxRepo([row]))
    second = await relay_signals_pass(FakeOutboxRepo([row]))  # same row replayed

    assert first == second == 1
    assert calls == [row.payload["event_id"]] * 2  # consumer saw it twice...
    # ...and dedupe is the consumer's job — proven against the real
    # MemoryRepository.append in test_duplicate_signal_is_idempotent below.


async def test_time_budget_defers_remaining_rows_without_publishing_them():
    """A pass that runs out of wall-clock commits what it did and leaves the rest."""
    async def consumer(event: DomainEvent) -> None:
        return None

    event_bus.subscribe(EventType.GMAIL_RECEIVED, consumer)
    rows = [FakeOutboxRow(EventType.GMAIL_RECEIVED.value) for _ in range(5)]
    repo = FakeOutboxRepo(rows)

    # A zero budget means the deadline is already past on the first iteration.
    published = await relay_signals_pass(repo, time_budget=0.0)

    assert published == 0
    assert repo.published_ids == []  # nothing lost — all still pending


# ---- Duplicate signal must not create duplicate memory ---------------------


async def test_duplicate_signal_does_not_create_duplicate_memory():
    """The real dedupe guarantee behind at-least-once relay: ingesting the SAME
    signal twice appends the facts once, because memory_id is deterministic."""
    from navis.memory.service import MemoryIngestionService
    from tests.memory.fakes import FakeMemoryRepo, gmail_view

    view = gmail_view()
    repo = FakeMemoryRepo()

    async def getter(_event_id):
        return view

    svc = MemoryIngestionService(None, memory_repo=repo, signal_getter=getter)

    first = await svc.ingest_signal(view.event_id)
    second = await svc.ingest_signal(view.event_id)   # replay

    assert first > 0, "first ingest should write facts"
    assert second == 0, "replay must write nothing"
    assert len(repo.records) == first
    assert len({r.memory_id for r in repo.records}) == len(repo.records)


# ---- Subscriber registration in the actual worker execution path -----------


def test_composition_guard_wires_subscribers_in_a_fresh_process():
    """Reproduces the worker's execution path in a REAL fresh interpreter.

    In-process this cannot be tested honestly: once `navis.registry` is in
    sys.modules, re-importing it will not re-run the `@subscribe` decorators. A
    subprocess that imports ONLY the relay module — exactly what a Celery worker
    that never loaded the composition root looks like — proves the guard is what
    puts handlers on the bus, and that calling it twice does not double-register.
    """
    import subprocess
    import sys

    script = """
import json
import navis.signals.outbox          # relay module only, as celery would load it
from navis.platform.events import event_bus, EventType
before = len(event_bus._handlers.get(EventType.GMAIL_RECEIVED, []))
from navis.platform.composition import ensure_subscribers
ensure_subscribers()
after = len(event_bus._handlers.get(EventType.GMAIL_RECEIVED, []))
ensure_subscribers()                  # idempotent — must not double-register
again = len(event_bus._handlers.get(EventType.GMAIL_RECEIVED, []))
mem = len(event_bus._handlers.get(EventType.PERSON_DETECTED, []))
print("RESULT" + json.dumps({"before": before, "after": after, "again": again, "memory": mem}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT"))
    result = json.loads(line[len("RESULT"):])

    assert result["before"] == 0, "handlers existed before the guard ran — test is not honest"
    assert result["after"] > 0, "ensure_subscribers() did not register GmailReceived handlers"
    assert result["again"] == result["after"], "guard double-registered on second call"
    assert result["memory"] > 0, "memory-layer subscribers not registered"


def test_worker_signal_hooks_cover_every_celery_pool():
    """prefork/solo fire worker_process_init; threads/gevent/eventlet only fire
    worker_init. Both must be connected or a pool silently loses all handlers."""
    from celery.signals import worker_init, worker_process_init

    import navis.platform.queue.celery_app  # noqa: F401  (connects the hooks)

    import weakref

    def names(sig):
        # celery Signal.receivers is a list of (lookup_key, weakref-or-function) pairs.
        out = set()
        for _key, ref in sig.receivers:
            fn = ref() if isinstance(ref, weakref.ReferenceType) else ref
            if fn is not None:
                out.add(getattr(fn, "__name__", ""))
        return out

    assert "_register_subscribers_in_child" in names(worker_process_init)
    assert "_register_subscribers_in_main" in names(worker_init)


def test_every_beat_scheduled_task_is_registered():
    """Beat dispatches by task NAME; a name whose module was never imported is
    rejected by the worker with "Received unregistered task" and silently never
    runs. That is how Gmail polling, watch renewal and the morning brief were
    dead while the schedule still looked correct."""
    import navis.registry  # noqa: F401  (the composition root under test)
    from navis.platform.queue import celery_app

    scheduled = {entry["task"] for entry in celery_app.conf.beat_schedule.values()}
    missing = sorted(scheduled - set(celery_app.tasks))
    assert not missing, f"beat schedules tasks the worker cannot resolve: {missing}"
