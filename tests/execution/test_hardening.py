"""Execution hardening: recovery, DLQ, rate limiting, relay, timeout, telemetry.
(Idempotent-consumer inbox dedupe is owned and tested by each consumer domain —
tests/observation, tests/learning.)"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from navis.execution.dead_letter import ExecutionDeadLetterService
from navis.execution.providers.base import Provider, ProviderError
from navis.execution.ratelimit import InMemoryRateLimiter
from navis.execution.recovery import ExecutionRecoveryService
from navis.execution.service import ExecutionService
from navis.execution.status import ExecutionStatus
from navis.execution.telemetry import ProviderTelemetry
from navis.execution.worker import execute_pass
from navis.execution.retry import ExecutionRetryPolicy
from tests.execution.fakes import DECISION, ORG_A, USER, FakeExecutionRepo

FAST = ExecutionRetryPolicy(max_attempts=2, initial_backoff=0.0, max_backoff=0.0, jitter=0.0)


async def _approved(repo, action="send_email", payload=None, decision=DECISION):
    """Default action is send_email with a COMPLETE payload.

    It used to be create_task, which was a convenient always-succeeds provider —
    conveniently, because it succeeded at nothing: there is no task system, and
    it now refuses every payload rather than reporting a task it never made.
    These tests are about worker mechanics (rate limiting, telemetry, retries),
    so they need an action that genuinely executes.
    """
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        ORG_A, decision, action,
        payload or {"to": "a@x.com", "subject": "x", "body": "a real body"},
        reasoning_signal_ids=["s1"], reasoning_version=1, state_watermark="w",
    )
    await svc.approve(req.id, USER)
    return svc, req


class AlwaysFailProvider(Provider):
    name = "tasks"
    action_type = "create_task"

    async def validate(self, payload):
        pass

    async def execute(self, payload):
        raise ProviderError("boom", retryable=True)

    async def status(self, reference):
        return "x"


# ---- P0-1 stale recovery ----

async def test_recovery_redrives_stale_executing():
    repo = FakeExecutionRepo()
    svc, req = await _approved(repo)
    await svc.begin_execution(req)  # -> EXECUTING
    req.updated_at = datetime(2020, 1, 1, tzinfo=timezone.utc)  # simulate dead worker

    result = await ExecutionRecoveryService(repo, stale_seconds=300).recover()
    assert result["recovered"] == 1
    assert (await repo.get(req.id)).status == ExecutionStatus.FAILED.value
    # FAILED + attempts remaining => re-claimable (no new status).
    claim = await repo.claim_executable(max_attempts=5)
    assert any(r.id == req.id for r in claim)


async def test_recovery_ignores_fresh_executing():
    repo = FakeExecutionRepo()
    svc, req = await _approved(repo)
    await svc.begin_execution(req)  # updated_at = now (fresh)
    result = await ExecutionRecoveryService(repo, stale_seconds=300).recover()
    assert result["recovered"] == 0
    assert (await repo.get(req.id)).status == ExecutionStatus.EXECUTING.value


# ---- P0-3 DLQ ----

async def test_exhausted_attempts_dead_letters_then_replays():
    repo = FakeExecutionRepo()
    # AlwaysFailProvider declares action_type="create_task", so the row must
    # match it. It supplies its own permissive `validate`, so the real (now
    # unsupported) TasksProvider is not involved.
    _, req = await _approved(repo, action="create_task", payload={"title": "x"})
    providers = {"create_task": AlwaysFailProvider()}

    res = await execute_pass(repo, providers=providers, policy=FAST, max_attempts=1)
    assert res["failed"] == 1
    assert len(repo.dead_letters) == 1  # not lost
    assert (await repo.get(req.id)).status == ExecutionStatus.FAILED.value

    replay = await ExecutionDeadLetterService(repo).replay()
    assert replay["replayed"] == 1
    assert (await repo.get(req.id)).attempts == 0  # reset for re-drive
    assert any(r.id == req.id for r in await repo.claim_executable(max_attempts=1))


async def test_validation_error_dead_letters_immediately():
    repo = FakeExecutionRepo()
    # send_email without subject -> GmailProvider.validate fails (non-retryable)
    _, req = await _approved(repo, action="send_email", payload={"to": "a@x.com"})
    res = await execute_pass(repo, policy=FAST, max_attempts=3)
    assert res["failed"] == 1
    assert len(repo.dead_letters) == 1  # validation errors never recover


# ---- poison request: unknown action_type must never wedge the pass ----

async def test_missing_provider_fails_closed_without_wedging_the_pass():
    """Regression: a claimed request whose action_type has no registered provider used to
    call fail_execution on an APPROVED row — but APPROVED->FAILED is not a legal edge, so
    InvalidTransition escaped execute_pass, aborted the WHOLE pass (rollback), and the
    poison row was re-claimed forever: a permanent org-wide execution wedge. It must fail
    CLOSED through the machine (EXECUTING->FAILED), dead-letter exactly once, exhaust its
    retry budget (a missing provider cannot succeed on retry), and let the rest of the
    batch execute. DLQ replay stays the deliberate re-drive path once a provider ships."""
    repo = FakeExecutionRepo()
    _, poison = await _approved(repo, action="time_travel", decision=uuid.uuid4())
    _, _ok = await _approved(repo, action="create_task", decision=uuid.uuid4())

    res = await execute_pass(repo, policy=FAST, max_attempts=3)
    assert res["failed"] == 1 and res["executed"] == 1        # pass survived the poison row
    assert (await repo.get(poison.id)).status == ExecutionStatus.FAILED.value
    assert len(repo.dead_letters) == 1                        # recorded exactly once
    assert not any(r.id == poison.id for r in await repo.claim_executable(max_attempts=3))
    assert (await ExecutionDeadLetterService(repo).replay())["replayed"] == 1
    assert any(r.id == poison.id for r in await repo.claim_executable(max_attempts=3))


# ---- P0-4 rate limiting ----

async def test_rate_limiter_throttles_without_dropping():
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    for i in range(2):
        r, _ = await svc.request(
            ORG_A, uuid.uuid4(), "send_email",
            {"to": f"t{i}@x.com", "subject": f"t{i}", "body": "a real body"},
            reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
        )
        await svc.approve(r.id, USER)

    limiter = InMemoryRateLimiter({"gmail": 1}, clock=lambda: 0)  # 1 per (fixed) window
    res = await execute_pass(repo, policy=FAST, rate_limiter=limiter, max_attempts=3)
    assert res["executed"] == 1 and res["skipped"] == 1  # one ran, one throttled (still APPROVED)
    approved_left = [r for r in repo.rows.values() if r.status == ExecutionStatus.APPROVED.value]
    assert len(approved_left) == 1  # the throttled one survives for the next pass


# ---- crash consistency: EXECUTING must be durable BEFORE any provider runs ----

async def test_executing_is_committed_before_any_provider_runs():
    """Regression: the whole pass used to run in ONE transaction, so a worker crash
    mid-pass rolled every claimed row back to APPROVED — re-firing providers whose side
    effects had already happened (unbounded duplicates across a crash loop, attempts
    never durably counted) and leaving the stale-EXECUTING recovery service nothing it
    could ever reap. The worker must commit the EXECUTING batch (phase 1) before any
    provider executes, and commit each terminal transition as it lands (phase 2)."""
    from navis.execution.contract import ProviderResult

    repo = FakeExecutionRepo()
    await _approved(repo, decision=uuid.uuid4())
    await _approved(repo, decision=uuid.uuid4())

    class ProbeProvider(Provider):
        name = "tasks"
        action_type = "create_task"

        async def validate(self, payload):
            pass

        async def execute(self, payload):
            repo.log.append("execute")
            return ProviderResult(ok=True, reference=None)

        async def status(self, reference):
            return "x"

    await execute_pass(repo, providers={"create_task": ProbeProvider()}, policy=FAST, max_attempts=3)
    # Durable EXECUTING first, then each terminal transition commits as it lands.
    assert repo.log == ["commit", "execute", "commit", "execute", "commit"]


# ---- relay: poison outbox rows must drain, never wedge ----

async def test_relay_drains_poison_event_and_publishes_the_rest():
    """Regression (production fix in worker.relay_pass): a stored event_type outside the
    EventType enum (drift across deploys) used to raise ValueError, abort the batch, and —
    batches being oldest-first — wedge the relay for every org forever. It must drain the
    poison row (marked published, kept for audit) and keep publishing the rest."""
    from types import SimpleNamespace

    from navis.execution.worker import relay_pass

    repo = FakeExecutionRepo()
    svc, req = await _approved(repo)  # emits real REQUESTED/APPROVED outbox events
    repo.events.insert(0, SimpleNamespace(  # oldest row is poison
        id=uuid.uuid4(), organization_id=ORG_A, execution_id=req.id,
        event_type="retired_event_type", payload={}, published_at=None,
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    ))
    n = await relay_pass(repo)
    assert n == len(repo.events)                       # poison drained + real ones published
    assert await repo.pending_events() == []           # nothing left to wedge the next pass


# ---- provider execute timeout: liveness bound owned by the retry layer ----

async def test_hung_provider_times_out_and_is_not_retried_in_call():
    """Regression (production fix in retry.py): provider.execute had NO deadline — a hung
    real sender would pin a worker slot forever and outlive the stale-recovery window,
    letting a second worker re-run the request while the first was still in flight. A
    timeout must fail the call (ambiguous outcome -> NOT retried in-call; the pass-level
    attempts budget owns re-drive)."""
    import asyncio

    calls = []

    class HungProvider(Provider):
        name = "tasks"
        action_type = "create_task"

        async def validate(self, payload):
            pass

        async def execute(self, payload):
            from navis.execution.contract import ProviderResult

            calls.append(1)
            await asyncio.sleep(5)  # cancelled by the timeout; return satisfies the ABC
            return ProviderResult(ok=True)

        async def status(self, reference):
            return "x"

    from navis.execution.retry import ExecutionRetryPolicy, execute_with_retry

    policy = ExecutionRetryPolicy(max_attempts=3, initial_backoff=0.0, max_backoff=0.0,
                                  jitter=0.0, execute_timeout_s=0.01)
    with pytest.raises(ProviderError, match="timed out"):
        await execute_with_retry(HungProvider(), {"title": "x"}, policy)
    assert len(calls) == 1  # ambiguous side effect: never retried within the call


# ---- recovery race: terminal transitions must not abort the pass ----

async def test_recovery_race_is_contained_to_one_request():
    """Regression (production fix in worker._settle): if stale recovery reaps an EXECUTING
    row while its provider is still running, the worker's terminal transition raises
    InvalidTransition. Uncontained, it aborted the rest of the pass, stranding every
    remaining runnable row. It must cost only the raced request."""
    from navis.execution.contract import ProviderReference, ProviderResult

    repo = FakeExecutionRepo()
    _, raced = await _approved(repo, decision=uuid.uuid4())
    _, healthy = await _approved(repo, decision=uuid.uuid4())

    class RacedProvider(Provider):
        name = "tasks"
        action_type = "create_task"

        async def validate(self, payload):
            pass

        async def execute(self, payload):
            row = repo.rows[raced.id]
            if row.status == ExecutionStatus.EXECUTING.value:
                row.status = ExecutionStatus.FAILED.value  # simulate recovery reaping mid-flight
            return ProviderResult(ok=True, reference=ProviderReference("tasks", "t-1"))

        async def status(self, reference):
            return "x"

    res = await execute_pass(repo, providers={"create_task": RacedProvider()},
                             policy=FAST, max_attempts=3)
    assert res["executed"] == 2                        # pass completed; no escape
    assert (await repo.get(healthy.id)).status == ExecutionStatus.EXECUTED.value


# ---- recovery: exhausted attempts must reach the DLQ, never vanish ----

async def test_recovery_dead_letters_a_request_whose_attempts_are_exhausted():
    """Regression (production fix in recovery.py): a crash-looped request spends its
    attempts via begin_execution, but only the WORKER's failure paths dead-lettered.
    A request whose final attempt died with its worker was recovered to FAILED with an
    exhausted budget — unclaimable, un-replayed, absent from the DLQ: silently lost,
    violating the DLQ's "never lose a failed execution" contract. Recovery must
    dead-letter it; DLQ replay then remains the deliberate re-drive path."""
    repo = FakeExecutionRepo()
    svc, req = await _approved(repo)
    for _ in range(3):  # crash loop: each begin consumes an attempt, worker dies
        await svc.begin_execution(req)
        req.updated_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        await ExecutionRecoveryService(repo, stale_seconds=300, max_attempts=3).recover()

    assert (await repo.get(req.id)).attempts == 3
    assert not await repo.claim_executable(max_attempts=3)      # budget exhausted
    assert len(repo.dead_letters) == 1                          # NOT lost — dead-lettered once
    replay = await ExecutionDeadLetterService(repo).replay()
    assert replay["replayed"] == 1                              # re-drive path intact


async def test_in_memory_rate_limiter_stays_bounded_across_windows():
    """Regression (production fix in ratelimit.py): the (key, second)-keyed counter dict
    accumulated one entry per elapsed second forever; it must keep only the current
    window per key."""
    t = {"now": 0}
    limiter = InMemoryRateLimiter({"gmail": 1}, clock=lambda: t["now"])
    for second in range(50):
        t["now"] = second
        assert await limiter.allow("gmail") is True   # fresh window each second
    assert len(limiter._buckets) == 1                 # one entry per key, not per second
    assert await limiter.allow("gmail") is False      # same window: limit enforced


# ---- DLQ replay: deterministic, starvation-free ----

async def test_dlq_replay_resolves_unreplayable_rows_instead_of_starving():
    """Regression (production fix in dead_letter.replay + repository ordering): a dead
    letter whose request is gone or terminal (e.g. cancelled after dead-lettering) was
    skipped but never resolved — permanently occupying the front of the LIMIT window and
    starving every replayable dead letter behind it. Such rows must be RESOLVED (without
    claiming a replay); FAILED requests still replay with a reset budget."""
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    dlq = ExecutionDeadLetterService(repo)

    # Row 1: dead-lettered, then human-cancelled -> never replayable.
    _, cancelled = await _approved(repo, decision=uuid.uuid4())
    await svc.begin_execution(cancelled)
    await svc.fail_execution(cancelled, "boom")
    await dlq.record(cancelled, provider="tasks", error="boom")
    await svc.cancel(cancelled.id)

    # Row 2: dead-lettered, still FAILED -> replayable.
    _, failed = await _approved(repo, decision=uuid.uuid4())
    await svc.begin_execution(failed)
    await svc.fail_execution(failed, "boom")
    await dlq.record(failed, provider="tasks", error="boom")

    out = await dlq.replay()
    assert out == {"replayed": 1, "resolved": 1}
    assert (await repo.get(failed.id)).attempts == 0                 # re-drivable
    assert all(d.resolved_at is not None for d in repo.dead_letters)  # window is clear
    cancelled_dl = next(d for d in repo.dead_letters if d.execution_id == cancelled.id)
    assert cancelled_dl.replayed_at is None                           # resolved, not "replayed"
    assert (await dlq.replay()) == {"replayed": 0, "resolved": 0}     # nothing re-scanned


# ---- P1 telemetry ----

async def test_telemetry_records_provider_success():
    repo = FakeExecutionRepo()
    await _approved(repo)
    tel = ProviderTelemetry()
    await execute_pass(repo, policy=FAST, telemetry=tel, max_attempts=3)
    assert tel.counters.get("gmail.calls") == 1
    assert tel.success_rate("gmail") == 1.0


async def test_an_unroutable_row_fails_itself_without_killing_the_pass():
    """A row whose action_type has no registered provider must fail alone.

    APPROVED -> FAILED is not a legal transition, so failing straight from the
    claimed state raised InvalidTransition out of `execute_pass` and took every
    other claimed row down with it — one bad row stalling the whole queue.

    The row is written straight into the repository rather than through
    `ExecutionService.request`, because intake now refuses an unregistered
    action_type outright (UnknownAction). That closes the front door but not this
    case: a row PERSISTED by an earlier deploy whose action type has since been
    unregistered — a connector removed, a spec renamed — is still sitting in the
    table, still claimable, and must still fail alone. This constructs exactly
    that row.
    """
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    orphan, _ = await repo.upsert_request(dict(
        id=uuid.uuid4(), organization_id=ORG_A, decision_id=uuid.uuid4(),
        reasoning_signal_ids=[], source_signal_event_ids=[], reasoning_version=1,
        state_watermark="w", action_type="unknown_action_type",
        status=ExecutionStatus.REQUESTED.value,
        idempotency_key="legacy-unregistered", obligation_key=None,
        payload={"x": 1}, provider_reference={}, attempts=0, origin="proactive",
    ))
    await svc.approve(orphan.id, USER)
    _, healthy = await _approved(repo, decision=uuid.uuid4())

    res = await execute_pass(repo, policy=FAST, max_attempts=3)

    assert res["failed"] == 1
    assert res["executed"] == 1, "the healthy row must still run"
    assert (await repo.get(orphan.id)).status == ExecutionStatus.FAILED.value
    assert (await repo.get(healthy.id)).status == ExecutionStatus.EXECUTED.value
    assert any(d.execution_id == orphan.id for d in repo.dead_letters)


# ---- retry CLASSIFICATION: the worker must read ProviderError.retryable ----


class _CountingProvider(Provider):
    """Records how many times the external system was actually reached."""

    name = "tasks"
    action_type = "create_task"

    def __init__(self, *, retryable: bool) -> None:
        self.calls = 0
        self._retryable = retryable

    async def validate(self, payload):
        pass

    async def execute(self, payload):
        self.calls += 1
        raise ProviderError("the target record was deleted", retryable=self._retryable)

    async def status(self, reference):
        return "x"


async def test_a_permanent_provider_failure_reaches_the_provider_once():
    """`ProviderError.retryable=False` is a connector saying "this can never
    succeed". Until the worker read it, that claim was ignored.

    `execute_with_retry` consults it (tenacity retries only retryable errors),
    but a non-retryable ProviderError propagated out of it into the worker's
    catch-all `except Exception`, which classifies everything as TRANSIENT. So
    the row went FAILED with attempts remaining, the next pass re-claimed it,
    and the provider was called again — measured at FIVE external calls for a
    failure declared terminal on the first.

    What must hold: one call, budget exhausted, one dead letter carrying the
    connector's own message.
    """
    repo = FakeExecutionRepo()
    _, req = await _approved(repo, action="create_task", payload={"title": "x"})
    provider = _CountingProvider(retryable=False)

    for _ in range(6):
        await execute_pass(
            repo, providers={"create_task": provider}, policy=FAST, max_attempts=5
        )

    assert provider.calls == 1, (
        f"a permanently-failed action reached the external system "
        f"{provider.calls} times"
    )
    row = await repo.get(req.id)
    assert row.status == ExecutionStatus.FAILED.value
    assert row.attempts == 5, "the budget is spent, so the row stops being claimable"
    assert len(repo.dead_letters) == 1
    assert "the target record was deleted" in repo.dead_letters[0].error


async def test_a_retryable_provider_failure_still_spends_its_whole_budget():
    """The other direction, which the fix must not break: a connector reporting
    a transient failure keeps every attempt it is entitled to."""
    repo = FakeExecutionRepo()
    _, req = await _approved(repo, action="create_task", payload={"title": "x"})
    provider = _CountingProvider(retryable=True)

    for _ in range(6):
        await execute_pass(
            repo, providers={"create_task": provider}, policy=FAST, max_attempts=3
        )

    # FAST allows 2 in-provider retries per attempt, across 3 worker attempts.
    assert provider.calls == 6
    assert (await repo.get(req.id)).attempts == 3
    assert len(repo.dead_letters) == 1
