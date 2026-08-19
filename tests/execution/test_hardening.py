"""Execution hardening: recovery, DLQ, rate limiting, inbox dedupe, telemetry."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from navis.execution.dead_letter import ExecutionDeadLetterService
from navis.execution.inbox import process_once
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


# ---- P1 inbox (idempotent consumer) ----

async def test_inbox_dedupes_duplicate_delivery():
    repo = FakeExecutionRepo()
    eid = uuid.uuid4()
    calls = []

    async def handler():
        calls.append(1)

    first = await process_once(repo, "observation", eid, handler)
    second = await process_once(repo, "observation", eid, handler)  # at-least-once redelivery
    assert first is True and second is False
    assert len(calls) == 1  # single processing


async def test_inbox_isolates_consumers():
    repo = FakeExecutionRepo()
    eid = uuid.uuid4()
    a = await process_once(repo, "observation", eid, _noop)
    b = await process_once(repo, "learning", eid, _noop)  # different consumer, same event
    assert a is True and b is True  # each consumer processes once


async def _noop():
    return None


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
