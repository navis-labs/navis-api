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


async def _approved(repo, action="create_task", payload=None, decision=DECISION):
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        ORG_A, decision, action, payload or {"title": "x"},
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
    _, req = await _approved(repo)
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
        r, _ = await svc.request(ORG_A, uuid.uuid4(), "create_task", {"title": f"t{i}"},
                                 reasoning_signal_ids=[], reasoning_version=1, state_watermark="w")
        await svc.approve(r.id, USER)

    limiter = InMemoryRateLimiter({"tasks": 1}, clock=lambda: 0)  # 1 per (fixed) window
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
    assert tel.counters.get("tasks.calls") == 1
    assert tel.success_rate("tasks") == 1.0
