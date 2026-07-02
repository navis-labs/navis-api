"""Execution V2: status machine, idempotency, lifecycle, outbox, providers, retry."""
from __future__ import annotations

import uuid

import pytest

from navis.execution.idempotency import execution_id, idempotency_key
from navis.execution.models import ActionType
from navis.execution.providers import GmailProvider, TasksProvider, get_provider
from navis.execution.providers.base import ProviderError, ProviderValidationError
from navis.execution.proposer import propose_from_decisions
from navis.execution.retry import ExecutionRetryPolicy, execute_with_retry
from navis.execution.service import ExecutionNotFound, ExecutionService
from navis.execution.status import (
    ExecutionStatus,
    InvalidTransition,
    assert_transition,
    can_transition,
)
from navis.execution.worker import execute_pass, relay_pass
from navis.platform.events import EventType
from tests.execution.fakes import DECISION, ORG_A, ORG_B, USER, FakeExecutionRepo, event_types

FAST = ExecutionRetryPolicy(max_attempts=4, initial_backoff=0.0, max_backoff=0.0, jitter=0.0)


# ---- Status machine ----

def test_valid_and_invalid_transitions():
    assert can_transition(ExecutionStatus.REQUESTED, ExecutionStatus.APPROVED)
    assert can_transition(ExecutionStatus.APPROVED, ExecutionStatus.EXECUTING)
    assert can_transition(ExecutionStatus.EXECUTING, ExecutionStatus.EXECUTED)
    assert can_transition(ExecutionStatus.FAILED, ExecutionStatus.EXECUTING)  # retry
    assert not can_transition(ExecutionStatus.REQUESTED, ExecutionStatus.EXECUTED)  # no jump
    assert not can_transition(ExecutionStatus.EXECUTED, ExecutionStatus.EXECUTING)  # terminal
    with pytest.raises(InvalidTransition):
        assert_transition(ExecutionStatus.REQUESTED, ExecutionStatus.EXECUTED)


# ---- Idempotency ----

def test_idempotency_is_deterministic():
    k1 = idempotency_key(ORG_A, DECISION, "send_email", {"to": "a@x.com"})
    k2 = idempotency_key(ORG_A, DECISION, "send_email", {"to": "a@x.com"})
    assert k1 == k2
    assert execution_id(ORG_A, k1) == execution_id(ORG_A, k2)
    # payload-sensitive + tenant-sensitive
    assert idempotency_key(ORG_A, DECISION, "send_email", {"to": "b@x.com"}) != k1
    assert execution_id(ORG_B, k1) != execution_id(ORG_A, k1)


async def test_request_is_idempotent_no_duplicate():
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    payload = {"to": "a@x.com", "subject": "Hi"}
    r1, c1 = await svc.request(ORG_A, DECISION, "send_email", payload,
                               reasoning_signal_ids=["s1"], reasoning_version=1, state_watermark="w")
    r2, c2 = await svc.request(ORG_A, DECISION, "send_email", payload,
                               reasoning_signal_ids=["s1"], reasoning_version=1, state_watermark="w")
    assert c1 is True and c2 is False  # second is a no-op
    assert r1.id == r2.id
    assert event_types(repo).count(EventType.ACTION_REQUESTED.value) == 1  # one event only


# ---- Approval lifecycle + outbox ----

async def _requested(repo):
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(ORG_A, DECISION, "send_email", {"to": "a@x.com", "subject": "Hi"},
                               reasoning_signal_ids=["s1"], reasoning_version=1, state_watermark="wm-9")
    return svc, req


async def test_approve_emits_event_and_no_provider_call():
    repo = FakeExecutionRepo()
    svc, req = await _requested(repo)
    approved = await svc.approve(req.id, USER)
    assert approved.status == ExecutionStatus.APPROVED.value
    assert approved.approved_by == USER and approved.approved_at is not None
    assert EventType.ACTION_APPROVED.value in event_types(repo)
    # No EXECUTED/provider effect happened during approve.
    assert EventType.ACTION_EXECUTED.value not in event_types(repo)


async def test_double_approval_is_rejected():
    repo = FakeExecutionRepo()
    svc, req = await _requested(repo)
    await svc.approve(req.id, USER)
    with pytest.raises(InvalidTransition):
        await svc.approve(req.id, USER)  # APPROVED -> APPROVED is not allowed


async def test_reject_and_cancel_paths():
    repo = FakeExecutionRepo()
    svc, req = await _requested(repo)
    rejected = await svc.reject(req.id, USER, reason="not now")
    assert rejected.status == ExecutionStatus.REJECTED.value
    with pytest.raises(InvalidTransition):
        await svc.cancel(req.id)  # REJECTED is terminal


async def test_missing_request_raises():
    svc = ExecutionService(repo=FakeExecutionRepo())
    with pytest.raises(ExecutionNotFound):
        await svc.approve(uuid.uuid4(), USER)


# ---- Worker execute pass + lineage hook ----

async def test_worker_executes_approved_and_emits_outcome_hook():
    repo = FakeExecutionRepo()
    svc, req = await _requested(repo)
    await svc.approve(req.id, USER)

    result = await execute_pass(repo, policy=FAST)
    assert result["executed"] == 1 and result["failed"] == 0

    row = await repo.get(req.id)
    assert row.status == ExecutionStatus.EXECUTED.value
    assert row.provider_reference.get("provider") == "gmail"
    # ACTION_EXECUTED carries the OutcomeReference hook for Observation.
    executed = [e for e in repo.events if e.event_type == EventType.ACTION_EXECUTED.value][0]
    outcome = executed.payload["outcome"]
    assert outcome["decision_id"] == str(DECISION)
    assert outcome["reasoning_signal_ids"] == ["s1"]
    assert outcome["state_watermark"] == "wm-9"


async def test_worker_marks_failed_on_validation_error():
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    # Missing 'subject' -> GmailProvider.validate fails (not retryable).
    req, _ = await svc.request(ORG_A, DECISION, "send_email", {"to": "a@x.com"},
                               reasoning_signal_ids=[], reasoning_version=1, state_watermark="w")
    await svc.approve(req.id, USER)
    result = await execute_pass(repo, policy=FAST)
    assert result["executed"] == 0 and result["failed"] == 1
    assert (await repo.get(req.id)).status == ExecutionStatus.FAILED.value


# ---- Outbox relay ----

async def test_relay_publishes_then_marks_published():
    repo = FakeExecutionRepo()
    svc, req = await _requested(repo)
    await svc.approve(req.id, USER)
    assert len(await repo.pending_events()) >= 1
    n = await relay_pass(repo)
    assert n >= 1
    assert await repo.pending_events() == []  # all published, none pending


# ---- Providers ----

async def test_provider_registry_and_dry_run():
    p = get_provider(ActionType.SEND_EMAIL.value)
    assert isinstance(p, GmailProvider)
    res = await p.execute({"to": "a@x.com", "subject": "Hi"})
    assert res.ok and res.reference.extra.get("dry_run") is True  # safe by default


def test_injected_empty_registry_means_no_providers():
    """Regression (production fix in get_provider): `registry or _DEFAULT_PROVIDERS`
    treated an explicitly injected EMPTY registry as falsy and silently fell back to the
    LIVE defaults — a hidden fallback that would dispatch real providers where the caller
    wired none. Empty must mean empty (lookups miss -> the worker fails closed); only
    None selects the default registry."""
    assert get_provider(ActionType.CREATE_TASK.value, {}) is None
    assert get_provider(ActionType.CREATE_TASK.value) is not None  # None -> defaults


def test_every_action_type_has_a_registered_provider():
    """Registry <-> enum contract lock: a declared ActionType with no provider is exactly the
    poison-request condition (the worker now fails it closed, but it should never ship). A new
    ActionType MUST land with its provider in the same change; unknown types stay None."""
    for t in ActionType:
        assert get_provider(t.value) is not None, f"no provider registered for {t.value}"
    assert get_provider("not_a_real_action") is None  # unknowns are None -> worker fails closed


async def test_provider_validation_raises():
    with pytest.raises(ProviderValidationError):
        await GmailProvider().validate({"to": "a@x.com"})  # no subject


# ---- Retry ----

class _FlakyProvider(TasksProvider):
    def __init__(self, fail_times: int):
        self._left = fail_times

    async def execute(self, payload):
        if self._left > 0:
            self._left -= 1
            raise ProviderError("transient", retryable=True)
        from navis.execution.contract import ProviderReference, ProviderResult
        return ProviderResult(ok=True, reference=ProviderReference("tasks", "t-1"))


async def test_retry_recovers_then_gives_up():
    ok = await execute_with_retry(_FlakyProvider(2), {"title": "x"}, FAST)
    assert ok.ok  # recovered after 2 transient failures
    with pytest.raises(ProviderError):
        await execute_with_retry(_FlakyProvider(99), {"title": "x"}, FAST)  # exhausts


# ---- Proposer (Decision -> Execution) ----

def test_proposer_maps_categories_and_carries_lineage():
    decisions = [
        {"decision_id": str(uuid.uuid4()), "category": "risk", "title": "Commitment slipping",
         "evidence": {"speaker_email": "a@acme.com"}, "reasoning_signal_ids": ["s1"],
         "reasoning_version": 1, "state_watermark": "wm"},
        {"decision_id": str(uuid.uuid4()), "category": "delegation", "title": "Hub company",
         "evidence": {}, "reasoning_signal_ids": ["s2"], "reasoning_version": 1, "state_watermark": "wm"},
        {"decision_id": str(uuid.uuid4()), "category": "priority", "title": "Load",
         "evidence": {}, "reasoning_signal_ids": ["s3"], "reasoning_version": 1, "state_watermark": "wm"},
    ]
    proposed = propose_from_decisions(decisions, ORG_A)
    # risk-with-contact -> email; delegation -> task; priority -> skipped (informational)
    kinds = [p.action_type for p in proposed]
    assert ActionType.SEND_EMAIL.value in kinds
    assert ActionType.CREATE_TASK.value in kinds
    assert len(proposed) == 2
    assert all(p.org_id == ORG_A and p.reasoning_signal_ids for p in proposed)


def test_proposer_is_deterministic():
    decisions = [{"decision_id": str(DECISION), "category": "risk", "title": "X",
                  "evidence": {"person_email": "p@x.com"}, "reasoning_signal_ids": ["s"],
                  "reasoning_version": 1, "state_watermark": "w"}]
    a = propose_from_decisions(decisions, ORG_A)
    b = propose_from_decisions(decisions, ORG_A)
    assert [(p.action_type, p.payload) for p in a] == [(p.action_type, p.payload) for p in b]


# ---- Multi-tenancy ----

async def test_same_decision_different_org_distinct_requests():
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    payload = {"to": "a@x.com", "subject": "Hi"}
    a, _ = await svc.request(ORG_A, DECISION, "send_email", payload,
                             reasoning_signal_ids=[], reasoning_version=1, state_watermark="w")
    b, _ = await svc.request(ORG_B, DECISION, "send_email", payload,
                             reasoning_signal_ids=[], reasoning_version=1, state_watermark="w")
    assert a.id != b.id and a.organization_id != b.organization_id
