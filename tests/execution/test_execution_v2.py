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
from navis.execution.contract import ProviderResult
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
    # A body is REQUIRED — an email with none is refused before it reaches
    # Gmail (see providers/gmail.py). The fixture describes a real message so
    # these tests exercise the lifecycle, not the blank-email guard.
    req, _ = await svc.request(
        ORG_A, DECISION, "send_email",
        {"to": "a@x.com", "subject": "Hi", "body": "Following up on the deck."},
        reasoning_signal_ids=["s1"], reasoning_version=1, state_watermark="wm-9",
    )
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


async def test_double_approval_is_a_no_op_not_an_error():
    """Re-approving is idempotent: the outcome asked for already happened.

    This used to raise InvalidTransition, which surfaced to the user as
    "invalid execution transition executed -> approved (409)" AFTER their
    meeting had actually been scheduled — a double-click, a stale action list
    or a retry all produced an error for an operation that succeeded.
    """
    repo = FakeExecutionRepo()
    svc, req = await _requested(repo)
    first = await svc.approve(req.id, USER)
    approved_at = first.approved_at

    second = await svc.approve(req.id, uuid.uuid4())

    assert second.status == ExecutionStatus.APPROVED.value
    assert second.approved_by == USER, "the original approver must not be overwritten"
    assert second.approved_at == approved_at
    # Crucially: no second event, so no second side effect downstream.
    assert event_types(repo).count(EventType.ACTION_APPROVED.value) == 1


async def test_approving_an_already_executed_request_is_a_no_op():
    """The exact reported failure: approve arrives after execution finished."""
    repo = FakeExecutionRepo()
    svc, req = await _requested(repo)
    await svc.approve(req.id, USER)
    await svc.begin_execution(req)
    await svc.complete_execution(req, ProviderResult(ok=True))
    assert req.status == ExecutionStatus.EXECUTED.value

    again = await svc.approve(req.id, USER)

    assert again.status == ExecutionStatus.EXECUTED.value, "must not be dragged back"
    assert event_types(repo).count(EventType.ACTION_APPROVED.value) == 1


async def test_approving_while_executing_is_a_no_op():
    repo = FakeExecutionRepo()
    svc, req = await _requested(repo)
    await svc.approve(req.id, USER)
    await svc.begin_execution(req)

    again = await svc.approve(req.id, USER)

    assert again.status == ExecutionStatus.EXECUTING.value


@pytest.mark.parametrize("decline", ["reject", "cancel"])
async def test_approving_something_a_human_declined_still_conflicts(decline):
    """Idempotency must not silently resurrect a request someone refused."""
    repo = FakeExecutionRepo()
    svc, req = await _requested(repo)
    if decline == "reject":
        await svc.reject(req.id, USER, reason="not now")
    else:
        await svc.cancel(req.id)

    with pytest.raises(InvalidTransition):
        await svc.approve(req.id, USER)


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


async def test_provider_validation_raises():
    with pytest.raises(ProviderValidationError):
        await GmailProvider().validate({"to": "a@x.com"})  # no subject


# ---- Retry ----

class _FlakyProvider(TasksProvider):
    """Exercises RETRY MECHANICS, not task semantics.

    It borrows create_task's action_type for convenience, so it must override
    `validate` — TasksProvider's now refuses every payload (there is no task
    system, see providers/tasks.py) and would fail this test before a single
    retry was attempted.
    """

    def __init__(self, fail_times: int):
        self._left = fail_times

    async def validate(self, payload):
        return None

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
def test_proposer_only_emits_actions_it_can_author_honestly():
    """Phase 1 contract: propose nothing rather than propose something unsafe.

    This test previously asserted the OPPOSITE — that `risk` produced a
    send_email and `delegation` produced a create_task. Both shipped: the email
    had `body: ""` and went to a real person, and the task was filed in a system
    that does not exist. Until drafting lands, those categories yield nothing,
    and the founder reads them on the brief instead.
    """
    decisions = [
        {"decision_id": str(uuid.uuid4()), "category": "risk", "title": "Contact at risk",
         "evidence": {"person_email": "p@x.com"}, "reasoning_signal_ids": ["s1"],
         "reasoning_version": 1, "state_watermark": "wm"},
        {"decision_id": str(uuid.uuid4()), "category": "delegation", "title": "Hub company",
         "evidence": {}, "reasoning_signal_ids": ["s2"], "reasoning_version": 1,
         "state_watermark": "wm"},
        {"decision_id": str(uuid.uuid4()), "category": "priority", "title": "Load",
         "evidence": {}, "reasoning_signal_ids": ["s3"], "reasoning_version": 1,
         "state_watermark": "wm"},
    ]
    assert propose_from_decisions(decisions, ORG_A) == []


def test_proposer_emits_a_reply_candidate_and_carries_lineage():
    """The one category that CAN be authored honestly today: someone waiting on
    a reply. Lineage rides along so the resulting row is fully traceable."""
    decisions = [
        {"decision_id": str(uuid.uuid4()), "category": "waiting_reply",
         "title": "John is waiting for your reply",
         # rfc_message_id is what WaitingReplyReasoningEngine actually emits
         # (engines/waiting.py) and is now REQUIRED: it is the stable identity
         # of the reply obligation, so a candidate without one is withheld
         # rather than given an identity that cannot be de-duplicated.
         "evidence": {"person_email": "john@x.com", "thread_id": "t-1",
                      "rfc_message_id": "<contract-1@mail.example>",
                      "subject": "Contract", "waiting_days": 4.0},
         "reasoning_signal_ids": ["s9"], "reasoning_version": 1, "state_watermark": "wm"},
    ]
    proposed = propose_from_decisions(decisions, ORG_A)
    assert [p.action_type for p in proposed] == [ActionType.REPLY_TO_EMAIL.value]
    p = proposed[0]
    assert p.org_id == ORG_A
    assert p.reasoning_signal_ids == ["s9"]
    assert p.reasoning_version == 1
    assert p.state_watermark == "wm"
    # Proposed, but deliberately not sendable.
    assert p.payload["requires_draft"] is True


def test_proposer_is_deterministic():
    decisions = [{"decision_id": str(DECISION), "category": "waiting_reply", "title": "X",
                  "evidence": {"person_email": "p@x.com", "thread_id": "t-1",
                               "rfc_message_id": "<x-1@mail.example>",
                               "subject": "X", "waiting_days": 3.0},
                  "reasoning_signal_ids": ["s"],
                  "reasoning_version": 1, "state_watermark": "w"}]
    a = propose_from_decisions(decisions, ORG_A)
    b = propose_from_decisions(decisions, ORG_A)
    assert a, "determinism over an empty list would be vacuous"
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


# ---- provenance ID namespaces stay separate ---------------------------------
#
# `reasoning_signal_ids` is what Learning aggregates on and Intelligence cites
# as `supporting_reasoning_signal_ids`, so it must hold ONLY
# `reasoning_signals.signal_id` values. The chat scheduling path was writing
# `signal_events.id` into it (`active_signal_id` is looked up as a
# SignalEventRecord; `source_signal_event_id` is derived from it), which would
# have keyed a signal_intelligence insight to something that is not a reasoning
# signal — a reference nothing can ever resolve.


def test_chat_scheduling_writes_source_events_not_reasoning_ids():
    """The exact bug, pinned at the source: chat must not put a signal_events.id
    into reasoning_signal_ids."""
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[2]
           / "navis" / "api" / "chat_router.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Find every `reasoning_signal_ids=<value>` keyword passed anywhere in chat.
    passed: list[ast.AST] = [
        kw.value
        for node in ast.walk(tree) if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "reasoning_signal_ids"
    ]
    assert passed, "chat no longer passes reasoning_signal_ids at all"
    for value in passed:
        # Chat concludes nothing from a persisted ReasoningSignal, so the only
        # honest value it can pass is an empty list.
        assert isinstance(value, ast.List) and not value.elts, (
            "chat_router passes a non-empty reasoning_signal_ids; a signal_events.id "
            "there is unresolvable in the reasoning_signals namespace"
        )
    assert "source_signal_event_ids=source_signal_event_ids" in src, (
        "source-event provenance must be preserved in its own field, not dropped"
    )


async def test_execution_keeps_the_two_namespaces_apart():
    """Both survive the round-trip without being merged."""
    svc = ExecutionService(repo=FakeExecutionRepo())
    req, _ = await svc.request(
        ORG_A, DECISION, ActionType.SCHEDULE_MEETING.value, {"x": 1},
        reasoning_signal_ids=["11111111-1111-1111-1111-111111111111"],
        source_signal_event_ids=["22222222-2222-2222-2222-222222222222"],
        reasoning_version=3, state_watermark="wm-1",
    )
    assert req.reasoning_signal_ids == ["11111111-1111-1111-1111-111111111111"]
    assert req.source_signal_event_ids == ["22222222-2222-2222-2222-222222222222"]


async def test_source_events_default_empty_for_existing_callers():
    """Backward compatible: every pre-existing call site omits the new field."""
    svc = ExecutionService(repo=FakeExecutionRepo())
    req, _ = await svc.request(
        ORG_A, DECISION, ActionType.SCHEDULE_MEETING.value, {"y": 2},
        reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
    )
    assert req.source_signal_event_ids == []


# ---- PHASE 5.1 AUDIT: what the Decision -> Execution boundary actually carries ----
#
# These tests do not assert a desired design. They PIN the behaviour the audit
# measured, so a later phase that changes it has to change a test that says why.


def test_supporting_insight_ids_are_dropped_at_the_execution_boundary():
    """Phase 4 provenance stops at the proposer. PINNED, not endorsed.

    `Decision.to_dict()` emits `supporting_insight_ids` (decisions/base.py) —
    WHICH ORGANIZATIONAL FACTS informed this decision. The proposer reads only
    named keys, and that field is not one of them, so it never reaches
    `ProposedAction`, `ExecutionService.request`, `execution_requests`, the
    outbox payload, or `Outcome`.

    Consequence: "why did Navis perform this action?" resolves through
    reasoning_signal_ids only. The Intelligence half of the Phase 4 chain
    (insight -> supporting_reasoning_signal_ids) cannot be reached from an
    execution row. Closing this needs a column + a field, so it is a Phase 5.2
    decision, not a drive-by fix.
    """
    decision = {
        "decision_id": str(DECISION),
        "category": "waiting_reply",
        "title": "John is waiting for your reply",
        "evidence": {
            "person_email": "john@x.com", "thread_id": "t-1",
            "rfc_message_id": "<audit-1@mail.example>", "subject": "Contract",
        },
        "reasoning_signal_ids": ["s-1"],
        "reasoning_version": 1,
        "state_watermark": "wm",
        "supporting_insight_ids": ["insight-a", "insight-b"],
    }
    proposed = propose_from_decisions([decision], ORG_A)
    assert len(proposed) == 1, "precondition: this decision IS proposable"
    p = proposed[0]

    # Reasoning lineage survives...
    assert p.reasoning_signal_ids == ["s-1"]
    # ...Intelligence lineage does not. No field exists to carry it.
    assert not hasattr(p, "supporting_insight_ids")
    assert "supporting_insight_ids" not in p.payload


async def test_execution_request_has_nowhere_to_put_insight_provenance():
    """The gap is structural, not a missed assignment: neither the intake
    signature nor the persisted row has a slot for it."""
    import inspect

    from navis.execution.models import ExecutionRequest as _Row

    params = inspect.signature(ExecutionService.request).parameters
    assert "supporting_insight_ids" not in params
    assert not hasattr(_Row, "supporting_insight_ids")

    # And it does not sneak through on the outbox event either.
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        ORG_A, DECISION, ActionType.SCHEDULE_MEETING.value, {"title": "x", "start": "s"},
        reasoning_signal_ids=["s-1"], reasoning_version=1, state_watermark="wm",
    )
    await svc.approve(req.id, USER)
    for e in repo.events:
        assert "supporting_insight_ids" not in e.payload


# ---- PHASE 5.1 AUDIT: organization isolation on the HTTP surface ----


async def test_require_org_refuses_a_foreign_orgs_execution():
    """`_require_org` is the ONLY thing standing between an authenticated
    principal and another tenant's execution row: ExecutionService.approve /
    reject / cancel take a bare execution_id and never check the org.

    A foreign row must be indistinguishable from a missing one (404, not 403 —
    existence is itself tenant information)."""
    from fastapi import HTTPException

    from navis.execution.router import _require_org

    eid = uuid.uuid4()
    row = type("Row", (), {"id": eid, "organization_id": ORG_B})()

    class _DB:
        async def get(self, _model, _pk):
            return row

    assert await _require_org(_DB(), eid, ORG_B) is row  # own org: allowed

    with pytest.raises(HTTPException) as exc:
        await _require_org(_DB(), eid, ORG_A)  # foreign org
    assert exc.value.status_code == 404


def test_every_execution_id_route_is_org_scoped():
    """Structural guarantee: a future handler that forgets `_require_org` fails
    HERE rather than by leaking a tenant's row in production."""
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[2]
           / "navis" / "execution" / "router.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    checked = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        is_route = any(
            isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
            and isinstance(d.func.value, ast.Name) and d.func.value.id == "router"
            for d in node.decorator_list
        )
        args = {a.arg for a in node.args.args}
        if not is_route or "execution_id" not in args:
            continue
        called = {
            n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "_require_org" in called, (
            f"route {node.name!r} takes an execution_id but never org-scopes it"
        )
        checked.append(node.name)

    # Guards the guard: if the routes are renamed away, this test must not pass vacuously.
    assert len(checked) >= 6, f"expected the full /actions/{{id}} surface, saw {checked}"


# ---- PHASE 5.2.1: a simulation must never be recorded as a real execution ----
#
# `Provider.execute()` dry-runs when no sender is injected, returning ok=True
# with a synthetic `dry_run=True` reference. That is CORRECT for
# `_DEFAULT_PROVIDERS` — the tests above depend on it and it keeps the suite off
# Google. It was a silent misconfiguration in `production_providers()`, where
# `send_slack` was registered with no sender: an approved Slack action ran the
# dry-run branch, the worker recorded EXECUTED, and Observation captured a
# success for a message that was never sent.


def test_production_registry_never_serves_a_simulating_provider():
    """THE invariant, and the reason this fix generalises.

    Any provider in the production registry either reaches a real external
    system (`is_live`) or refuses outright (`unsupported`). Nothing in between —
    a provider that would return a synthetic dry-run reference cannot be served
    to the worker, so no future connector can be wired without its sender and
    quietly report fake successes.
    """
    from navis.execution.providers import production_providers

    for action_type, provider in production_providers().items():
        assert provider.is_live or provider.unsupported, (
            f"{action_type} would SIMULATE in production: no sender is wired and "
            f"{type(provider).__name__} does not refuse, so an approved action "
            "would be recorded as EXECUTED without happening"
        )


def test_unsupported_providers_are_still_served_in_production():
    """`create_task` has no sender and never will until a backend exists, but it
    raises at validate() rather than simulating. Keeping it is deliberate: it
    can explain itself, where a withheld type dead-ends on 'no provider'."""
    from navis.execution.providers import production_providers

    registry = production_providers()
    assert ActionType.CREATE_TASK.value in registry
    assert registry[ActionType.CREATE_TASK.value].unsupported is True


async def test_a_simulating_provider_cannot_reach_executed_in_production():
    """A provider with NO sender must never report success in production.

    The original reproduction used `send_slack`, which was wired into the
    production registry with no sender: it ran the dry-run branch, the worker
    recorded EXECUTED, and Observation captured a success for a message that was
    never sent. Slack now HAS a real sender (see providers/_senders.py), so it is
    no longer an example of the defect — the mechanism is therefore pinned with a
    constructed senderless provider, which keeps this invariant under test for
    every connector that ships after this one rather than until the next one does.
    """
    from navis.execution.providers import _production_safe
    from navis.execution.providers.base import Provider

    class _Unwired(Provider):
        name = "slack"
        action_type = "send_slack"

        def __init__(self):
            self._sender = None  # is_live -> False: would simulate

        async def validate(self, payload):
            pass

        async def execute(self, payload):  # pragma: no cover - must be unreachable
            raise AssertionError("a withheld provider must never execute")

        async def status(self, reference):
            return "sent"

    registry = _production_safe({"send_slack": _Unwired()})
    assert registry == {}, "a provider that would simulate must be withheld"

    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        ORG_A, DECISION, ActionType.SEND_SLACK.value,
        {"channel": "#general", "text": "hello"},
        reasoning_signal_ids=[], reasoning_version=1, state_watermark="wm",
    )
    await svc.approve(req.id, USER)

    result = await execute_pass(repo, providers=registry, policy=FAST)

    assert result == {"executed": 0, "failed": 1, "skipped": 0}
    assert req.status == ExecutionStatus.FAILED.value
    # The three things that made the old behaviour a lie.
    assert req.executed_at is None
    assert req.provider_reference == {}
    assert EventType.ACTION_EXECUTED.value not in event_types(repo)
    # Observation is told the truth instead.
    assert EventType.ACTION_FAILED.value in event_types(repo)
    # And it is preserved for an operator rather than silently dropped.
    assert [d.action_type for d in repo.dead_letters] == [ActionType.SEND_SLACK.value]


async def test_a_live_provider_still_executes_and_keeps_its_reference():
    """Guards against over-correction: the fix must not touch the real path.

    A wired provider still reaches EXECUTED and its genuine external id still
    survives onto the row and into the outcome Observation consumes.
    """
    async def sender(payload):
        return "real-gmail-id-123"

    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        ORG_A, DECISION, ActionType.SEND_EMAIL.value,
        {"to": "a@x.com", "subject": "Hi", "body": "Real body"},
        reasoning_signal_ids=[], reasoning_version=1, state_watermark="wm",
    )
    await svc.approve(req.id, USER)

    live = GmailProvider(sender=sender)
    assert live.is_live is True
    result = await execute_pass(
        repo, providers={ActionType.SEND_EMAIL.value: live}, policy=FAST
    )

    assert result["executed"] == 1
    assert req.status == ExecutionStatus.EXECUTED.value
    assert req.executed_at is not None
    assert req.provider_reference["external_id"] == "real-gmail-id-123"
    assert "dry_run" not in req.provider_reference
    executed = [e for e in repo.events if e.event_type == EventType.ACTION_EXECUTED.value]
    assert executed[0].payload["outcome"]["provider_reference"]["external_id"] == (
        "real-gmail-id-123"
    )
