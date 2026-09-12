"""The run telemetry contract — economics provenance, and the rollup arithmetic.

WHAT IS PINNED HERE, AND WHY IT NEEDS NO DATABASE. `RunTelemetryOut.of` is pure:
rows in, totals out. That is deliberate, and it is what lets the part most worth
being sure about — that an estimated cost is never added to a provider-reported
one — be tested on plain objects rather than through four layers of transport.

Schema parity between these ORM models and migration 0049 belongs in
`tests/integration/`, which needs a disposable Postgres and is skipped without
one. Nothing here starts a database.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from navis.developer.lifecycle import CostBasis, COST_BASES, StepStatus, StepType, ToolKind
from navis.developer.run_schemas import RecordStepRequest, RunTelemetryOut


_ORG = uuid.uuid4()
_AGENT = uuid.uuid4()
_TASK = uuid.uuid4()
_RUN = uuid.uuid4()
_T0 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)


class _Row:
    """A stand-in for an ORM row. The rollup reads attributes and nothing else."""

    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, value)


def _run(**overrides) -> _Row:
    base = dict(
        id=_RUN,
        organization_id=_ORG,
        task_id=_TASK,
        agent_id=_AGENT,
        status="completed",
        created_at=_T0,
        updated_at=_T0,
        started_at=_T0 + timedelta(milliseconds=500),
        completed_at=_T0 + timedelta(seconds=4),
        failure_category=None,
        failure_reason=None,
        context_snapshot_id="snap-1",
        context_version="ctx-v2",
        correlation_id=None,
        run_metadata={},
    )
    base.update(overrides)
    return _Row(**base)


def _step(sequence: int, step_type: str, **overrides) -> _Row:
    base = dict(
        id=uuid.uuid4(),
        organization_id=_ORG,
        run_id=_RUN,
        sequence=sequence,
        step_type=step_type,
        status=StepStatus.COMPLETED.value,
        label="",
        started_at=None,
        completed_at=None,
        tool_name=None,
        tool_kind=None,
        model_provider=None,
        model_name=None,
        model_generation_id=None,
        input_tokens=None,
        output_tokens=None,
        cached_tokens=None,
        cache_read_tokens=None,
        cache_write_tokens=None,
        reasoning_tokens=None,
        cost_micros=None,
        cost_currency=None,
        cost_basis=None,
        pricing_version=None,
        context_items=None,
        context_tokens=None,
        retrieval_count=None,
        latency_ms=None,
        error_category=None,
        error_message=None,
        created_at=_T0,
        request_metadata={},
        response_metadata={},
    )
    base.update(overrides)
    return _Row(**base)


def _telemetry(steps, *, run=None, outcome=None, attempt=1, task_attempts=1) -> _Row:
    return _Row(
        run=run if run is not None else _run(),
        task_description="reconcile the billing entity",
        attempt=attempt,
        task_attempts=task_attempts,
        steps=steps,
        outcome=outcome,
    )


# ── Cost provenance ─────────────────────────────────────────────────────────


def test_the_cost_basis_vocabulary_is_closed_and_matches_its_enum():
    assert COST_BASES == {b.value for b in CostBasis}


def test_a_cost_without_a_basis_is_refused_at_the_edge():
    """THE CONSTRAINT THIS DOMAIN EXISTS FOR. An unattributed currency figure is
    read as billing by everyone who sees it later."""
    with pytest.raises(ValidationError, match="cost_basis is required"):
        RecordStepRequest(
            step_type=StepType.MODEL_GENERATION,
            cost_micros=4120,
            cost_currency="USD",
        )


def test_a_basis_without_a_cost_is_allowed():
    """The converse is harmless: a caller saying how it WOULD cost something it
    has not costed asserts nothing false."""
    request = RecordStepRequest(
        step_type=StepType.MODEL_GENERATION,
        cost_basis=CostBasis.ESTIMATED,
        pricing_version="anthropic-2026-06-24",
    )
    assert request.cost_micros is None


def test_estimated_and_provider_reported_costs_are_never_added_together():
    """The whole point of `cost_basis`. One is arithmetic over a price list, the
    other is the provider's own figure, and their sum is neither."""
    rollup = RunTelemetryOut.of(
        _telemetry(
            [
                _step(
                    0,
                    StepType.MODEL_GENERATION.value,
                    model_provider="anthropic",
                    model_name="claude-opus-5",
                    cost_micros=1000,
                    cost_currency="USD",
                    cost_basis=CostBasis.ESTIMATED.value,
                    pricing_version="anthropic-2026-06-24",
                ),
                _step(
                    1,
                    StepType.MODEL_GENERATION.value,
                    model_provider="anthropic",
                    model_name="claude-opus-5",
                    cost_micros=250,
                    cost_currency="USD",
                    cost_basis=CostBasis.PROVIDER_REPORTED.value,
                ),
            ]
        )
    )
    assert rollup.economics.estimated_cost_micros == 1000
    assert rollup.economics.provider_reported_cost_micros == 250
    assert rollup.economics.pricing_versions == ["anthropic-2026-06-24"]


def test_an_absent_basis_totals_as_nothing_rather_than_as_zero():
    """A run whose costs were never recorded must not report a $0.00 cost — that
    is a claim, and the wrong one."""
    rollup = RunTelemetryOut.of(
        _telemetry(
            [
                _step(
                    0,
                    StepType.MODEL_GENERATION.value,
                    model_provider="anthropic",
                    model_name="claude-opus-5",
                    input_tokens=100,
                )
            ]
        )
    )
    assert rollup.economics.estimated_cost_micros is None
    assert rollup.economics.provider_reported_cost_micros is None
    assert rollup.economics.uncosted_generations == 1


def test_mixed_currencies_refuse_to_total():
    """Adding micro-USD to micro-EUR produces a number with no meaning, and
    nothing downstream could tell it from a real one."""
    rollup = RunTelemetryOut.of(
        _telemetry(
            [
                _step(
                    0,
                    StepType.MODEL_GENERATION.value,
                    cost_micros=1000,
                    cost_currency="USD",
                    cost_basis=CostBasis.ESTIMATED.value,
                ),
                _step(
                    1,
                    StepType.MODEL_GENERATION.value,
                    cost_micros=900,
                    cost_currency="EUR",
                    cost_basis=CostBasis.ESTIMATED.value,
                ),
            ]
        )
    )
    assert rollup.economics.mixed_currency is True
    assert rollup.economics.currency is None
    assert rollup.economics.estimated_cost_micros is None


# ── Cache accounting ────────────────────────────────────────────────────────


def test_cache_reads_and_writes_stay_separate_through_the_rollup():
    """They differ by more than twelve times in price. A total cannot be priced,
    which is why the split exists at all."""
    rollup = RunTelemetryOut.of(
        _telemetry(
            [
                _step(
                    0,
                    StepType.MODEL_GENERATION.value,
                    model_provider="anthropic",
                    model_name="claude-opus-5",
                    input_tokens=500,
                    output_tokens=200,
                    cache_read_tokens=8000,
                    cache_write_tokens=1200,
                    reasoning_tokens=150,
                )
            ]
        )
    )
    usage = rollup.models[0]
    assert usage.cache_read_tokens == 8000
    assert usage.cache_write_tokens == 1200
    assert usage.reasoning_tokens == 150
    # The legacy total is not synthesized from the pair: a caller that did not
    # send one has not reported one.
    assert usage.cached_tokens == 0


def test_generations_group_by_provider_and_model():
    rollup = RunTelemetryOut.of(
        _telemetry(
            [
                _step(
                    0,
                    StepType.MODEL_GENERATION.value,
                    model_provider="anthropic",
                    model_name="claude-opus-5",
                    input_tokens=10,
                ),
                _step(
                    1,
                    StepType.MODEL_GENERATION.value,
                    model_provider="anthropic",
                    model_name="claude-opus-5",
                    input_tokens=15,
                ),
                _step(
                    2,
                    StepType.MODEL_GENERATION.value,
                    model_provider="anthropic",
                    model_name="claude-haiku-4-5",
                    input_tokens=7,
                ),
            ]
        )
    )
    by_model = {m.model: m for m in rollup.models}
    assert by_model["claude-opus-5"].generations == 2
    assert by_model["claude-opus-5"].input_tokens == 25
    assert by_model["claude-haiku-4-5"].generations == 1


# ── Context, tools, latency ─────────────────────────────────────────────────


def test_context_is_summed_across_fetches_and_the_version_comes_from_the_run():
    """Context is the independent variable of the comparison. A run may fetch
    repeatedly; the version qualifies the snapshot and does not vary within one
    attempt."""
    rollup = RunTelemetryOut.of(
        _telemetry(
            [
                _step(
                    0,
                    StepType.CONTEXT_FETCH.value,
                    context_items=4,
                    context_tokens=1200,
                    retrieval_count=9,
                    latency_ms=80,
                ),
                _step(
                    1,
                    StepType.CONTEXT_FETCH.value,
                    context_items=2,
                    context_tokens=300,
                    retrieval_count=3,
                    latency_ms=40,
                ),
            ]
        )
    )
    assert rollup.context.fetches == 2
    assert rollup.context.items == 6
    assert rollup.context.tokens == 1500
    assert rollup.context.retrievals == 12
    assert rollup.context.version == "ctx-v2"
    assert rollup.context.snapshot_id == "snap-1"
    assert rollup.latency.context_ms == 120


def test_tool_failures_are_counted_apart_from_calls():
    rollup = RunTelemetryOut.of(
        _telemetry(
            [
                _step(
                    0,
                    StepType.TOOL_CALL.value,
                    tool_name="github.search",
                    tool_kind=ToolKind.REST.value,
                    latency_ms=120,
                ),
                _step(
                    1,
                    StepType.TOOL_CALL.value,
                    tool_name="github.search",
                    tool_kind=ToolKind.REST.value,
                    status=StepStatus.FAILED.value,
                    latency_ms=30,
                ),
            ]
        )
    )
    tool = rollup.tools[0]
    assert tool.calls == 2
    assert tool.failures == 1
    assert tool.latency_ms == 150


def test_queue_time_is_separated_from_run_time():
    """Wait before work began is latency the agent did not cause."""
    rollup = RunTelemetryOut.of(_telemetry([]))
    assert rollup.latency.queue_ms == 500
    assert rollup.latency.total_ms == 3500


def test_a_backwards_clock_clamps_instead_of_reporting_a_negative_duration():
    """These timestamps come from an external agent's clock. A negative span is
    skew, and it would corrupt every average built on it."""
    run = _run(
        started_at=_T0 + timedelta(seconds=4),
        completed_at=_T0 + timedelta(seconds=1),
    )
    rollup = RunTelemetryOut.of(_telemetry([], run=run))
    assert rollup.latency.total_ms == 0


def test_an_unfinished_run_reports_no_total_latency():
    run = _run(status="running", completed_at=None)
    rollup = RunTelemetryOut.of(_telemetry([], run=run))
    assert rollup.latency.total_ms is None


# ── Outcome and attempts ────────────────────────────────────────────────────


def test_an_unjudged_run_reports_no_outcome_rather_than_a_default_one():
    """`None` must stay distinguishable from a judged failure — nothing
    evaluates agent outcomes automatically yet."""
    assert RunTelemetryOut.of(_telemetry([])).outcome is None


def test_retries_are_reported_as_attempts_at_the_task():
    """A retry is another RUN of the same task, which the ledger already holds
    structurally. A counter would be a second source free to disagree."""
    rollup = RunTelemetryOut.of(_telemetry([], attempt=3, task_attempts=4))
    assert rollup.attempt == 3
    assert rollup.task_attempts == 4


# ── The security properties, restated on the new fields ─────────────────────


def test_no_telemetry_request_field_accepts_an_identity():
    """The tenant is resolved from the credential in every handler. A new
    telemetry field must not become the exception."""
    forbidden = {"organization_id", "org_id", "tenant_id", "user_id", "agent_id"}
    assert forbidden.isdisjoint(RecordStepRequest.model_fields)


def test_the_new_step_fields_are_counts_and_labels_only():
    """Telemetry records HOW MUCH context was supplied, never what it said. A
    field that could carry organizational content would make this table a second
    copy of the organization's state."""
    request = RecordStepRequest(
        step_type=StepType.CONTEXT_FETCH,
        context_items=3,
        context_tokens=900,
        retrieval_count=5,
    )
    assert request.context_items == 3
    # There is no field on the contract for the content itself.
    assert not {"context", "context_body", "entities", "documents"} & set(
        RecordStepRequest.model_fields
    )
