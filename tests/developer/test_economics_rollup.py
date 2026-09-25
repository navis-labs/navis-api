"""The Agent Economics rollup, on plain objects.

WHY THESE RUN WITHOUT A DATABASE. The arithmetic that decides whether context
was worth its tokens is the thing most worth being sure about, and a test that
needs Postgres to check a division is a test nobody runs while changing the
division. `EconomicsRollupOut.of` is pure for exactly this reason — the same
discipline `RunTelemetryOut.of` already follows.

Tenant isolation, the join and the SQL reduction are NOT tested here; they are
properties of the query and are proved in
`tests/integration/test_developer_economics_postgres.py` against real
PostgreSQL.
"""
from __future__ import annotations

import uuid

from navis.developer.run_schemas import EconomicsRollupOut
from navis.developer.run_service import RunEconomics

TASK_A = uuid.UUID("aaaaaaaa-0000-4000-8000-00000000000a")
TASK_B = uuid.UUID("bbbbbbbb-0000-4000-8000-00000000000b")
AGENT = uuid.UUID("cccccccc-0000-4000-8000-00000000000c")


def run(
    *,
    task_id: uuid.UUID = TASK_A,
    fetched: bool = True,
    version: str | None = "v1",
    estimated: int | None = None,
    reported: int | None = None,
    uncosted: int = 0,
    currency: str = "USD",
    outcome: str | None = "success",
    quality: float | None = None,
    corrections: int | None = None,
    latency: int | None = None,
    model: str = "claude-opus-5",
    pricing: str = "2026-09-01",
) -> RunEconomics:
    return RunEconomics(
        task_id=task_id,
        task_description="ship the thing",
        agent_id=AGENT,
        context_fetched=fetched,
        context_version=version,
        estimated_micros=estimated,
        reported_micros=reported,
        uncosted_generations=uncosted,
        currencies=frozenset({currency}) if (estimated or reported) else frozenset(),
        pricing_versions=frozenset({pricing}),
        outcome_status=outcome,
        quality_score=quality,
        human_corrections=corrections,
        latency_ms=latency,
        models=frozenset({model}),
    )


def only(rollup: EconomicsRollupOut):
    assert len(rollup.groups) == 1, f"expected one group, got {len(rollup.groups)}"
    return rollup.groups[0]


# ── The headline metric ─────────────────────────────────────────────────────


def test_one_successful_run_costs_what_it_cost():
    g = only(EconomicsRollupOut.of([run(estimated=1_000)]))
    assert g.runs == 1
    assert g.successful_outcomes == 1
    assert g.success_rate == 1.0
    assert g.expected_estimated_cost_per_success_micros == 1_000


def test_failed_attempts_are_charged_to_the_success_they_bought():
    """THE WHOLE POINT OF THE METRIC. Three failures then a success cost 4,000
    to produce one success, not 1,000 — an agent that retries its way to an
    answer is not as cheap as one that gets it first time, and cost-per-run
    would report exactly that falsehood."""
    rows = [
        run(estimated=1_000, outcome="failure"),
        run(estimated=1_000, outcome="failure"),
        run(estimated=1_000, outcome="failure"),
        run(estimated=1_000, outcome="success"),
    ]
    g = only(EconomicsRollupOut.of(rows))
    assert g.runs == 4
    assert g.successful_outcomes == 1
    assert g.estimated_cost_micros == 4_000
    assert g.expected_estimated_cost_per_success_micros == 4_000
    assert g.success_rate == 0.25


def test_no_successes_reports_null_rather_than_the_raw_total():
    """An infinite cost per success is not a number. Reporting the total there
    would read as though the spend had bought something."""
    g = only(EconomicsRollupOut.of([run(estimated=500, outcome="failure")] * 3))
    assert g.successful_outcomes == 0
    assert g.estimated_cost_micros == 1_500
    assert g.expected_estimated_cost_per_success_micros is None
    assert g.success_rate == 0.0


def test_an_empty_dataset_is_an_empty_rollup_not_a_zero():
    rollup = EconomicsRollupOut.of([])
    assert rollup.groups == []
    assert rollup.count == 0
    assert rollup.runs_considered == 0


# ── Outcome correctness ─────────────────────────────────────────────────────


def test_unknown_outcomes_are_not_failures_and_do_not_enter_the_rate():
    """`unknown` is the default and the common case: nothing judged the run.
    Counting it as failure would invent a verdict and poison the first metric
    anyone builds on this."""
    rows = [
        run(estimated=100, outcome="success"),
        run(estimated=100, outcome=None),
        run(estimated=100, outcome="unknown"),
    ]
    g = only(EconomicsRollupOut.of(rows))
    assert g.successful_outcomes == 1
    assert g.failed_outcomes == 0
    assert g.unknown_outcomes == 2
    # Judged population is one run, and it succeeded.
    assert g.success_rate == 1.0


def test_nothing_judged_means_no_success_rate_at_all():
    g = only(EconomicsRollupOut.of([run(estimated=100, outcome=None)] * 4))
    assert g.unknown_outcomes == 4
    assert g.success_rate is None, "no evidence must not read as a 0% success rate"


def test_partial_is_neither_success_nor_failure():
    g = only(EconomicsRollupOut.of([run(outcome="partial"), run(outcome="success")]))
    assert g.partial_outcomes == 1
    assert g.successful_outcomes == 1
    assert g.success_rate == 0.5


# ── Cost correctness ────────────────────────────────────────────────────────


def test_estimated_and_provider_reported_are_never_added_together():
    """The rule `EconomicsOut` exists to enforce, held across runs too: one is
    arithmetic over a price list, the other is the provider's own figure, and
    their sum is neither."""
    rows = [run(estimated=1_000), run(reported=7)]
    g = only(EconomicsRollupOut.of(rows))
    assert g.estimated_cost_micros == 1_000
    assert g.provider_reported_cost_micros == 7
    assert g.expected_estimated_cost_per_success_micros == 500
    assert g.expected_reported_cost_per_success_micros == 3  # 7 // 2


def test_mixed_currencies_withhold_the_total_rather_than_adding_them():
    rows = [run(estimated=1_000, currency="USD"), run(estimated=1_000, currency="EUR")]
    g = only(EconomicsRollupOut.of(rows))
    assert g.mixed_currency is True
    assert g.currency is None
    assert g.estimated_cost_micros is None
    assert g.expected_estimated_cost_per_success_micros is None
    assert g.p50_run_cost_micros is None


def test_uncosted_generations_are_reported_not_absorbed():
    """A model step with no cost figure is missing evidence. The total over the
    others is incomplete, and the reader has to be able to see that."""
    rows = [run(estimated=1_000, uncosted=2), run(estimated=None, uncosted=1)]
    g = only(EconomicsRollupOut.of(rows))
    assert g.uncosted_generations == 3
    assert g.runs_with_no_cost_recorded == 1
    assert g.estimated_cost_micros == 1_000, "the costed run still counts"


def test_a_run_with_no_cost_is_not_a_free_run():
    g = only(EconomicsRollupOut.of([run(estimated=None, reported=None)]))
    assert g.estimated_cost_micros is None
    assert g.provider_reported_cost_micros is None
    assert g.runs_with_no_cost_recorded == 1


def test_percentiles_report_an_observed_cost_never_an_interpolated_one():
    rows = [run(estimated=c) for c in (100, 200, 300, 400, 1_000)]
    g = only(EconomicsRollupOut.of(rows))
    assert g.p50_run_cost_micros == 300
    assert g.p90_run_cost_micros == 1_000
    assert g.p50_run_cost_micros in {100, 200, 300, 400, 1_000}


# ── Grouping ────────────────────────────────────────────────────────────────


def test_context_fetched_and_not_fetched_are_separate_populations():
    """The comparison the rollup exists for."""
    rows = [
        run(fetched=True, estimated=1_000, outcome="success"),
        run(fetched=True, estimated=1_000, outcome="success"),
        run(fetched=False, estimated=1_000, outcome="failure"),
        run(fetched=False, estimated=1_000, outcome="success"),
    ]
    rollup = EconomicsRollupOut.of(rows)
    assert len(rollup.groups) == 2
    with_ctx = next(g for g in rollup.groups if g.context_fetched)
    without = next(g for g in rollup.groups if not g.context_fetched)
    assert with_ctx.success_rate == 1.0
    assert without.success_rate == 0.5
    assert with_ctx.expected_estimated_cost_per_success_micros == 1_000
    assert without.expected_estimated_cost_per_success_micros == 2_000


def test_context_versions_do_not_pool():
    rows = [run(version="v1", estimated=100), run(version="v2", estimated=900)]
    rollup = EconomicsRollupOut.of(rows)
    assert {g.context_version for g in rollup.groups} == {"v1", "v2"}
    assert all(g.runs == 1 for g in rollup.groups)


def test_tasks_do_not_pool():
    rollup = EconomicsRollupOut.of([run(task_id=TASK_A), run(task_id=TASK_B)])
    assert {g.task_id for g in rollup.groups} == {TASK_A, TASK_B}


def test_models_are_reported_but_do_not_split_the_group():
    """Model is a FILTER, not a grouping key: adding it to the key produces
    populations of one, where a success rate is 0% or 100% and means neither."""
    rows = [run(model="claude-opus-5", estimated=100), run(model="gpt-5", estimated=100)]
    g = only(EconomicsRollupOut.of(rows))
    assert g.runs == 2
    assert g.models == ["claude-opus-5", "gpt-5"]


# ── Presentation guarantees ─────────────────────────────────────────────────


def test_quality_averages_only_over_runs_that_were_scored():
    rows = [run(quality=1.0), run(quality=0.5), run(quality=None)]
    g = only(EconomicsRollupOut.of(rows))
    assert g.mean_quality_score == 0.75, "an unscored run must not read as a zero"


def test_human_corrections_accumulate():
    g = only(EconomicsRollupOut.of([run(corrections=2), run(corrections=1), run()]))
    assert g.human_corrections == 3


def test_latency_is_reported_separately_from_cost():
    rows = [run(estimated=100, latency=1_000), run(estimated=100, latency=9_000)]
    g = only(EconomicsRollupOut.of(rows))
    assert g.p50_latency_ms == 1_000
    assert g.p90_latency_ms == 9_000
    assert g.estimated_cost_micros == 200, "latency must not leak into cost"


def test_the_result_carries_its_own_observational_caveat():
    """The response can be stored, forwarded and read by somebody who did not
    issue the request, so the caveat travels with the numbers."""
    rollup = EconomicsRollupOut.of([run()])
    assert "correlation" in rollup.interpretation.lower()


def test_ordering_is_deterministic():
    rows = [run(task_id=TASK_B), run(task_id=TASK_A), run(task_id=TASK_A)]
    first = EconomicsRollupOut.of(rows)
    second = EconomicsRollupOut.of(list(reversed(rows)))
    assert [g.task_id for g in first.groups] == [g.task_id for g in second.groups]
    assert first.groups[0].runs == 2, "largest population first"


def test_truncation_is_surfaced():
    rollup = EconomicsRollupOut.of([run()], truncated=True)
    assert rollup.runs_truncated is True
