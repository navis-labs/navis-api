"""Wire contracts for the agent execution lifecycle.

NO MODEL BELOW CARRIES AN IDENTITY INPUT. There is no `organization_id` on any
REQUEST, and none may be added: the tenant is `DeveloperContext.organization_id`
in every handler, resolved from the credential. The `organization_id` on the
response models is an echo of what the server resolved, present so a client can
assert it is talking to the tenant it thinks it is.

`agent_id` IS NOT A REQUEST FIELD EITHER. It is a PATH parameter
(`/developer/agents/{agent_id}/tasks`), and a run takes its agent from its task
rather than from anything a caller sends. The architecture guard
`test_no_developer_route_accepts_an_identity_parameter` enforces exactly this —
an identity name may address a resource in a path, never arrive alongside one in
a body, because that is how "filter" quietly becomes "scope".

NO SECRET IS REPRESENTABLE. Nothing here has a secret field, and the metadata
objects are scrubbed by `sanitize.py` before they reach a column and are echoed
back only as stored — so a redacted value stays redacted on the way out.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from navis.developer.lifecycle import (
    MAX_STEP_LABEL,
    MAX_TASK_DESCRIPTION,
    MAX_TOOL_NAME,
    CostBasis,
    ErrorCategory,
    OutcomeStatus,
    RunStatus,
    StepStatus,
    StepType,
    TaskStatus,
    ToolKind,
)

__all__ = [
    "CloseTaskRequest",
    "ContextUsageOut",
    "CreateTaskRequest",
    "EconomicsOut",
    "FinishRunRequest",
    "LatencyOut",
    "ModelUsageOut",
    "OutcomeOut",
    "RecordOutcomeRequest",
    "RecordStepRequest",
    "RunOut",
    "RunPageOut",
    "RunTelemetryOut",
    "StartRunRequest",
    "StepOut",
    "StepPageOut",
    "TaskOut",
    "TaskPageOut",
    "ToolUsageOut",
]


def _span_ms(start: datetime | None, end: datetime | None) -> int | None:
    """Milliseconds between two instants, or None if either is missing.

    NEGATIVE SPANS ARE CLAMPED TO ZERO, not reported. These timestamps come from
    an external agent's clock, and a run whose `completed_at` precedes its
    `started_at` is a clock skew rather than a negative duration — emitting -240ms
    would corrupt any average built on it, and there is no honest positive value
    to invent instead.
    """
    if start is None or end is None:
        return None
    return max(0, int((end - start).total_seconds() * 1000))


# ── Tasks ───────────────────────────────────────────────────────────────────


class CreateTaskRequest(BaseModel):
    """Record what an agent was asked to do."""

    description: str = Field(
        min_length=1,
        max_length=MAX_TASK_DESCRIPTION,
        description="What was asked, in the operator's words.",
    )
    idempotency_key: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Retry-safe creation. The same key in the same organization returns "
            "the EXISTING task rather than creating a second one."
        ),
    )
    metadata: dict | None = Field(
        default=None,
        description="Bounded, scrubbed free-form facts. Secrets are redacted before storage.",
    )


class TaskOut(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    agent_id: uuid.UUID
    description: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def of(cls, row) -> "TaskOut":
        return cls(
            id=row.id,
            organization_id=row.organization_id,
            agent_id=row.agent_id,
            description=row.description,
            status=row.status,
            created_at=row.created_at,
            updated_at=row.updated_at,
            metadata=row.task_metadata or {},
        )


class CloseTaskRequest(BaseModel):
    status: TaskStatus = Field(description="`completed` or `abandoned`. `open` is not a close.")


# ── Runs ────────────────────────────────────────────────────────────────────


class StartRunRequest(BaseModel):
    """Open an attempt. The task is a PATH parameter and the agent comes from
    the task, so neither appears here — a run can never belong to a different
    agent than the work it attempts."""

    context_snapshot_id: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "The context snapshot this run worked from. A REFERENCE — the "
            "projection is not copied into the run."
        ),
    )
    context_version: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Which version of context assembly produced that snapshot. Grouping "
            "runs by this is how a context change is compared against a baseline."
        ),
    )
    correlation_id: str | None = Field(
        default=None,
        max_length=128,
        description="Correlates this run with the caller's own logs.",
    )
    idempotency_key: str | None = Field(default=None, max_length=128)
    metadata: dict | None = None


class FinishRunRequest(BaseModel):
    status: RunStatus = Field(description="`completed`, `failed` or `cancelled`.")
    failure_category: ErrorCategory | None = None
    failure_reason: str | None = Field(default=None, max_length=500)


class RunOut(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    task_id: uuid.UUID
    agent_id: uuid.UUID
    status: RunStatus
    started_at: datetime | None
    completed_at: datetime | None
    failure_category: ErrorCategory | None
    failure_reason: str | None
    context_snapshot_id: str | None
    context_version: str | None
    correlation_id: str | None
    created_at: datetime
    updated_at: datetime
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def of(cls, row) -> "RunOut":
        return cls(
            id=row.id,
            organization_id=row.organization_id,
            task_id=row.task_id,
            agent_id=row.agent_id,
            status=row.status,
            started_at=row.started_at,
            completed_at=row.completed_at,
            failure_category=row.failure_category,
            failure_reason=row.failure_reason,
            context_snapshot_id=row.context_snapshot_id,
            context_version=row.context_version,
            correlation_id=row.correlation_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            metadata=row.run_metadata or {},
        )


# ── Steps ───────────────────────────────────────────────────────────────────


class RecordStepRequest(BaseModel):
    """One event in a run's timeline.

    THE SEQUENCE IS NOT A FIELD. The server assigns it, so a caller cannot
    interleave itself into another writer's ordering or collide on the unique
    constraint.
    """

    step_type: StepType
    status: StepStatus = StepStatus.COMPLETED
    label: str = Field(default="", max_length=MAX_STEP_LABEL)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    tool_name: str | None = Field(default=None, max_length=MAX_TOOL_NAME)
    tool_kind: ToolKind | None = None

    # Free-form strings, not enums: Navis does not own the model, and a closed
    # set here would need a migration every time a vendor ships a version.
    model_provider: str | None = Field(default=None, max_length=64)
    model_name: str | None = Field(default=None, max_length=128)
    model_generation_id: str | None = Field(default=None, max_length=128)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Legacy TOTAL cached tokens. Send it if your provider reports only a "
            "total; prefer the split pair below, which is what a cost estimate "
            "can actually be computed from."
        ),
    )
    cache_read_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Tokens served FROM cache — billed at a large discount (~0.1x on Anthropic).",
    )
    cache_write_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Tokens written TO cache — billed at a PREMIUM (~1.25x on Anthropic).",
    )
    reasoning_tokens: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Thinking tokens where the provider exposes them. Usually already "
            "counted inside output_tokens — reported separately, not added."
        ),
    )
    cost_micros: int | None = Field(
        default=None, ge=0, description="Millionths of a currency unit. Integer, never a float."
    )
    cost_currency: str | None = Field(default=None, min_length=3, max_length=3)
    cost_basis: CostBasis | None = Field(
        default=None,
        description=(
            "REQUIRED whenever cost_micros is sent. `estimated` means usage was "
            "multiplied by a price list; `provider_reported` means the provider "
            "returned the figure. Neither is an invoice."
        ),
    )
    pricing_version: str | None = Field(
        default=None,
        max_length=64,
        description="Which price list an estimate used, e.g. `anthropic-2026-06-24`.",
    )

    context_items: int | None = Field(
        default=None, ge=0, description="How many context items this fetch supplied."
    )
    context_tokens: int | None = Field(
        default=None, ge=0, description="How many tokens of context this fetch supplied."
    )
    retrieval_count: int | None = Field(
        default=None, ge=0, description="How many retrievals backed this fetch."
    )

    latency_ms: int | None = Field(default=None, ge=0)

    error_category: ErrorCategory | None = None
    error_message: str | None = Field(default=None, max_length=500)

    request_metadata: dict | None = None
    response_metadata: dict | None = None

    @model_validator(mode="after")
    def _cost_must_state_its_basis(self) -> "RecordStepRequest":
        """A currency figure without a basis is refused, at the edge and in the DB.

        Mirrors `ck_agent_run_steps_cost_has_basis`. Validated in both places for
        the reason every other vocabulary here is: the edge gives the caller a
        message it can act on, and the constraint holds for the writers that are
        not this edge. An unattributed cost is the one value this domain must not
        be able to store, because a number nobody can source gets read as billing.
        """
        if self.cost_micros is not None and self.cost_basis is None:
            raise ValueError(
                "cost_basis is required when cost_micros is supplied — say whether "
                "the figure is `estimated` or `provider_reported`"
            )
        return self


class StepOut(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    run_id: uuid.UUID
    sequence: int
    step_type: StepType
    status: StepStatus
    label: str
    started_at: datetime | None
    completed_at: datetime | None
    tool_name: str | None
    tool_kind: ToolKind | None
    model_provider: str | None
    model_name: str | None
    model_generation_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None
    cost_micros: int | None
    cost_currency: str | None
    cost_basis: CostBasis | None
    pricing_version: str | None
    context_items: int | None
    context_tokens: int | None
    retrieval_count: int | None
    latency_ms: int | None
    error_category: ErrorCategory | None
    error_message: str | None
    created_at: datetime
    request_metadata: dict = Field(default_factory=dict)
    response_metadata: dict = Field(default_factory=dict)

    @classmethod
    def of(cls, row) -> "StepOut":
        return cls(
            id=row.id,
            organization_id=row.organization_id,
            run_id=row.run_id,
            sequence=row.sequence,
            step_type=row.step_type,
            status=row.status,
            label=row.label,
            started_at=row.started_at,
            completed_at=row.completed_at,
            tool_name=row.tool_name,
            tool_kind=row.tool_kind,
            model_provider=row.model_provider,
            model_name=row.model_name,
            model_generation_id=row.model_generation_id,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            cached_tokens=row.cached_tokens,
            cache_read_tokens=row.cache_read_tokens,
            cache_write_tokens=row.cache_write_tokens,
            reasoning_tokens=row.reasoning_tokens,
            cost_micros=row.cost_micros,
            cost_currency=row.cost_currency,
            cost_basis=row.cost_basis,
            pricing_version=row.pricing_version,
            context_items=row.context_items,
            context_tokens=row.context_tokens,
            retrieval_count=row.retrieval_count,
            latency_ms=row.latency_ms,
            error_category=row.error_category,
            error_message=row.error_message,
            created_at=row.created_at,
            request_metadata=row.request_metadata or {},
            response_metadata=row.response_metadata or {},
        )


# ── Outcome ─────────────────────────────────────────────────────────────────


class RecordOutcomeRequest(BaseModel):
    status: OutcomeStatus = Field(
        description=(
            "Whether the RESULT was any good — not whether the run finished. "
            "`unknown` is the honest default until something evaluates it."
        )
    )
    summary: str = Field(default="", max_length=4000)
    result_reference: str | None = Field(
        default=None,
        max_length=500,
        description="Where the result actually is — a URL, ticket id or commit sha.",
    )
    quality_score: float | None = Field(default=None, ge=0, le=1)
    human_corrections: int | None = Field(
        default=None,
        ge=0,
        description=(
            "How many times a human had to fix this result. Omit it if nobody "
            "has reviewed the work — `0` asserts a review found nothing to fix, "
            "and defaulting it would make an unreviewed backlog read as flawless."
        ),
    )
    produced_at: datetime | None = None
    metadata: dict | None = None
    if_newer: bool = Field(
        default=False,
        description=(
            "Refuse this write with 409 when the stored outcome already carries a "
            "`produced_at` at or after the one submitted. A conditional write for "
            "callers that publish a monotonic series of observations about the same "
            "run, so a retry or a delayed request cannot replace a newer statement "
            "with an older one. Ignored when either side has no `produced_at`, since "
            "a write that does not say when it was observed makes no claim to "
            "compare. Default `false`: an evaluator or a backfill revising a verdict "
            "long after the fact is not a stale writer and must not be refused."
        ),
    )


class OutcomeOut(BaseModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    run_id: uuid.UUID
    agent_id: uuid.UUID
    status: OutcomeStatus
    summary: str
    result_reference: str | None
    quality_score: float | None
    human_corrections: int | None
    produced_at: datetime | None
    created_at: datetime
    updated_at: datetime
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def of(cls, row) -> "OutcomeOut":
        return cls(
            id=row.id,
            organization_id=row.organization_id,
            run_id=row.run_id,
            agent_id=row.agent_id,
            status=row.status,
            summary=row.summary,
            result_reference=row.result_reference,
            quality_score=row.quality_score,
            human_corrections=row.human_corrections,
            produced_at=row.produced_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
            metadata=row.outcome_metadata or {},
        )


class LatestOutcomeOut(BaseModel):
    """The newest outcome of one task, with the attempt that produced it.

    BOTH HALVES, because either alone is unusable. The outcome carries the
    content and the reality clock; the run carries `correlation_id`, which is
    how a caller knows WHICH session wrote it — the provenance that makes two
    concurrent sessions distinguishable rather than merely ordered.

    Composed from the existing `OutcomeOut` and `RunOut` rather than flattened
    into a third shape: a client that already renders a run or an outcome
    renders these unchanged, and no field means something different here than
    it does at `/runs/{id}` or `/runs/{id}/outcome`.
    """

    outcome: OutcomeOut
    run: RunOut

    @classmethod
    def of(cls, outcome, run) -> "LatestOutcomeOut":
        return cls(outcome=OutcomeOut.of(outcome), run=RunOut.of(run))


# ── Telemetry rollup ────────────────────────────────────────────────────────
#
# WHY A ROLLUP EXISTS AT ALL, when `/runs/{id}`, `/runs/{id}/steps` and
# `/runs/{id}/outcome` already return every underlying fact.
#
# Because the aggregation is not trivial and there are four clients. "What did
# this run cost, and was it worth it" means: sum tokens across model_generation
# steps, sum context across context_fetch steps, group tool calls by name, count
# attempts on the parent task, and — the part that is easy to get wrong — keep
# estimated cost apart from provider-reported cost instead of adding them. Done
# client-side, that arithmetic is written four times (REST consumer, TypeScript
# SDK, Python SDK, MCP) and drifts four ways, which is precisely the divergent
# logic this platform forbids. Done here, there is one implementation and every
# surface reports the same number.
#
# It adds NO new facts and NO new authorization. Every field below is derived
# from rows the caller could already read with the same permission.


class ModelUsageOut(BaseModel):
    """One provider/model pair's share of a run, summed over its generations.

    EVERY TOKEN FIGURE IS NULLABLE, AND THAT IS THE WHOLE CONTRACT. `null` means
    no generation of this model reported that measurement; `0` means one did and
    the count really was zero. They are different facts and this object refuses
    to merge them.

    It used to. Each field was a non-nullable `int` summed with `or 0`, so a
    generation whose usage nobody recorded contributed a silent zero — and the
    row still counted in `generations`. A reader got `generations: 1,
    input_tokens: 0`, which is indistinguishable from a model that genuinely
    consumed nothing, and there was no field anywhere that said otherwise. The
    step rows underneath reported `null` correctly the whole time, so the two
    routes over one row disagreed about whether a measurement existed.

    THE SEMANTICS MIRROR `EconomicsOut`, deliberately, because this is the same
    question about a different column:

        null            nothing reported it — the sum would be a fabrication
        a number        the sum over the generations that DID report it
        untokened_generations   how many did not, so a partial total is readable
                                as partial rather than as complete

    A PARTIAL SUM IS STILL PUBLISHED. If two of three generations reported
    input tokens, `input_tokens` is the sum of those two — a real measurement of
    a real subset — and `untokened_generations` is 1. Withholding it entirely
    would lose a figure that was honestly obtained; publishing it without the
    counter would present two thirds of a run as all of it.

    THE FIELDS ARE INDEPENDENT. A provider may report input and output but not
    cache or reasoning tokens, so each is decided on its own evidence rather
    than on whether the generation reported anything at all.
    """

    provider: str | None
    model: str | None
    #: How many generations of this model the run recorded. A COUNT of rows,
    #: always known — never null, and never affected by what they measured.
    generations: int
    #: Generations above that reported NEITHER input nor output tokens.
    #:
    #: The token counterpart of `EconomicsOut.uncosted_generations`, and scoped
    #: per model rather than per run because that is where the sums it qualifies
    #: live: "the run has three unmeasured generations" cannot say which model
    #: they belonged to, and the answer changes what a reader does about it.
    #:
    #: Cache and reasoning tokens are deliberately NOT part of this test. Most
    #: providers never return them, so requiring all six would make every
    #: ordinary generation look unmeasured and the counter would mean nothing.
    untokened_generations: int = 0
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None
    #: Wall time across this model's generations. NOT a token measurement and
    #: not covered by the rule above — it is still summed with `or 0`, which is
    #: its own unknown-versus-zero question and a separate change.
    latency_ms: int


class ToolUsageOut(BaseModel):
    """One tool's share of a run. Names only — never arguments or results."""

    tool_name: str | None
    tool_kind: ToolKind | None
    calls: int
    failures: int
    latency_ms: int


class ContextUsageOut(BaseModel):
    """What organizational context the run was given, as counts.

    THE INDEPENDENT VARIABLE of a baseline-versus-context comparison. Metadata
    only: how much was supplied, never what it said.
    """

    version: str | None
    snapshot_id: str | None
    fetches: int
    items: int
    tokens: int
    retrievals: int


class EconomicsOut(BaseModel):
    """What the run cost — with the provenance kept intact.

    THE TWO TOTALS ARE NEVER ADDED TOGETHER, and that is the entire reason this
    object is shaped this way. `estimated_cost_micros` is arithmetic over a price
    list; `provider_reported_cost_micros` is the provider's own figure. Summing
    them would produce a number that is neither, and presenting either as billing
    would be a claim Navis cannot support — a provider API returns token counts,
    not money.

    `uncosted_generations` is reported rather than hidden: if half the model
    steps carried no cost, a total computed from the other half is not the run's
    cost, and a reader has to be able to see that.
    """

    currency: str | None = Field(
        default=None,
        description=(
            "The single currency every costed step agreed on. NULL when steps "
            "disagree — see `mixed_currency`, in which case the totals are NULL "
            "too, because adding across currencies produces a meaningless number."
        ),
    )
    mixed_currency: bool = False
    estimated_cost_micros: int | None = None
    provider_reported_cost_micros: int | None = None
    pricing_versions: list[str] = Field(
        default_factory=list,
        description="Every price list an estimate in this run was computed against.",
    )
    uncosted_generations: int = 0


class LatencyOut(BaseModel):
    """Where the run's wall time went.

    `total_ms` is the run's own span (`started_at` to `completed_at`) and is NOT
    the sum of the parts: steps may overlap, and a run has time in it that no
    step claims. `queue_ms` is the wait before work began — `created_at` to
    `started_at` — which is latency the agent did not cause and should not be
    charged with.
    """

    total_ms: int | None
    queue_ms: int | None
    model_ms: int
    tool_ms: int
    context_ms: int


class _TokenTotal:
    """A running sum that remembers whether anything was ever added to it.

    THE POINT IS THE FLAG, not the sum. `0` is the correct answer for a column
    that was measured as zero and the wrong answer for one nobody measured, and
    an `int` alone cannot tell the two apart — which is exactly how `or 0`
    turned unmeasured generations into zero-token ones. `seen` is what makes
    `resolve()` able to answer `None` instead of guessing.

    Mirrors the `estimated`/`saw_estimated` pair `EconomicsOut` is built from;
    this is that pair given a name, because tokens need six of it.
    """

    __slots__ = ("total", "seen")

    def __init__(self) -> None:
        self.total = 0
        self.seen = False

    def add(self, value: int | None) -> None:
        """A `None` is NOT a zero. It is skipped, and it does not set `seen`."""
        if value is None:
            return
        self.total += value
        self.seen = True

    def resolve(self) -> int | None:
        return self.total if self.seen else None


class _ModelAccumulator:
    """One provider/model pair, accumulating across a run's generations.

    Exists because the aggregation needs SEVEN pieces of state per model (six
    token totals with their flags, plus the untokened count) and the previous
    rebuild-an-immutable-model-per-step loop had nowhere to keep the flags.
    Private and local to `RunTelemetryOut.of`, which stays the only place this
    arithmetic lives.
    """

    __slots__ = (
        "provider", "model", "generations", "untokened", "latency_ms",
        "input", "output", "cached", "cache_read", "cache_write", "reasoning",
    )

    def __init__(self, *, provider: str | None, model: str | None) -> None:
        self.provider = provider
        self.model = model
        self.generations = 0
        self.untokened = 0
        self.latency_ms = 0
        self.input = _TokenTotal()
        self.output = _TokenTotal()
        self.cached = _TokenTotal()
        self.cache_read = _TokenTotal()
        self.cache_write = _TokenTotal()
        self.reasoning = _TokenTotal()

    def add(self, step, latency: int) -> None:
        self.generations += 1
        self.latency_ms += latency
        # UNTOKENED IS DECIDED ON INPUT AND OUTPUT ALONE. Cache and reasoning
        # tokens are absent from most providers' responses, so counting a
        # generation as unmeasured because it lacked them would mark nearly
        # every real generation and the number would carry no information.
        if step.input_tokens is None and step.output_tokens is None:
            self.untokened += 1
        self.input.add(step.input_tokens)
        self.output.add(step.output_tokens)
        self.cached.add(step.cached_tokens)
        self.cache_read.add(step.cache_read_tokens)
        self.cache_write.add(step.cache_write_tokens)
        self.reasoning.add(step.reasoning_tokens)

    def resolve(self) -> ModelUsageOut:
        return ModelUsageOut(
            provider=self.provider,
            model=self.model,
            generations=self.generations,
            untokened_generations=self.untokened,
            input_tokens=self.input.resolve(),
            output_tokens=self.output.resolve(),
            cached_tokens=self.cached.resolve(),
            cache_read_tokens=self.cache_read.resolve(),
            cache_write_tokens=self.cache_write.resolve(),
            reasoning_tokens=self.reasoning.resolve(),
            latency_ms=self.latency_ms,
        )


class RunTelemetryOut(BaseModel):
    """One agent attempt, whole — WHO, WHAT, WHICH MODEL, and what it cost."""

    run: RunOut
    task_description: str
    #: Which attempt at the task this is, 1-based, and how many exist. This is
    #: the domain's answer to "retries": a retry is another RUN of the same task,
    #: which the model already represents — a counter on the run would be a
    #: second, disagreeable source for a fact the ledger holds structurally.
    attempt: int
    task_attempts: int
    steps_recorded: int
    context: ContextUsageOut
    models: list[ModelUsageOut]
    tools: list[ToolUsageOut]
    latency: LatencyOut
    economics: EconomicsOut
    #: NULL when nothing has judged the run yet — which is the common case and
    #: must stay distinguishable from a judged failure.
    outcome: OutcomeOut | None

    @classmethod
    def of(cls, source) -> "RunTelemetryOut":
        """Aggregate a `RunTelemetry` into the rollup.

        Pure: takes rows, returns totals, touches nothing. That is what lets the
        summation — including the estimated/reported split, which is the part
        worth being sure about — be tested without a database.
        """
        run = source.run
        steps = list(source.steps)

        models: dict[tuple[str | None, str | None], _ModelAccumulator] = {}
        tools: dict[tuple[str | None, str | None], ToolUsageOut] = {}
        ctx_fetches = ctx_items = ctx_tokens = ctx_retrievals = 0
        model_ms = tool_ms = context_ms = 0
        estimated = reported = 0
        saw_estimated = saw_reported = False
        currencies: set[str] = set()
        pricing_versions: set[str] = set()
        uncosted = 0

        for step in steps:
            latency = step.latency_ms or 0

            if step.step_type == StepType.MODEL_GENERATION.value:
                model_ms += latency
                key = (step.model_provider, step.model_name)
                current = models.get(key)
                if current is None:
                    current = _ModelAccumulator(
                        provider=step.model_provider, model=step.model_name
                    )
                    models[key] = current
                current.add(step, latency)

                # THE SPLIT. An estimate and a provider's own figure accumulate
                # into different totals and are never added to each other: one is
                # arithmetic over a price list, the other is the provider's, and a
                # sum of the two is neither.
                if step.cost_micros is None:
                    uncosted += 1
                elif step.cost_basis == CostBasis.PROVIDER_REPORTED.value:
                    reported += step.cost_micros
                    saw_reported = True
                else:
                    estimated += step.cost_micros
                    saw_estimated = True

                if step.cost_micros is not None and step.cost_currency:
                    currencies.add(step.cost_currency)
                if step.pricing_version:
                    pricing_versions.add(step.pricing_version)

            elif step.step_type == StepType.TOOL_CALL.value:
                tool_ms += latency
                key = (step.tool_name, step.tool_kind)
                current_tool = tools.get(key)
                if current_tool is None:
                    current_tool = ToolUsageOut(
                        tool_name=step.tool_name,
                        tool_kind=step.tool_kind,
                        calls=0,
                        failures=0,
                        latency_ms=0,
                    )
                tools[key] = ToolUsageOut(
                    tool_name=current_tool.tool_name,
                    tool_kind=current_tool.tool_kind,
                    calls=current_tool.calls + 1,
                    failures=(
                        current_tool.failures
                        + (1 if step.status == StepStatus.FAILED.value else 0)
                    ),
                    latency_ms=current_tool.latency_ms + latency,
                )

            elif step.step_type == StepType.CONTEXT_FETCH.value:
                context_ms += latency
                ctx_fetches += 1
                ctx_items += step.context_items or 0
                ctx_tokens += step.context_tokens or 0
                ctx_retrievals += step.retrieval_count or 0

        # Mixed currencies are reported, never summed. Adding 300 micro-USD to
        # 300 micro-EUR yields a number with no meaning, and a reader has no way
        # to tell it apart from a real one.
        mixed = len(currencies) > 1
        currency = next(iter(currencies)) if len(currencies) == 1 else None

        economics = EconomicsOut(
            currency=currency,
            mixed_currency=mixed,
            estimated_cost_micros=None if mixed or not saw_estimated else estimated,
            provider_reported_cost_micros=None if mixed or not saw_reported else reported,
            pricing_versions=sorted(pricing_versions),
            uncosted_generations=uncosted,
        )

        return cls(
            run=RunOut.of(run),
            task_description=source.task_description,
            attempt=source.attempt,
            task_attempts=source.task_attempts,
            steps_recorded=len(steps),
            context=ContextUsageOut(
                version=run.context_version,
                snapshot_id=run.context_snapshot_id,
                fetches=ctx_fetches,
                items=ctx_items,
                tokens=ctx_tokens,
                retrievals=ctx_retrievals,
            ),
            models=[acc.resolve() for acc in models.values()],
            tools=list(tools.values()),
            latency=LatencyOut(
                total_ms=_span_ms(run.started_at, run.completed_at),
                queue_ms=_span_ms(run.created_at, run.started_at),
                model_ms=model_ms,
                tool_ms=tool_ms,
                context_ms=context_ms,
            ),
            economics=economics,
            outcome=OutcomeOut.of(source.outcome) if source.outcome is not None else None,
        )


# ── Pages ───────────────────────────────────────────────────────────────────


class TaskPageOut(BaseModel):
    tasks: list[TaskOut]
    count: int
    total: int
    limit: int
    offset: int


class RunPageOut(BaseModel):
    runs: list[RunOut]
    count: int
    total: int
    limit: int
    offset: int


class StepPageOut(BaseModel):
    steps: list[StepOut]
    count: int
    total: int
    limit: int
    offset: int
