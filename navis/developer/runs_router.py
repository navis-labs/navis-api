"""The agent execution ledger — `/api/v1/developer/tasks` and `/developer/runs`.

THE CHAIN, FOR EVERY ROUTE HERE:

    credential -> get_principal -> Principal(org from membership/api_keys row)
               -> require_permission(view_observation | record_agent_runs)  (403)
               -> require_capability(STATE)                                 (402)
               -> rate_limit                                                (429)
               -> AgentLifecycleService, scoped to ctx.organization_id

TWO PERMISSIONS, AND THE SPLIT IS THE POINT. Reads are `view_observation` — this
IS the observability surface, and the permission that already opens `/outcomes`
is the one that should open the run ledger. Writes are `record_agent_runs`, a
permission of its own, because `view_observation` is a READ permission and must
never authorize a write, and `manage_service_accounts` is wrong in the other
direction: it would mean only an agent that can MINT other agents may report
what it did.

WRITES ARE NOT HUMAN-ONLY, unlike the agent lifecycle next door. That asymmetry
is deliberate. Registering an agent is human-only because an agent that can
create agents can mint identities its own revocation does not reach. Recording a
run is the OPPOSITE case: the agent is the only party that knows what it did, and
a ledger only a human can write is a ledger nobody writes.

THE CAPABILITY IS `STATE`, matching the agent routes rather than `OBSERVATION`.
Considered and rejected: `OBSERVATION` would tier the execution ledger by plan
while agent registration stays free, leaving an organization able to create
agents that cannot record what they did. `STATE` is included from FREE upward, so
this gate does not price the ledger — while still enforcing LIFECYCLE through
`EntitlementService._decide`, so a SUSPENDED or CHURNED organization stops
writing.

NO ROUTE TAKES AN ORGANIZATION, and no route takes an identity in a BODY or a
QUERY. `agent_id` appears only as a PATH segment, which is the rule
`test_no_developer_route_accepts_an_identity_parameter` enforces: a name that
addresses a resource in a path is looked up under the tenant, while the same word
arriving alongside a resolved resource is how "filter" quietly becomes "scope".

That guard is what produced the shape below, and the shape is better for it. A
run is opened at `/developer/tasks/{task_id}/runs` and takes its agent FROM THE
TASK — so a run can never belong to a different agent than the work it attempts,
an invariant that was previously true only because every caller happened to pass
a matching id.

For a machine caller the path `agent_id` is additionally overwritten with its own
`service_account_id`, so an agent credential can only ever write its own history.
Every id in a path is resolved UNDER `ctx.organization_id`, so another tenant's id
answers 404 — indistinguishable from one that exists nowhere.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from navis.developer.gateway import DeveloperContext, require_developer
from navis.developer.lifecycle import DEFAULT_PAGE_SIZE, MAX_ECONOMICS_RUNS, MAX_PAGE_SIZE
from navis.developer.run_schemas import (
    CloseTaskRequest,
    CreateTaskRequest,
    EconomicsRollupOut,
    FinishRunRequest,
    LatestOutcomeOut,
    OutcomeOut,
    RecordOutcomeRequest,
    RecordStepRequest,
    RunOut,
    RunPageOut,
    RunTelemetryOut,
    StartRunRequest,
    StepOut,
    StepPageOut,
    TaskOut,
    TaskPageOut,
)
from navis.developer.run_service import (
    AgentLifecycleService,
    InvalidTransition,
    LifecycleError,
    StaleWrite,
    StepLimitReached,
    UnknownAgent,
)
from navis.developer.sanitize import PayloadTooLarge
from navis.developer.schemas import DEVELOPER_ERRORS_GATED, DeveloperErrorOut
from navis.identity.authz import PrincipalKind
from navis.identity.permissions import Permission
from navis.identity.throttle import rate_limit
from navis.organization.contract import Capability
from navis.platform.database import get_session

router = APIRouter(prefix="/developer", tags=["developer"])

#: Same capability as the agent routes — see the module note on why not
#: OBSERVATION.
_LEDGER_CAPABILITY = Capability.STATE

_reader = require_developer(Permission.VIEW_OBSERVATION, _LEDGER_CAPABILITY)
_writer = require_developer(Permission.RECORD_AGENT_RUNS, _LEDGER_CAPABILITY)

_READ_LIMIT = Depends(rate_limit("read", "rate_limit_read_per_window"))
_WRITE_LIMIT = Depends(rate_limit("run_write", "rate_limit_run_write_per_window"))

_ERRORS: dict[int | str, dict] = {
    **DEVELOPER_ERRORS_GATED,
    404: {
        "model": DeveloperErrorOut,
        "description": (
            "No such record in THIS organization. An id that exists in another "
            "tenant answers exactly this, and is indistinguishable from an id "
            "that exists nowhere."
        ),
    },
}

_WRITE_ERRORS: dict[int | str, dict] = {
    **_ERRORS,
    400: {"model": DeveloperErrorOut, "description": "The payload was refused — see the detail."},
    409: {"model": DeveloperErrorOut, "description": "The lifecycle does not permit that move."},
}


def _acting_agent(ctx: DeveloperContext, requested: uuid.UUID | None) -> uuid.UUID:
    """Which agent this write records against.

    A MACHINE CALLER CANNOT CHOOSE. `service_account_id` comes from the
    `api_keys` row that authenticated the request, so an agent credential always
    records its own history and a supplied `agent_id` is ignored rather than
    honoured — there is no request an agent can construct that writes to another
    agent's ledger.

    A human operator must name one, and the service validates it belongs to this
    tenant before writing.
    """
    if ctx.kind == PrincipalKind.SERVICE_ACCOUNT.value and ctx.service_account_id is not None:
        return ctx.service_account_id
    if requested is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "agent_id is required when a signed-in member records on an agent's behalf",
        )
    return requested


def _refuse(exc: Exception) -> HTTPException:
    """Map a service refusal to the status code that describes it.

    `UnknownAgent` is 404 and not 403: the resource is not addressable by this
    caller, and saying "forbidden" would confirm that the id names something
    real somewhere else.
    """
    if isinstance(exc, UnknownAgent):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    if isinstance(exc, (InvalidTransition, StepLimitReached, StaleWrite)):
        return HTTPException(status.HTTP_409_CONFLICT, str(exc))
    return HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))


# ── Tasks ───────────────────────────────────────────────────────────────────


@router.post(
    "/agents/{agent_id}/tasks",
    response_model=TaskOut,
    status_code=status.HTTP_201_CREATED,
    responses=_WRITE_ERRORS,
    dependencies=[_WRITE_LIMIT],
    summary="Record what an agent was asked to do",
)
async def create_task(
    agent_id: uuid.UUID,
    body: CreateTaskRequest,
    ctx: DeveloperContext = Depends(_writer),
    db: AsyncSession = Depends(get_session),
) -> TaskOut:
    service = AgentLifecycleService(db)
    try:
        task = await service.create_task(
            ctx.organization_id,
            agent_id=_acting_agent(ctx, agent_id),
            description=body.description,
            idempotency_key=body.idempotency_key,
            created_by=ctx.subject_id,
            metadata=body.metadata,
        )
    except (LifecycleError, PayloadTooLarge) as exc:
        raise _refuse(exc)
    await db.commit()
    return TaskOut.of(task)


@router.get(
    "/tasks",
    response_model=TaskPageOut,
    responses=_ERRORS,
    dependencies=[_READ_LIMIT],
    summary="This organization's tasks, newest first",
)
async def list_tasks(
    task_status: str | None = Query(default=None, alias="status"),
    # ── A FILTER, and the reason it exists ─────────────────────────────
    # A client that records work against a task and then LOSES the id — a
    # developer tool restarting in the same worktree, a scheduler resuming
    # after a deploy — has no way back to it, because a task is addressable
    # only by an id it no longer holds. Its own key, written into
    # `metadata.navisWorkKey` when the task was created, is the way back.
    #
    # Without this the client must page the organization's open tasks and
    # match the key itself. That works until the tenant has more open tasks
    # than the client is willing to read, and then it does not FAIL — it
    # reports the work stream as absent and a duplicate one is opened beside
    # the original. Exactly the failure `task_id` above was added to fix for
    # runs, one level up.
    #
    # Like every other filter here it narrows the tenant predicate and never
    # replaces it: a work key belonging to another organization matches no
    # rows rather than reaching them. Matched EXACTLY, and backed by
    # `ix_agent_tasks_org_work_key` (organization_id, task_metadata ->>
    # 'navisWorkKey').
    #
    # Bounded at 128 characters so the parameter cannot become a channel for
    # an arbitrarily large string; anything longer is refused as malformed
    # rather than sent to the database.
    work_key: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    ctx: DeveloperContext = Depends(_reader),
    db: AsyncSession = Depends(get_session),
) -> TaskPageOut:
    page = await AgentLifecycleService(db).list_tasks(
        ctx.organization_id,
        status=task_status,
        work_key=work_key,
        limit=limit,
        offset=offset,
    )
    rows = [TaskOut.of(r) for r in page.rows]
    return TaskPageOut(
        tasks=rows, count=len(rows), total=page.total, limit=page.limit, offset=page.offset
    )


@router.get("/tasks/{task_id}", response_model=TaskOut, responses=_ERRORS, dependencies=[_READ_LIMIT])
async def get_task(
    task_id: uuid.UUID,
    ctx: DeveloperContext = Depends(_reader),
    db: AsyncSession = Depends(get_session),
) -> TaskOut:
    task = await AgentLifecycleService(db).get_task(ctx.organization_id, task_id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such task in this organization")
    return TaskOut.of(task)


@router.patch(
    "/tasks/{task_id}",
    response_model=TaskOut,
    responses=_WRITE_ERRORS,
    dependencies=[_WRITE_LIMIT],
    summary="Close a task as completed or abandoned",
)
async def close_task(
    task_id: uuid.UUID,
    body: CloseTaskRequest,
    ctx: DeveloperContext = Depends(_writer),
    db: AsyncSession = Depends(get_session),
) -> TaskOut:
    service = AgentLifecycleService(db)
    try:
        task = await service.close_task(ctx.organization_id, task_id, status=body.status.value)
    except LifecycleError as exc:
        raise _refuse(exc)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such task in this organization")
    await db.commit()
    return TaskOut.of(task)


# ── Runs ────────────────────────────────────────────────────────────────────


@router.post(
    "/tasks/{task_id}/runs",
    response_model=RunOut,
    status_code=status.HTTP_201_CREATED,
    responses=_WRITE_ERRORS,
    dependencies=[_WRITE_LIMIT],
    summary="Open one attempt at a task",
)
async def start_run(
    task_id: uuid.UUID,
    body: StartRunRequest,
    ctx: DeveloperContext = Depends(_writer),
    db: AsyncSession = Depends(get_session),
) -> RunOut:
    service = AgentLifecycleService(db)
    try:
        run = await service.start_run(
            ctx.organization_id,
            # THE PATH, and the agent is not passed at all. The service reads it
            # from the task, which is the invariant this route exists to hold: a
            # run can never belong to a different agent than the work it
            # attempts. Taking a task id from the BODY would have reintroduced
            # exactly the arbitrary agent/task/run combination the path shape
            # was chosen to make unrepresentable.
            task_id=task_id,
            context_snapshot_id=body.context_snapshot_id,
            context_version=body.context_version,
            correlation_id=body.correlation_id,
            idempotency_key=body.idempotency_key,
            created_by=ctx.subject_id,
            metadata=body.metadata,
        )
    except (LifecycleError, PayloadTooLarge) as exc:
        raise _refuse(exc)
    await db.commit()
    return RunOut.of(run)


@router.get(
    "/runs",
    response_model=RunPageOut,
    responses=_ERRORS,
    dependencies=[_READ_LIMIT],
    summary="This organization's runs, newest first",
)
async def list_runs(
    run_status: str | None = Query(default=None, alias="status"),
    # ── A FILTER, and the reason it exists ─────────────────────────────
    # Without it a caller looking for one task's runs must read the newest
    # page of the ORGANIZATION's runs and sift them itself. That is not
    # merely slow: once other work pushes a task's newest run past the page,
    # the task's runs are simply absent from the answer, and a caller cannot
    # tell "this task has no runs" from "its runs are further down". The
    # Developer CLI reads a work stream's checkpoint that way, so the failure
    # surfaces as a session restoring nothing while a checkpoint exists.
    #
    # Like every other filter here it narrows the tenant predicate and never
    # replaces it: a task id belonging to another organization matches no
    # rows rather than reaching them. `ix_agent_runs_task_created`
    # (task_id, created_at DESC) already backs exactly this query.
    task_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    ctx: DeveloperContext = Depends(_reader),
    db: AsyncSession = Depends(get_session),
) -> RunPageOut:
    page = await AgentLifecycleService(db).list_runs(
        ctx.organization_id,
        task_id=task_id,
        status=run_status,
        limit=limit,
        offset=offset,
    )
    rows = [RunOut.of(r) for r in page.rows]
    return RunPageOut(
        runs=rows, count=len(rows), total=page.total, limit=page.limit, offset=page.offset
    )


@router.get("/runs/{run_id}", response_model=RunOut, responses=_ERRORS, dependencies=[_READ_LIMIT])
async def get_run(
    run_id: uuid.UUID,
    ctx: DeveloperContext = Depends(_reader),
    db: AsyncSession = Depends(get_session),
) -> RunOut:
    run = await AgentLifecycleService(db).get_run(ctx.organization_id, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run in this organization")
    return RunOut.of(run)


@router.patch(
    "/runs/{run_id}",
    response_model=RunOut,
    responses=_WRITE_ERRORS,
    dependencies=[_WRITE_LIMIT],
    summary="Move a run to a terminal state",
)
async def finish_run(
    run_id: uuid.UUID,
    body: FinishRunRequest,
    ctx: DeveloperContext = Depends(_writer),
    db: AsyncSession = Depends(get_session),
) -> RunOut:
    service = AgentLifecycleService(db)
    try:
        run = await service.finish_run(
            ctx.organization_id,
            run_id,
            status=body.status.value,
            failure_category=body.failure_category.value if body.failure_category else None,
            failure_reason=body.failure_reason,
        )
    except (LifecycleError, PayloadTooLarge) as exc:
        raise _refuse(exc)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run in this organization")
    await db.commit()
    return RunOut.of(run)


# ── Steps ───────────────────────────────────────────────────────────────────


@router.post(
    "/runs/{run_id}/steps",
    response_model=StepOut,
    status_code=status.HTTP_201_CREATED,
    responses=_WRITE_ERRORS,
    dependencies=[_WRITE_LIMIT],
    summary="Append one event to a run's timeline",
)
async def record_step(
    run_id: uuid.UUID,
    body: RecordStepRequest,
    ctx: DeveloperContext = Depends(_writer),
    db: AsyncSession = Depends(get_session),
) -> StepOut:
    service = AgentLifecycleService(db)
    try:
        step = await service.record_step(
            ctx.organization_id,
            run_id,
            step_type=body.step_type.value,
            status=body.status.value,
            label=body.label,
            started_at=body.started_at,
            completed_at=body.completed_at,
            tool_name=body.tool_name,
            tool_kind=body.tool_kind.value if body.tool_kind else None,
            model_provider=body.model_provider,
            model_name=body.model_name,
            model_generation_id=body.model_generation_id,
            input_tokens=body.input_tokens,
            output_tokens=body.output_tokens,
            cached_tokens=body.cached_tokens,
            cache_read_tokens=body.cache_read_tokens,
            cache_write_tokens=body.cache_write_tokens,
            reasoning_tokens=body.reasoning_tokens,
            cost_micros=body.cost_micros,
            cost_currency=body.cost_currency,
            cost_basis=body.cost_basis.value if body.cost_basis else None,
            pricing_version=body.pricing_version,
            context_items=body.context_items,
            context_tokens=body.context_tokens,
            retrieval_count=body.retrieval_count,
            latency_ms=body.latency_ms,
            error_category=body.error_category.value if body.error_category else None,
            error_message=body.error_message,
            request_metadata=body.request_metadata,
            response_metadata=body.response_metadata,
        )
    except (LifecycleError, PayloadTooLarge) as exc:
        raise _refuse(exc)
    await db.commit()
    return StepOut.of(step)


@router.get(
    "/runs/{run_id}/steps",
    response_model=StepPageOut,
    responses=_ERRORS,
    dependencies=[_READ_LIMIT],
    summary="One run's timeline, in sequence order",
)
async def list_steps(
    run_id: uuid.UUID,
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    ctx: DeveloperContext = Depends(_reader),
    db: AsyncSession = Depends(get_session),
) -> StepPageOut:
    page = await AgentLifecycleService(db).list_steps(
        ctx.organization_id, run_id, limit=limit, offset=offset
    )
    if page is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run in this organization")
    rows = [StepOut.of(r) for r in page.rows]
    return StepPageOut(
        steps=rows, count=len(rows), total=page.total, limit=page.limit, offset=page.offset
    )


# ── Outcome ─────────────────────────────────────────────────────────────────


@router.put(
    "/runs/{run_id}/outcome",
    response_model=OutcomeOut,
    responses=_WRITE_ERRORS,
    dependencies=[_WRITE_LIMIT],
    summary="Record or revise what came of a run",
)
async def record_outcome(
    run_id: uuid.UUID,
    body: RecordOutcomeRequest,
    ctx: DeveloperContext = Depends(_writer),
    db: AsyncSession = Depends(get_session),
) -> OutcomeOut:
    service = AgentLifecycleService(db)
    try:
        outcome = await service.record_outcome(
            ctx.organization_id,
            run_id,
            status=body.status.value,
            summary=body.summary,
            result_reference=body.result_reference,
            quality_score=body.quality_score,
            human_corrections=body.human_corrections,
            produced_at=body.produced_at,
            metadata=body.metadata,
            if_newer=body.if_newer,
        )
    except (LifecycleError, PayloadTooLarge) as exc:
        raise _refuse(exc)
    if outcome is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run in this organization")
    await db.commit()
    # THE PERSISTED ROW, read back off the flushed object — never an echo of
    # the request. Everything this domain does to a payload happens between the
    # two: the summary is scrubbed, the metadata is scrubbed and bounded, and a
    # value that matched a credential pattern is not what was sent. A client
    # that compares this against what it submitted can therefore detect a write
    # that succeeded and stored something else, which is the only way the
    # 2,637-character checkpoint stored as ten characters would have been
    # caught at the time instead of a day later.
    return OutcomeOut.of(outcome)


@router.get(
    "/tasks/{task_id}/latest-outcome",
    response_model=LatestOutcomeOut,
    responses=_ERRORS,
    dependencies=[_READ_LIMIT],
    summary="The newest outcome carrying content across every run of one task",
)
async def get_latest_task_outcome(
    task_id: uuid.UUID,
    ctx: DeveloperContext = Depends(_reader),
    db: AsyncSession = Depends(get_session),
) -> LatestOutcomeOut:
    """One indexed read where a client would otherwise scan.

    AN OUTCOME IS ADDRESSABLE ONLY BY `run_id`, so "what is the latest state of
    this work" had no answer but: list the task's runs, then fetch outcomes one
    run at a time until something turns up. That forces every client to pick a
    scan depth, and a scan depth is a silent correctness bound — past it a real
    checkpoint reads as ABSENT, indistinguishable from work that never existed.

    NO NEW FACT AND NO NEW AUTHORITY. Every field returned is one
    `/developer/runs/{id}` and `/developer/runs/{id}/outcome` already serve to
    `view_observation`. What changes is that the selection happens where the
    tenant predicate and the index are, instead of across N round trips.

    Returns the newest outcome whose summary is non-empty, ordered by reality
    time (`produced_at`) and not by row time, with the run that produced it.
    404 when the task is not this organization's, and equally when it has no
    such outcome yet — the caller's next move is the same either way.
    """
    found = await AgentLifecycleService(db).latest_task_outcome(ctx.organization_id, task_id)
    if found is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "no outcome with a summary recorded for that task in this organization",
        )
    outcome, run = found
    return LatestOutcomeOut.of(outcome, run)


# ── Telemetry rollup ────────────────────────────────────────────────────────


@router.get(
    "/runs/{run_id}/telemetry",
    response_model=RunTelemetryOut,
    responses=_ERRORS,
    dependencies=[_READ_LIMIT],
    summary="One run, aggregated: model, context, tools, latency, economics, outcome",
)
async def get_run_telemetry(
    run_id: uuid.UUID,
    ctx: DeveloperContext = Depends(_reader),
    db: AsyncSession = Depends(get_session),
) -> RunTelemetryOut:
    """Everything about one attempt, in one read.

    SAME PERMISSION, SAME CAPABILITY, NO NEW FACTS. Every figure is derived from
    rows `/runs/{id}`, `/runs/{id}/steps` and `/runs/{id}/outcome` already serve
    to `view_observation`; this route runs the summation once, server-side, so
    the REST consumer, both SDKs and MCP cannot each derive a different total.

    It is a READ, and it stays one. The rollup is where a future budget or
    routing decision will read its inputs — but reading is all it does, and
    nothing in this phase acts on what it returns.
    """
    telemetry = await AgentLifecycleService(db).get_run_telemetry(ctx.organization_id, run_id)
    if telemetry is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run in this organization")
    return RunTelemetryOut.of(telemetry)


@router.get(
    "/economics",
    response_model=EconomicsRollupOut,
    responses=_ERRORS,
    dependencies=[_READ_LIMIT],
    summary="Cross-run economics: what reaching a successful outcome cost",
)
async def get_economics_rollup(
    ctx: DeveloperContext = Depends(_reader),
    db: AsyncSession = Depends(get_session),
    task_id: uuid.UUID | None = Query(default=None, description="One task's attempts only."),
    # NO `agent_id` QUERY PARAMETER, deliberately. `agent_id` is allowed only as
    # a PATH segment on this boundary — the rule
    # `test_no_developer_route_accepts_an_identity_parameter` enforces — because
    # an identity supplied alongside an already-resolved resource is how a
    # filter quietly becomes a scope. The service still takes the argument, and
    # the Postgres suite exercises agent isolation through it; if narrowing by
    # agent is wanted at the edge it belongs at `/agents/{agent_id}/economics`,
    # which is a resource path and is looked up under the tenant.
    model_name: str | None = Query(
        default=None,
        max_length=128,
        description=(
            "Runs that used this model at least once. A FILTER, not a grouping "
            "key — a multi-model run is reported whole, with its whole cost."
        ),
    ),
    context_version: str | None = Query(default=None, max_length=64),
) -> EconomicsRollupOut:
    """What a successful outcome costs, grouped by task and by whether Navis
    context was fetched.

    THE QUESTION THIS ANSWERS. Per-run telemetry already says what one attempt
    cost; it cannot say what SUCCESS costs, because a retry is another run of
    the same task and dividing by attempts makes an agent that needed four
    tries look four times cheaper than one that needed none. So the numerator
    is every attempt and the denominator is successful outcomes.

    DERIVED, NOT STORED. Every figure is arithmetic over `agent_runs`,
    `agent_run_steps` and `agent_run_outcomes` — rows this same credential can
    already read through `/runs`, `/runs/{id}/steps` and `/runs/{id}/outcome`,
    under the same permission, the same capability and the same tenant filter.
    No table was added for it, and nothing is cached: recomputing is what keeps
    the answer right when an outcome is judged after the fact.

    METADATA ONLY. Counts, costs, rates and versions. No context CONTENT, no
    memory, no source document, no argument value — `ContextUsageOut` already
    draws that line for one run and this holds it across many.

    ── WHAT IT DOES NOT CLAIM ──────────────────────────────────────────
    `context_fetched` records that the agent ASKED Navis for context, not that
    the context caused the outcome. Runs are not assigned to the two groups;
    they self-select, so a difference between them is a correlation. It also
    does not identify WHICH context was supplied — Navis issues no context
    identity today, so two runs in one group may have been given different
    facts. The response repeats this in `interpretation`, because a number
    travels further than the endpoint that produced it.
    """
    rows = await AgentLifecycleService(db).list_run_economics(
        ctx.organization_id,
        task_id=task_id,
        model_name=model_name,
        context_version=context_version,
    )
    truncated = len(rows) > MAX_ECONOMICS_RUNS
    return EconomicsRollupOut.of(rows[:MAX_ECONOMICS_RUNS], truncated=truncated)


@router.get(
    "/runs/{run_id}/outcome",
    response_model=OutcomeOut,
    responses=_ERRORS,
    dependencies=[_READ_LIMIT],
)
async def get_outcome(
    run_id: uuid.UUID,
    ctx: DeveloperContext = Depends(_reader),
    db: AsyncSession = Depends(get_session),
) -> OutcomeOut:
    outcome = await AgentLifecycleService(db).get_outcome(ctx.organization_id, run_id)
    if outcome is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "no outcome recorded for that run in this organization"
        )
    return OutcomeOut.of(outcome)
