"""The agent execution lifecycle service — the only writer to the four tables.

EVERY METHOD TAKES THE TENANT AS ITS FIRST ARGUMENT, and never reads it from
anything a client sent. The routes pass `DeveloperContext.organization_id`, which
`authz` resolved from the credential. There is no method here that accepts an
organization from a payload, and none may be added: a parameter like that is one
somebody eventually trusts.

EVERY READ IS FILTERED BY TENANT, INCLUDING READS BY PRIMARY KEY. Looking a run
up by id alone and then checking its organization afterwards is the same bug
written two ways — the row has already been loaded, and the check is one `return`
away from being forgotten. So the tenant is part of the WHERE clause, and a run
belonging to another organization is indistinguishable from one that does not
exist. That is why every lookup here returns `None` rather than raising: the
route turns it into 404, which is the answer that leaks nothing.

THE AGENT IS ALSO NOT A PARAMETER, on the write path. When an agent records its
own work the routes pass `DeveloperContext.service_account_id`, so an agent
credential can only ever write its own history. A human operator may name an
agent explicitly, and that name is validated to belong to the caller's tenant
before anything is written.

NOTHING HERE STORES A SECRET. Every free-text and metadata field goes through
`sanitize.py` before it reaches a column.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import (
    String,
    and_,
    case,
    distinct,
    exists,
    func,
    literal_column,
    nulls_last,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from navis.developer.lifecycle import (
    ALLOWED_RUN_TRANSITIONS,
    DEFAULT_PAGE_SIZE,
    MAX_ECONOMICS_RUNS,
    MAX_PAGE_SIZE,
    MAX_STEP_LABEL,
    MAX_STEPS_PER_RUN,
    MAX_TASK_DESCRIPTION,
    MAX_TOOL_NAME,
    CostBasis,
    RunStatus,
    StepStatus,
    StepType,
    TaskStatus,
    may_transition,
)
from navis.developer.run_models import AgentRun, AgentRunOutcome, AgentRunStep, AgentTask
from navis.developer.sanitize import PayloadTooLarge, sanitize_metadata, sanitize_text
from navis.identity.enterprise_models import ServiceAccount
from navis.platform.monitoring.metrics import timed


class LifecycleError(Exception):
    """A refusal the caller can act on. Routes map it to 400 or 409."""


class UnknownAgent(LifecycleError):
    """The named agent does not exist IN THIS TENANT. Routes map it to 404."""


class InvalidTransition(LifecycleError):
    """A run was asked to move somewhere its current status does not allow."""


class StepLimitReached(LifecycleError):
    """This run already holds as many steps as the platform will record."""


class StaleWrite(LifecycleError):
    """A conditional write lost to an outcome that was already newer.

    Routes map it to 409. Distinct from every other `LifecycleError` because it
    is not a malformed request: the caller did nothing wrong, it simply arrived
    second. A client is expected to treat it as a no-op, not as a failure to
    retry — retrying a stale write reproduces it exactly.
    """


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Compare instants, never wall clocks.

    `DateTime(timezone=True)` returns aware datetimes, but a caller may hand in
    a naive one and Python refuses to order the two — a `TypeError` raised from
    inside a concurrency guard would turn a losing write into a 500 rather than
    a refusal. A naive value is read as UTC, which is the only assumption this
    codebase makes anywhere else about a timestamp with no zone.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def loses_to_stored_outcome(
    *, stored: datetime | None, incoming: datetime | None, if_newer: bool
) -> bool:
    """Should this conditional write be refused as stale?

    PURE, and module-level, so the concurrency rule can be pinned on plain
    values instead of through a session, a transaction and four layers of
    transport. The rule is small and the cost of getting it wrong is a silently
    replaced checkpoint, which is exactly the combination that deserves direct
    tests rather than incidental coverage.

    Refused only when the caller ASKED to be held to monotonic authorship, both
    sides can say when they were observed, and the incoming one is STRICTLY
    older.

    ── EQUALITY IS ADMITTED, AND THE REASON IS STRUCTURAL ──────────────
    An earlier draft refused an equal instant too, on the grounds that two
    writes bearing the same timestamp are not ordered by anything and letting
    the second win makes arrival the tiebreak. That reasoning is sound between
    WRITERS and does not apply here, because on one run there is only ever one.
    `agent_run_outcomes` is UNIQUE per run and a session owns exactly one run,
    so every write to a given row comes from the same author revising its own
    statement — and within one author, arrival order IS authorship order.

    Refusing equality therefore refused the normal case: an agent recording a
    decision and then a second decision inside the same millisecond had the
    second silently dropped. Caught by the existing checkpoint-merge tests,
    which is what they are for.

    What remains guarded is the case that is genuinely dangerous and cannot be
    ordered by arrival: a STRICTLY older statement arriving last, because a
    retry overtook the request it retried or a delayed write landed after a
    newer one. That write claims to describe an earlier moment, and letting it
    replace the row would take the append-only sections with it.
    """
    if not if_newer or stored is None or incoming is None:
        return False
    return _as_utc(incoming) < _as_utc(stored)


def _clamp(limit: int | None) -> int:
    """Bound every page size. `None` means "caller did not say", not "all rows".

    An unbounded list endpoint is the way a shared table becomes an outage, so
    there is no code path here that omits a LIMIT.
    """
    if limit is None:
        return DEFAULT_PAGE_SIZE
    return max(1, min(int(limit), MAX_PAGE_SIZE))


#: The one indexed metadata key. See `AgentTask.__table_args__`.
WORK_KEY_FIELD = "navisWorkKey"

#: The states a run cannot leave. Derived from the transition table rather than
#: written out again, so a status added there cannot be forgotten here.
_TERMINAL_RUN_STATUSES = frozenset(
    status for status, onward in ALLOWED_RUN_TRANSITIONS.items() if not onward
)


def _work_key_expression():
    """`task_metadata ->> 'navisWorkKey'`, with the KEY as a SQL literal.

    THE LITERAL IS THE WHOLE POINT, and it is not a style choice. The obvious
    spelling, `AgentTask.task_metadata["navisWorkKey"].astext`, renders the key
    as a BIND PARAMETER — `task_metadata ->> $2` — and PostgreSQL can only match
    a query against an expression index when the indexed expression appears
    verbatim. Under a generic plan `$2` is not `'navisWorkKey'`, the match
    fails, and the lookup degrades to a sequential scan of the tenant's tasks:
    the exact cost this index exists to remove, arrived at silently and only on
    the deployments that reuse plans.

    The VALUE stays bound, so the work key itself is never interpolated into
    SQL. `return_type=String` keeps the comparison textual; without it
    SQLAlchemy infers JSONB from the left operand and casts the bound value,
    which again stops the expression matching the index.
    """
    return AgentTask.task_metadata.op("->>", return_type=String)(
        literal_column(f"'{WORK_KEY_FIELD}'")
    )


@dataclass(frozen=True, slots=True)
class Page:
    """One page of rows plus the total, so a client can show "20 of 413".

    The total is a separate COUNT against the same predicate. It is honest about
    cost: on a tenant with millions of runs this is the expensive half of the
    query, and it exists because a list with no total forces a client to guess
    whether there is more.
    """

    rows: list
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class RunEconomics:
    """ONE run, reduced to the figures a cross-run comparison needs.

    THE UNIT OF THE ROLLUP IS THE RUN, not the step, and that is deliberate.
    Costs are summed per run in SQL so the Python layer never holds a step row;
    a tenant with a thousand tasks at a thousand steps each would otherwise ship
    a million rows to compute a handful of numbers.

    Every field is already recorded on an immutable row. Nothing here is new
    evidence — it is the same arithmetic `RunTelemetryOut.of` performs for one
    run, expressed so many runs can be grouped.
    """

    task_id: uuid.UUID
    task_description: str
    agent_id: uuid.UUID | None
    #: Whether this run asked Navis for organizational context. Derived from the
    #: presence of a `context_fetch` step, which is the only evidence Navis has
    #: that context was actually taken. It says the agent FETCHED context; it
    #: does not identify which bundle, and it is not a claim about causation.
    context_fetched: bool
    context_version: str | None
    #: Split, never summed — see `CostBasis`. NULL stays NULL: a run whose steps
    #: carried no cost is uncosted, not free.
    estimated_micros: int | None
    reported_micros: int | None
    uncosted_generations: int
    currencies: frozenset[str]
    pricing_versions: frozenset[str]
    #: `None` when nothing has judged the run — which must stay distinguishable
    #: from a judged failure, or the first metric built on this is poisoned.
    outcome_status: str | None
    quality_score: float | None
    human_corrections: int | None
    latency_ms: int | None
    models: frozenset[str]


@dataclass(frozen=True, slots=True)
class RunTelemetry:
    """The RAW rows behind one run's rollup, already tenant-checked.

    Deliberately not the aggregate. The service's job is to fetch exactly the
    rows a rollup is entitled to see; turning them into totals is presentation,
    and lives with the other `Out.of(...)` constructors in `run_schemas.py`. Kept
    apart so the summation can be tested on plain objects without a database.
    """

    run: AgentRun
    task_description: str
    attempt: int
    task_attempts: int
    steps: list[AgentRunStep]
    outcome: AgentRunOutcome | None


class AgentLifecycleService:
    """Task -> Run -> Step -> Outcome, scoped to one tenant per call."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── Agent resolution ────────────────────────────────────────────────

    async def _agent_in_tenant(self, org_id: uuid.UUID, agent_id: uuid.UUID) -> ServiceAccount | None:
        """The agent row, IF it belongs to this tenant.

        The tenant predicate is in the query rather than checked afterwards, so
        another organization's agent id is simply not found — indistinguishable
        from one that exists nowhere.
        """
        return await self._db.scalar(
            select(ServiceAccount).where(
                ServiceAccount.id == agent_id,
                ServiceAccount.organization_id == org_id,
            )
        )

    # ── Task ────────────────────────────────────────────────────────────

    async def create_task(
        self,
        org_id: uuid.UUID,
        *,
        agent_id: uuid.UUID,
        description: str,
        idempotency_key: str | None = None,
        created_by: uuid.UUID | None = None,
        metadata: dict | None = None,
    ) -> AgentTask:
        """Record what an agent was asked to do.

        Idempotent when `idempotency_key` is supplied: a retry returns the
        EXISTING task rather than creating a second one. The uniqueness is
        enforced by the database, and the race is handled by catching the
        violation rather than by checking first — a check-then-insert has a
        window between the two, and under a retry storm that window is exactly
        when both requests arrive.
        """
        if await self._agent_in_tenant(org_id, agent_id) is None:
            raise UnknownAgent("no such agent in this organization")

        description = sanitize_text(
            description, field="description", limit=MAX_TASK_DESCRIPTION
        )
        if not description.strip():
            raise LifecycleError("description is required")

        if idempotency_key is not None:
            existing = await self._task_by_key(org_id, idempotency_key)
            if existing is not None:
                return existing

        task = AgentTask(
            organization_id=org_id,
            agent_id=agent_id,
            description=description,
            status=TaskStatus.OPEN.value,
            idempotency_key=idempotency_key,
            created_by=created_by,
            task_metadata=sanitize_metadata(metadata, field="metadata"),
        )
        self._db.add(task)
        try:
            await self._db.flush()
        except IntegrityError:
            # Lost the race. The winner's row is the answer — this is what makes
            # the operation exactly-once rather than merely usually-once.
            await self._db.rollback()
            if idempotency_key is not None:
                existing = await self._task_by_key(org_id, idempotency_key)
                if existing is not None:
                    return existing
            raise
        return task

    async def _task_by_key(self, org_id: uuid.UUID, key: str) -> AgentTask | None:
        return await self._db.scalar(
            select(AgentTask).where(
                AgentTask.organization_id == org_id,
                AgentTask.idempotency_key == key,
            )
        )

    async def get_task(self, org_id: uuid.UUID, task_id: uuid.UUID) -> AgentTask | None:
        return await self._db.scalar(
            select(AgentTask).where(
                AgentTask.id == task_id, AgentTask.organization_id == org_id
            )
        )

    async def list_tasks(
        self,
        org_id: uuid.UUID,
        *,
        agent_id: uuid.UUID | None = None,
        status: str | None = None,
        work_key: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Page:
        """This tenant's tasks, newest first.

        `agent_id`, `status` and `work_key` are FILTERS applied on top of the
        tenant predicate, never scope: another organization's agent id or work
        key narrows the result to nothing rather than reaching its rows.

        `work_key` matches `task_metadata->>'navisWorkKey'` EXACTLY. Not a
        prefix, not a pattern, not a containment test — a client asking for one
        work stream must never be handed a neighbouring one, and an exact
        equality is also the only shape `ix_agent_tasks_org_work_key` can
        answer with a single index descent.

        An empty string is a filter like any other and matches the rows whose
        key is empty, which is none. That is deliberate: a caller that computed
        a blank key has a bug, and answering it with the tenant's entire task
        list would hide the bug behind a plausible-looking page.

        Ordering is `created_at DESC, id DESC`. The id tiebreak is not
        decoration — two rows written in the same microsecond would otherwise
        page non-deterministically, and a row can be seen twice or missed
        entirely as the offset moves.
        """
        limit = _clamp(limit)
        offset = max(0, int(offset))

        predicates = [AgentTask.organization_id == org_id]
        if agent_id is not None:
            predicates.append(AgentTask.agent_id == agent_id)
        if status is not None:
            predicates.append(AgentTask.status == status)
        if work_key is not None:
            predicates.append(_work_key_expression() == work_key)

        total = await self._db.scalar(
            select(func.count()).select_from(AgentTask).where(*predicates)
        )
        rows = (
            await self._db.scalars(
                select(AgentTask)
                .where(*predicates)
                .order_by(AgentTask.created_at.desc(), AgentTask.id.desc())
                .limit(limit)
                .offset(offset)
            )
        ).all()
        return Page(rows=list(rows), total=int(total or 0), limit=limit, offset=offset)

    async def close_task(
        self, org_id: uuid.UUID, task_id: uuid.UUID, *, status: str
    ) -> AgentTask | None:
        """Mark the work itself finished or abandoned.

        Deliberately NOT derived from the runs: three failed attempts do not make
        a task failed, because the fourth may succeed.
        """
        if status not in {TaskStatus.COMPLETED.value, TaskStatus.ABANDONED.value}:
            raise LifecycleError("a task closes as completed or abandoned")

        task = await self.get_task(org_id, task_id)
        if task is None:
            return None
        task.status = status
        await self._db.flush()
        return task

    # ── Run ─────────────────────────────────────────────────────────────

    async def start_run(
        self,
        org_id: uuid.UUID,
        *,
        task_id: uuid.UUID,
        context_snapshot_id: str | None = None,
        context_version: str | None = None,
        correlation_id: str | None = None,
        idempotency_key: str | None = None,
        created_by: uuid.UUID | None = None,
        metadata: dict | None = None,
    ) -> AgentRun:
        """Open one attempt at a task.

        THE AGENT IS NOT A PARAMETER. It is read from the task, so a run can
        never belong to a different agent than the work it attempts — an
        invariant that was previously only true because every caller happened to
        pass the matching id. The task is fetched with the tenant in the query,
        so a cross-tenant task id is a 404 and never a partial write.
        """
        task = await self.get_task(org_id, task_id)
        if task is None:
            raise UnknownAgent("no such task in this organization")
        agent_id = task.agent_id

        if idempotency_key is not None:
            existing = await self._run_by_key(org_id, idempotency_key)
            if existing is not None:
                return existing

        run = AgentRun(
            organization_id=org_id,
            task_id=task_id,
            agent_id=agent_id,
            status=RunStatus.RUNNING.value,
            started_at=_utcnow(),
            context_snapshot_id=context_snapshot_id,
            context_version=context_version,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            created_by=created_by,
            run_metadata=sanitize_metadata(metadata, field="metadata"),
        )
        self._db.add(run)
        try:
            await self._db.flush()
        except IntegrityError:
            await self._db.rollback()
            if idempotency_key is not None:
                existing = await self._run_by_key(org_id, idempotency_key)
                if existing is not None:
                    return existing
            raise
        return run

    async def _run_by_key(self, org_id: uuid.UUID, key: str) -> AgentRun | None:
        return await self._db.scalar(
            select(AgentRun).where(
                AgentRun.organization_id == org_id, AgentRun.idempotency_key == key
            )
        )

    async def get_run(self, org_id: uuid.UUID, run_id: uuid.UUID) -> AgentRun | None:
        return await self._db.scalar(
            select(AgentRun).where(AgentRun.id == run_id, AgentRun.organization_id == org_id)
        )

    async def list_runs(
        self,
        org_id: uuid.UUID,
        *,
        agent_id: uuid.UUID | None = None,
        task_id: uuid.UUID | None = None,
        status: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Page:
        """This tenant's runs, newest first. Same filter/scope rule as tasks."""
        limit = _clamp(limit)
        offset = max(0, int(offset))

        predicates = [AgentRun.organization_id == org_id]
        if agent_id is not None:
            predicates.append(AgentRun.agent_id == agent_id)
        if task_id is not None:
            predicates.append(AgentRun.task_id == task_id)
        if status is not None:
            predicates.append(AgentRun.status == status)

        total = await self._db.scalar(
            select(func.count()).select_from(AgentRun).where(*predicates)
        )
        rows = (
            await self._db.scalars(
                select(AgentRun)
                .where(*predicates)
                .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
                .limit(limit)
                .offset(offset)
            )
        ).all()
        return Page(rows=list(rows), total=int(total or 0), limit=limit, offset=offset)

    async def finish_run(
        self,
        org_id: uuid.UUID,
        run_id: uuid.UUID,
        *,
        status: str,
        failure_category: str | None = None,
        failure_reason: str | None = None,
    ) -> AgentRun | None:
        """Move a run to a terminal state.

        The state machine lives in `lifecycle.ALLOWED_RUN_TRANSITIONS` as DATA,
        and this enforces it. A terminal run has no outgoing edges at all, which
        is what stops a completed run being re-opened and its timeline rewritten
        — an execution ledger that can be edited is not a ledger.
        """
        run = await self.get_run(org_id, run_id)
        if run is None:
            return None

        # ── FINISHING TWICE THE SAME WAY IS A NO-OP, NOT A CONFLICT ────────
        # `ALLOWED_RUN_TRANSITIONS` gives a terminal run no outgoing edges, so a
        # REPEAT of the state it already holds was refused with 409 alongside
        # genuine rewrites. Those are not the same event and must not share an
        # answer.
        #
        # The caller that hits this is the one that cannot avoid it. A session
        # closes its run as the process is torn down, which is exactly when a
        # response is most likely to be lost — and the documented remedy for a
        # lost response is to retry. Answering the retry with a conflict tells a
        # client its shutdown failed when the ledger already says what the
        # client was trying to make it say, and the only way to distinguish the
        # two is an extra read on the path with the least time to spend.
        #
        # So the operation is idempotent in the precise sense: the same terminal
        # status returns the existing row unchanged. A DIFFERENT terminal status
        # is still refused, because `completed` becoming `failed` is a rewrite
        # of history and an execution ledger that can be edited is not a ledger.
        # `completed_at` is deliberately NOT restamped — it records when the run
        # ended, not when somebody last said so.
        if run.status == status and status in _TERMINAL_RUN_STATUSES:
            return run

        if not may_transition(run.status, status):
            raise InvalidTransition(f"a {run.status} run cannot become {status}")

        run.status = status
        run.completed_at = _utcnow()
        if failure_category is not None:
            run.failure_category = failure_category
        if failure_reason is not None:
            # Scrubbed: an exception message is a classic place for a connection
            # string or a token to appear.
            run.failure_reason = sanitize_text(
                failure_reason, field="failure_reason", limit=500
            )
        await self._db.flush()
        return run

    # ── Steps ───────────────────────────────────────────────────────────

    async def record_step(
        self,
        org_id: uuid.UUID,
        run_id: uuid.UUID,
        *,
        step_type: str,
        status: str = StepStatus.COMPLETED.value,
        label: str = "",
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        tool_name: str | None = None,
        tool_kind: str | None = None,
        model_provider: str | None = None,
        model_name: str | None = None,
        model_generation_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        cost_micros: int | None = None,
        cost_currency: str | None = None,
        cost_basis: str | None = None,
        pricing_version: str | None = None,
        context_items: int | None = None,
        context_tokens: int | None = None,
        retrieval_count: int | None = None,
        latency_ms: int | None = None,
        error_category: str | None = None,
        error_message: str | None = None,
        request_metadata: dict | None = None,
        response_metadata: dict | None = None,
    ) -> AgentRunStep:
        """Append one event to a run's timeline.

        THE SEQUENCE IS ASSIGNED HERE, NOT BY THE CALLER. A client-supplied
        position would let one writer interleave itself into another's ordering,
        and would make the unique constraint a source of 500s rather than a
        guarantee. The next position is `MAX(sequence) + 1` for this run.

        Appending to a TERMINAL run is refused. A finished run's timeline is
        history; allowing a late step would mean the record of what happened
        depends on when the reader looked.
        """
        run = await self.get_run(org_id, run_id)
        if run is None:
            raise UnknownAgent("no such run in this organization")
        if run.status in {
            RunStatus.COMPLETED.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
        }:
            raise InvalidTransition("cannot add a step to a finished run")

        count = await self._db.scalar(
            select(func.count()).select_from(AgentRunStep).where(AgentRunStep.run_id == run_id)
        )
        if int(count if count is not None else 0) >= MAX_STEPS_PER_RUN:
            raise StepLimitReached(
                f"this run already holds {MAX_STEPS_PER_RUN} steps, the maximum recorded"
            )

        # A COST MUST SAY WHERE IT CAME FROM — checked here as well as at the
        # edge and in the database, because this service has callers that are not
        # the router (a backfill, an evaluator, the benchmark harness) and an
        # unattributed currency figure is the one value this domain must never
        # store. The database CHECK would catch it too, as an opaque 500; this
        # raises the refusal the caller can actually read.
        if cost_micros is not None and cost_basis is None:
            raise LifecycleError(
                "cost_basis is required when cost_micros is supplied — say whether "
                "the figure is `estimated` or `provider_reported`"
            )

        # `is not None`, NOT `or`. `coalesce(max(sequence), -1)` returns 0 for a
        # run whose only step is at position 0, and `0 or -1` is -1 in Python —
        # so every step after the first was assigned sequence 0 and collided on
        # `uq_agent_run_steps_run_sequence`. The unique constraint turned a
        # silent ordering bug into a loud one, which is why it exists.
        highest = await self._db.scalar(
            select(func.max(AgentRunStep.sequence)).where(AgentRunStep.run_id == run_id)
        )
        next_sequence = 0 if highest is None else int(highest) + 1

        step = AgentRunStep(
            organization_id=org_id,
            run_id=run_id,
            sequence=next_sequence,
            step_type=step_type,
            status=status,
            label=sanitize_text(label, field="label", limit=MAX_STEP_LABEL),
            started_at=started_at,
            completed_at=completed_at,
            tool_name=(
                sanitize_text(tool_name, field="tool_name", limit=MAX_TOOL_NAME)
                if tool_name is not None
                else None
            ),
            tool_kind=tool_kind,
            model_provider=model_provider,
            model_name=model_name,
            model_generation_id=model_generation_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_micros=cost_micros,
            cost_currency=cost_currency,
            cost_basis=cost_basis,
            pricing_version=pricing_version,
            context_items=context_items,
            context_tokens=context_tokens,
            retrieval_count=retrieval_count,
            latency_ms=latency_ms,
            error_category=error_category,
            error_message=(
                sanitize_text(error_message, field="error_message", limit=500)
                if error_message is not None
                else None
            ),
            request_metadata=sanitize_metadata(request_metadata, field="request_metadata"),
            response_metadata=sanitize_metadata(response_metadata, field="response_metadata"),
        )
        self._db.add(step)
        await self._db.flush()
        return step

    async def list_steps(
        self,
        org_id: uuid.UUID,
        run_id: uuid.UUID,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Page | None:
        """One run's timeline, in sequence order.

        `None` when the run is not this tenant's — the route turns that into 404,
        so a cross-tenant run id reveals nothing, not even that it exists.

        Ordered by `sequence` ASC, which the unique constraint makes total: no
        two steps of a run share a position, so the timeline renders the same way
        every time it is read.
        """
        if await self.get_run(org_id, run_id) is None:
            return None

        limit = _clamp(limit)
        offset = max(0, int(offset))

        predicates = [AgentRunStep.organization_id == org_id, AgentRunStep.run_id == run_id]
        total = await self._db.scalar(
            select(func.count()).select_from(AgentRunStep).where(*predicates)
        )
        rows = (
            await self._db.scalars(
                select(AgentRunStep)
                .where(*predicates)
                .order_by(AgentRunStep.sequence.asc())
                .limit(limit)
                .offset(offset)
            )
        ).all()
        return Page(rows=list(rows), total=int(total or 0), limit=limit, offset=offset)

    # ── Outcome ─────────────────────────────────────────────────────────

    async def record_outcome(
        self,
        org_id: uuid.UUID,
        run_id: uuid.UUID,
        *,
        status: str,
        summary: str = "",
        result_reference: str | None = None,
        quality_score: float | None = None,
        human_corrections: int | None = None,
        produced_at: datetime | None = None,
        metadata: dict | None = None,
        if_newer: bool = False,
    ) -> AgentRunOutcome | None:
        """Record what came of a run. One per run; a second call UPDATES it.

        Updating rather than inserting a second row is deliberate: the unique
        constraint means a run has one verdict, and an evaluator that scores a
        result later is revising that verdict, not adding a contradictory one.

        ── `if_newer`: COMPARE-AND-SWAP ON THE REALITY CLOCK ───────────────
        Replacing the row wholesale is safe between SESSIONS, because a session
        owns one run and `UNIQUE(run_id)` stops it reaching another's. It is
        NOT safe within one run. Two writers on the same run — a compaction
        checkpoint and an explicit one, a retry that overtakes the request it
        retried — are ordered by arrival, and arrival order is not authorship
        order. The older statement lands last and silently replaces the newer
        one, taking the append-only sections with it.

        `if_newer=True` refuses that write instead: when the stored row already
        carries a `produced_at` at or after the incoming one, nothing is
        modified and `StaleWrite` is raised. This is the optimistic-concurrency
        rule durable-execution engines use — Temporal fences a reacquired
        history shard with a monotonic RangeID so a displaced owner's writes
        fail OCC rather than split-brain — applied to the clock this domain
        already has. No new column, no revision counter, no migration: the
        version IS `produced_at`, because `produced_at` is already what
        readers order by, and a version readers ignore would be decoration.

        DEFAULT OFF, so every existing caller is unaffected. An evaluator
        revising a verdict long after the fact, or a backfill writing history,
        is not a stale writer and must not be refused. Only a caller that
        claims monotonic authorship asks to be held to it.

        A write with no `produced_at` is never refused: it makes no claim about
        when it was observed, so there is nothing to compare.
        """
        run = await self.get_run(org_id, run_id)
        if run is None:
            return None

        existing = await self._db.scalar(
            select(AgentRunOutcome).where(
                AgentRunOutcome.run_id == run_id,
                AgentRunOutcome.organization_id == org_id,
            )
        )

        if existing is not None and loses_to_stored_outcome(
            stored=existing.produced_at, incoming=produced_at, if_newer=if_newer
        ):
            raise StaleWrite(
                "a newer outcome is already recorded for this run "
                f"(stored produced_at {existing.produced_at.isoformat()} "
                f"is at or after the submitted {produced_at.isoformat()})"
            )

        summary = sanitize_text(summary, field="summary", limit=4000)
        scrubbed = sanitize_metadata(metadata, field="metadata")

        if existing is not None:
            existing.status = status
            existing.summary = summary
            existing.result_reference = result_reference
            existing.quality_score = quality_score
            existing.human_corrections = human_corrections
            existing.produced_at = produced_at
            existing.outcome_metadata = scrubbed
            await self._db.flush()
            return existing

        outcome = AgentRunOutcome(
            organization_id=org_id,
            run_id=run_id,
            # Denormalized from the run, never from the caller: "this agent's
            # outcomes" must not be assertable by whoever writes the outcome.
            agent_id=run.agent_id,
            status=status,
            summary=summary,
            result_reference=result_reference,
            quality_score=quality_score,
            human_corrections=human_corrections,
            produced_at=produced_at,
            outcome_metadata=scrubbed,
        )
        self._db.add(outcome)
        await self._db.flush()
        return outcome

    async def get_outcome(
        self, org_id: uuid.UUID, run_id: uuid.UUID
    ) -> AgentRunOutcome | None:
        return await self._db.scalar(
            select(AgentRunOutcome).where(
                AgentRunOutcome.run_id == run_id,
                AgentRunOutcome.organization_id == org_id,
            )
        )

    async def latest_task_outcome(
        self, org_id: uuid.UUID, task_id: uuid.UUID
    ) -> tuple[AgentRunOutcome, AgentRun] | None:
        """The newest outcome CARRYING CONTENT across every run of one task.

        ── WHY THIS EXISTS: a scan depth is not a retrieval model ──────────
        Without it, a client asking "what is the latest state of this work"
        must list the task's runs and then fetch outcomes one run at a time,
        because `run_id` is the only address an outcome has. Every such client
        ends up choosing a scan depth, and every scan depth is wrong twice: too
        shallow and a real checkpoint under a run of quiet sessions reads as
        ABSENT — silently, and indistinguishably from work that never existed —
        too deep and a hook spends a developer's startup on empty probes. The
        developer CLI had settled on 8/16/32, which reaches 56 runs and loses
        anything beneath. Raising it to 128 moves the cliff; it does not remove
        it.

        The invariant the client actually needs is one the database can state
        exactly, so it is stated here instead of approximated there.

        ── ORDERING IS THE CONTRACT, NOT AN IMPLEMENTATION DETAIL ──────────
        `produced_at DESC NULLS LAST, updated_at DESC, run_id DESC` — reality
        time first, row time second, id last. It mirrors the client's own
        `selectCurrentCheckpoint` rung for rung, and it has to: a server that
        ordered differently would answer a DIFFERENT checkpoint than the paged
        path it replaces, and the two would disagree only for the tenants big
        enough to need this. Reality time leads because a write that lands late
        having observed early must not present itself as the newest state. The
        id tiebreak makes the order total, so two outcomes stamped in the same
        microsecond cannot swap between two calls.

        ── AN EMPTY SUMMARY IS SKIPPED, DELIBERATELY ───────────────────────
        An outcome with no summary states nothing about where work stands, and
        every run this feature opens gets an outcome row only if it checked in.
        Returning a contentless newest row would hide a real checkpoint beneath
        it — the same failure as too shallow a scan, arrived at from the other
        direction. The predicate is applied HERE, on the server, where the
        index is, rather than by a client filtering a page it already paid for.

        ── TENANT ──────────────────────────────────────────────────────────
        The predicate is on BOTH tables, not just the outcome. The join could
        reach a run through `run_id` alone, and `organization_id` on the run is
        then redundant — right up until the day a row is written with a
        mismatched tenant, at which point redundancy is the only thing that
        catches it. `None` for a task that is not this tenant's, which the
        route turns into 404: a cross-tenant task id reveals nothing.

        ── WHY IT IS TIMED ─────────────────────────────────────────────────
        This is the one Developer read whose cost grows with how much an
        organization has DONE, rather than with what it asked for. Measured on
        Postgres 18 with the indexes as shipped, the plan is a full scan of the
        organization's outcomes joined to the task's runs, and it costs 1.3ms
        at 1,000 outcomes, 2.7ms at 10,000 and 11.7ms at 100,000 — linear in
        the organization's total, not in the task's. Nothing about that is
        visible from a request count, so the latency is published here: an
        operator sees the slope arrive instead of inferring it after a
        complaint.

        The fix, when the slope matters, is NOT the ordering index it looks
        like it should be. An index on
        `(organization_id, produced_at DESC NULLS LAST, updated_at DESC,
        run_id DESC)` makes a RECENT task 9x faster and an OLD one 55x slower
        (1.4ms -> 77ms, 382,000 buffers), because it walks newest-first and an
        old task's first match is at the far end — and resuming old work is
        exactly what this route is for. Only putting the task IN the index
        helps uniformly, which means denormalizing `task_id` onto
        `agent_run_outcomes` as `agent_id` already is (0.3ms flat, 4 buffers).

        The four candidates were measured against both the recent-task and the
        ancient-task case; the numbers above are that run's results. No
        migration is written, deliberately: at the current workload this query
        is ~1.3ms, and the series published here is what says when that has
        stopped being true.
        """
        # Existence is checked against the TASK, so "this task has no
        # checkpoint yet" (404 with the task present) and "no such task" are
        # not merged into one answer the caller cannot act on.
        if await self.get_task(org_id, task_id) is None:
            return None

        with timed("developer.latest_outcome_ms"):
            row = (
                await self._db.execute(
                    select(AgentRunOutcome, AgentRun)
                    .join(AgentRun, AgentRun.id == AgentRunOutcome.run_id)
                    .where(
                        AgentRunOutcome.organization_id == org_id,
                        AgentRun.organization_id == org_id,
                        AgentRun.task_id == task_id,
                        AgentRunOutcome.summary != "",
                    )
                    .order_by(
                        nulls_last(AgentRunOutcome.produced_at.desc()),
                        AgentRunOutcome.updated_at.desc(),
                        AgentRunOutcome.run_id.desc(),
                    )
                    .limit(1)
                )
            ).first()

        return None if row is None else (row[0], row[1])

    # ── Telemetry rollup ────────────────────────────────────────────────

    async def get_run_telemetry(
        self, org_id: uuid.UUID, run_id: uuid.UUID
    ) -> RunTelemetry | None:
        """One run's facts, aggregated — the ONLY place this arithmetic lives.

        Every figure here is derived from rows the same caller could already read
        through `/runs/{id}`, `/runs/{id}/steps` and `/runs/{id}/outcome` with the
        same permission. It adds no fact and no authority; it exists so the four
        Developer surfaces do not each write their own summation and disagree.

        THE WHOLE TIMELINE IS LOADED, not a page of it. A partial rollup would be
        a wrong number rather than an incomplete one — a cost total over the first
        fifty of two hundred steps is not "most of" the cost, it is a figure that
        looks authoritative and is not. `MAX_STEPS_PER_RUN` bounds this at a
        thousand rows, which is why reading them all is affordable.
        """
        run = await self.get_run(org_id, run_id)
        if run is None:
            return None

        task = await self.get_task(org_id, run.task_id)
        steps = list(
            (
                await self._db.scalars(
                    select(AgentRunStep)
                    .where(
                        AgentRunStep.organization_id == org_id,
                        AgentRunStep.run_id == run_id,
                    )
                    .order_by(AgentRunStep.sequence.asc())
                )
            ).all()
        )
        outcome = await self.get_outcome(org_id, run_id)

        # Which attempt at the task this is. A "retry count" is not stored
        # because the ledger already holds it structurally: attempt N is the Nth
        # run of the task, ordered by creation. A counter would be a second
        # source for the same fact, free to disagree with the rows.
        sibling_runs = list(
            (
                await self._db.scalars(
                    select(AgentRun.id)
                    .where(
                        AgentRun.organization_id == org_id,
                        AgentRun.task_id == run.task_id,
                    )
                    .order_by(AgentRun.created_at.asc(), AgentRun.id.asc())
                )
            ).all()
        )
        attempt = sibling_runs.index(run.id) + 1 if run.id in sibling_runs else 1

        return RunTelemetry(
            run=run,
            task_description=task.description if task is not None else "",
            attempt=attempt,
            task_attempts=len(sibling_runs) or 1,
            steps=steps,
            outcome=outcome,
        )

    async def list_run_economics(
        self,
        org_id: uuid.UUID,
        *,
        task_id: uuid.UUID | None = None,
        agent_id: uuid.UUID | None = None,
        model_name: str | None = None,
        context_version: str | None = None,
        limit: int | None = None,
    ) -> list[RunEconomics]:
        """Every run's economics, reduced in SQL, for cross-run grouping.

        ONE STATEMENT, and the reason is the shape of the data rather than a
        preference. A run holds up to `MAX_STEPS_PER_RUN` steps; loading the
        timeline of every run to sum it in Python — which is what
        `get_run_telemetry` correctly does for ONE run — would be a million rows
        to produce a dozen numbers. The per-step sums are therefore aggregated
        by the database and only one row per run crosses the boundary.

        NO FACT AND NO AUTHORITY IS ADDED. Every column here is already readable
        through `/runs`, `/runs/{id}/steps` and `/runs/{id}/outcome` under the
        same permission; this groups them. The arithmetic deliberately mirrors
        `RunTelemetryOut.of` — estimated and provider-reported accumulate
        separately, a NULL `cost_micros` counts as an uncosted generation rather
        than as zero — because two places that sum the same rows differently is
        how a platform starts disagreeing with itself.

        TENANT FILTER ON EVERY TABLE, not only on the run. `agent_run_steps` and
        `agent_run_outcomes` both carry `organization_id` precisely so a join
        cannot widen the tenant, and both are constrained here even though the
        run join already implies it — the redundancy is what catches a future
        join written the other way round.
        """
        est = func.sum(
            case(
                (
                    and_(
                        AgentRunStep.cost_micros.isnot(None),
                        AgentRunStep.cost_basis != CostBasis.PROVIDER_REPORTED.value,
                    ),
                    AgentRunStep.cost_micros,
                ),
                else_=0,
            )
        )
        rep = func.sum(
            case(
                (
                    and_(
                        AgentRunStep.cost_micros.isnot(None),
                        AgentRunStep.cost_basis == CostBasis.PROVIDER_REPORTED.value,
                    ),
                    AgentRunStep.cost_micros,
                ),
                else_=0,
            )
        )
        # Counted, never treated as zero: a model step with no cost figure is
        # missing evidence, and a total computed around it is incomplete rather
        # than cheap. `EconomicsOut` already reports this per run.
        uncosted = func.count(
            case(
                (
                    and_(
                        AgentRunStep.step_type == StepType.MODEL_GENERATION.value,
                        AgentRunStep.cost_micros.is_(None),
                    ),
                    1,
                )
            )
        )
        costed = func.count(AgentRunStep.cost_micros)
        fetches = func.count(
            case((AgentRunStep.step_type == StepType.CONTEXT_FETCH.value, 1))
        )

        stmt = (
            select(
                AgentRun.id,
                AgentRun.task_id,
                AgentTask.description,
                AgentRun.agent_id,
                AgentRun.context_version,
                est.label("estimated"),
                rep.label("reported"),
                costed.label("costed"),
                uncosted.label("uncosted"),
                fetches.label("fetches"),
                func.sum(func.coalesce(AgentRunStep.latency_ms, 0)).label("latency_ms"),
                func.array_agg(distinct(AgentRunStep.cost_currency)).label("currencies"),
                func.array_agg(distinct(AgentRunStep.pricing_version)).label("pricing"),
                func.array_agg(distinct(AgentRunStep.model_name)).label("models"),
                AgentRunOutcome.status,
                AgentRunOutcome.quality_score,
                AgentRunOutcome.human_corrections,
            )
            .join(AgentTask, AgentTask.id == AgentRun.task_id)
            # OUTER, because a run with no steps is still an attempt: it
            # consumed nothing and may still have produced an outcome, and
            # dropping it would quietly shrink the denominator.
            .outerjoin(
                AgentRunStep,
                and_(
                    AgentRunStep.run_id == AgentRun.id,
                    AgentRunStep.organization_id == org_id,
                ),
            )
            # OUTER for the same reason, and because `unknown` is the common
            # case — an unjudged run is evidence of an attempt's cost.
            .outerjoin(
                AgentRunOutcome,
                and_(
                    AgentRunOutcome.run_id == AgentRun.id,
                    AgentRunOutcome.organization_id == org_id,
                ),
            )
            .where(AgentRun.organization_id == org_id, AgentTask.organization_id == org_id)
            .group_by(
                AgentRun.id,
                AgentRun.task_id,
                AgentTask.description,
                AgentRun.agent_id,
                AgentRun.context_version,
                AgentRunOutcome.status,
                AgentRunOutcome.quality_score,
                AgentRunOutcome.human_corrections,
            )
        )
        if task_id is not None:
            stmt = stmt.where(AgentRun.task_id == task_id)
        if agent_id is not None:
            stmt = stmt.where(AgentRun.agent_id == agent_id)
        if context_version is not None:
            stmt = stmt.where(AgentRun.context_version == context_version)
        if model_name is not None:
            # EXISTS rather than a predicate on the joined step: filtering the
            # join would also remove that run's OTHER steps from the sums, so a
            # multi-model run would report a fraction of its own cost.
            #
            # ALIASED, and it has to be. `AgentRunStep` is already in the outer
            # FROM through the join above, so an un-aliased subquery correlates
            # against that same table and SQLAlchemy is left with no FROM clause
            # at all — `InvalidRequestError: returned no FROM clauses due to
            # auto-correlation`. The alias makes this a second, independent
            # reference to the table.
            probe = aliased(AgentRunStep)
            stmt = stmt.where(
                exists(
                    select(1)
                    .select_from(probe)
                    .where(
                        probe.run_id == AgentRun.id,
                        probe.organization_id == org_id,
                        probe.model_name == model_name,
                    )
                )
            )
        # BOUNDED SCAN, and the caller is told when it bit. A rollup computed
        # over a truncated set of runs is a wrong number rather than a partial
        # one — the same reasoning `get_run_telemetry` gives for loading a whole
        # timeline — so the ceiling is high, ordered deterministically, and the
        # schema reports `runs_truncated` when it is reached.
        stmt = stmt.order_by(AgentRun.id).limit((limit or MAX_ECONOMICS_RUNS) + 1)

        def _clean(values) -> frozenset[str]:
            return frozenset(v for v in (values or []) if v)

        out: list[RunEconomics] = []
        for row in (await self._db.execute(stmt)).all():
            out.append(
                RunEconomics(
                    task_id=row.task_id,
                    task_description=row.description or "",
                    agent_id=row.agent_id,
                    context_fetched=bool(row.fetches),
                    context_version=row.context_version,
                    # NULL, not 0, when nothing was costed — the distinction
                    # between "cost nothing" and "we do not know" is the one
                    # `EconomicsOut` exists to preserve.
                    estimated_micros=int(row.estimated) if row.costed and row.estimated else None,
                    reported_micros=int(row.reported) if row.costed and row.reported else None,
                    uncosted_generations=int(row.uncosted or 0),
                    currencies=_clean(row.currencies),
                    pricing_versions=_clean(row.pricing),
                    outcome_status=row.status,
                    quality_score=row.quality_score,
                    human_corrections=row.human_corrections,
                    latency_ms=int(row.latency_ms) if row.latency_ms is not None else None,
                    models=_clean(row.models),
                )
            )
        return out


__all__ = [
    "AgentLifecycleService",
    "InvalidTransition",
    "LifecycleError",
    "Page",
    "PayloadTooLarge",
    "RunEconomics",
    "RunTelemetry",
    "StaleWrite",
    "StepLimitReached",
    "UnknownAgent",
    "loses_to_stored_outcome",
]
