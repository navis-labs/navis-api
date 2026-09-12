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

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from navis.developer.lifecycle import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MAX_STEP_LABEL,
    MAX_STEPS_PER_RUN,
    MAX_TASK_DESCRIPTION,
    MAX_TOOL_NAME,
    RunStatus,
    StepStatus,
    TaskStatus,
    may_transition,
)
from navis.developer.run_models import AgentRun, AgentRunOutcome, AgentRunStep, AgentTask
from navis.developer.sanitize import PayloadTooLarge, sanitize_metadata, sanitize_text
from navis.identity.enterprise_models import ServiceAccount


class LifecycleError(Exception):
    """A refusal the caller can act on. Routes map it to 400 or 409."""


class UnknownAgent(LifecycleError):
    """The named agent does not exist IN THIS TENANT. Routes map it to 404."""


class InvalidTransition(LifecycleError):
    """A run was asked to move somewhere its current status does not allow."""


class StepLimitReached(LifecycleError):
    """This run already holds as many steps as the platform will record."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clamp(limit: int | None) -> int:
    """Bound every page size. `None` means "caller did not say", not "all rows".

    An unbounded list endpoint is the way a shared table becomes an outage, so
    there is no code path here that omits a LIMIT.
    """
    if limit is None:
        return DEFAULT_PAGE_SIZE
    return max(1, min(int(limit), MAX_PAGE_SIZE))


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
        limit: int | None = None,
        offset: int = 0,
    ) -> Page:
        """This tenant's tasks, newest first.

        `agent_id` and `status` are FILTERS applied on top of the tenant
        predicate, never scope: another organization's agent id narrows the
        result to nothing rather than reaching its rows.

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
    ) -> AgentRunOutcome | None:
        """Record what came of a run. One per run; a second call UPDATES it.

        Updating rather than inserting a second row is deliberate: the unique
        constraint means a run has one verdict, and an evaluator that scores a
        result later is revising that verdict, not adding a contradictory one.
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


__all__ = [
    "AgentLifecycleService",
    "InvalidTransition",
    "LifecycleError",
    "Page",
    "PayloadTooLarge",
    "RunTelemetry",
    "StepLimitReached",
    "UnknownAgent",
]
