"""Agent execution persistence — Task, Run, Step, Outcome.

FOUR ADDITIVE TABLES. Nothing here alters an existing table, and nothing here is
a second identity, organization or authorization model: `agent_id` is a
`service_accounts.id` and `organization_id` is an `organizations.id`, both
resolved from the caller's credential and never accepted from a client.

WHY `organization_id` IS ON EVERY TABLE, INCLUDING THE CHILDREN. A step could
reach its tenant through `run -> task -> organization`, and denormalizing looks
redundant. It is not. Every list and every lookup in this domain is
tenant-scoped, and a two-join path to the tenant predicate means the planner has
to reach the parent before it can exclude another tenant's rows — so the
isolation guarantee depends on a join being written correctly at every call site.
With the column present, `WHERE organization_id = :org` is the FIRST predicate on
every query and every index leads with it. A tenant filter that can be forgotten
is a tenant filter that eventually is.

TIMESTAMPS. `Timestamped` gives `created_at`/`updated_at` on the mutable rows.
The lifecycle timestamps (`started_at`, `completed_at`) are separate and
nullable, because "when did the work happen" is not "when was the row written" —
an agent may report a run after it finished, and conflating the two makes every
duration wrong.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import text as sa_text

from navis.developer.lifecycle import (
    COST_BASES,
    ERROR_CATEGORIES,
    MAX_STEP_LABEL,
    MAX_TASK_DESCRIPTION,
    MAX_TOOL_NAME,
    OUTCOME_STATUSES,
    RUN_STATUSES,
    STEP_STATUSES,
    STEP_TYPES,
    TASK_STATUSES,
    TOOL_KINDS,
)
from navis.platform.database.base import Base, Timestamped, UUIDPrimaryKey

_JSONB_DICT = MutableDict.as_mutable(JSONB)


def _one_of(column: str, values: frozenset[str]) -> str:
    """A SQL `IN (...)` over a closed vocabulary, sorted for stable DDL.

    Same helper as `identity/enterprise_models.py`; duplicated rather than
    imported so this module keeps its "no cross-domain imports" property. Sorted
    so Alembic's autogenerate does not see a difference on every run.
    """
    literals = ", ".join(f"'{v}'" for v in sorted(values))
    return f"{column} IN ({literals})"


def _nullable_one_of(column: str, values: frozenset[str]) -> str:
    """As `_one_of`, but a NULL is allowed — for optional closed vocabularies."""
    literals = ", ".join(f"'{v}'" for v in sorted(values))
    return f"{column} IS NULL OR {column} IN ({literals})"


class AgentTask(UUIDPrimaryKey, Timestamped, Base):
    """WHAT was asked of an agent. Survives the attempts at it.

    A task is intent, not execution. It exists so three retries are three runs of
    ONE task rather than three unrelated events — which is the whole reason
    "did this eventually work?" is answerable.
    """

    __tablename__ = "agent_tasks"
    __table_args__ = (
        CheckConstraint(_one_of("status", TASK_STATUSES), name="ck_agent_tasks_status"),
        # The hot list: this tenant's tasks, newest first. Leading with the
        # tenant keeps one organization's volume off another's scan.
        Index("ix_agent_tasks_org_created", "organization_id", sa_text("created_at DESC")),
        # "What has this agent been asked to do" — the agent detail surface.
        Index("ix_agent_tasks_agent_created", "agent_id", sa_text("created_at DESC")),
        # ── The continuity lookup: "is there open work under this key" ─────
        #
        # ONE metadata key is indexed, and only one. `navisWorkKey` is written
        # by a client that needs to find its own task again across processes —
        # a developer tool reopening the work stream for a worktree — and that
        # question has no other answer here: a task is addressable by id, and
        # the id is precisely what a fresh process does not have.
        #
        # Without this the caller must PAGE the organization's open tasks and
        # match the key itself, which stops working the moment the tenant has
        # more open tasks than the client's window. The work stream does not
        # error then; it silently reads as absent, and a second one is opened
        # beside the first.
        #
        # An EXPRESSION B-TREE, not GIN. The query is a single equality on one
        # known key, which is what a b-tree answers in one descent; GIN over
        # the whole document would index every key to serve one, and could not
        # lead with the tenant. Leading with `organization_id` keeps one
        # organization's volume off another's scan, exactly as above.
        #
        # Tasks without the key index as NULL and cost a row in the index
        # apiece. Deliberately not partial (`WHERE ... IS NOT NULL`): a partial
        # index is only usable when the planner can prove the predicate, and
        # the saving on a table whose rows are mostly keyed is not worth the
        # plan fragility.
        Index(
            "ix_agent_tasks_org_work_key",
            "organization_id",
            sa_text("(task_metadata ->> 'navisWorkKey')"),
        ),
        # Idempotent creation. A client that retries a POST with the same key
        # gets the same task rather than a duplicate. Scoped to the tenant so
        # one organization's key choice can never collide with another's.
        UniqueConstraint(
            "organization_id", "idempotency_key", name="uq_agent_tasks_org_idempotency"
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    #: The `service_accounts` row this task was asked of. CASCADE because an
    #: agent's history is meaningless once the identity is gone, and because a
    #: tenant deletion must not leave orphans behind.
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("service_accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    description: Mapped[str] = mapped_column(String(MAX_TASK_DESCRIPTION), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="open", server_default="open"
    )
    #: Caller-supplied dedupe key. NULL means "no idempotency requested" — and
    #: because NULLs do not collide in a UNIQUE index, unkeyed creates stay
    #: independent while keyed ones are exactly-once per tenant.
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    #: Who created it: a member's user id, or the agent's own service-account id.
    #: Attribution only, never authority.
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    #: Bounded, scrubbed. See `sanitize.py`.
    task_metadata: Mapped[dict] = mapped_column(
        _JSONB_DICT, nullable=False, default=dict, server_default=sa_text("'{}'::jsonb")
    )


class AgentRun(UUIDPrimaryKey, Timestamped, Base):
    """ONE attempt at a task.

    `context_snapshot_id` is the provenance link and the reason this table earns
    its place: it pins WHICH view of the organization the agent was working from.
    Without it, "why did the agent do that?" is unanswerable six weeks later.
    """

    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint(_one_of("status", RUN_STATUSES), name="ck_agent_runs_status"),
        CheckConstraint(
            _nullable_one_of("failure_category", ERROR_CATEGORIES),
            name="ck_agent_runs_failure_category",
        ),
        # A completed run must say when it completed, and a run that has not
        # completed must not claim to have. The database holds this because a
        # duration computed from a missing timestamp is silently wrong.
        CheckConstraint(
            "(status IN ('completed','failed','cancelled')) = (completed_at IS NOT NULL)",
            name="ck_agent_runs_completed_at",
        ),
        Index("ix_agent_runs_org_created", "organization_id", sa_text("created_at DESC")),
        Index("ix_agent_runs_agent_created", "agent_id", sa_text("created_at DESC")),
        # The task detail view: every attempt at one task, oldest first is what
        # a reader wants, but the index serves either direction.
        Index("ix_agent_runs_task_created", "task_id", sa_text("created_at DESC")),
        # "What is running right now" for one tenant, without scanning history.
        Index("ix_agent_runs_org_status", "organization_id", "status"),
        UniqueConstraint(
            "organization_id", "idempotency_key", name="uq_agent_runs_org_idempotency"
        ),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("service_accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="queued", server_default="queued"
    )

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: Why it failed, in a groupable vocabulary. NULL unless the run failed.
    failure_category: Mapped[str | None] = mapped_column(String(24), nullable=True)
    #: A short operator-facing reason. Scrubbed and bounded like all free text —
    #: an exception message is a classic place for a connection string to appear.
    failure_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)

    #: The context snapshot handed to the agent, if it took one. A REFERENCE, not
    #: a copy: duplicating the projection into every run would put the whole
    #: organizational state in this table many times over.
    context_snapshot_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    #: WHICH VERSION of the context assembly produced that snapshot.
    #:
    #: On the RUN rather than the step, unlike every other telemetry field added
    #: alongside it, because it is provenance rather than measurement — it
    #: belongs next to `context_snapshot_id`, which it qualifies. The numeric
    #: context facts (items, tokens, retrievals) are per-FETCH and live on the
    #: step, because a run may fetch context repeatedly and summing them is the
    #: point; a version is not summable and does not vary within one attempt.
    #:
    #: This is also the column that makes the comparison honest later: grouping
    #: runs by context version is how "did the new context assembly help?" is
    #: asked without re-running the baseline.
    context_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    #: Correlates this run with the caller's own logs and with the request that
    #: created it. Not a secret, and not an authorization input.
    correlation_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    run_metadata: Mapped[dict] = mapped_column(
        _JSONB_DICT, nullable=False, default=dict, server_default=sa_text("'{}'::jsonb")
    )


class AgentRunStep(UUIDPrimaryKey, Base):
    """One observable event in a run's timeline.

    APPEND-ONLY: `created_at` and no `updated_at`. A step may be closed (its
    status and `completed_at` set) but its identity, order and type never change.
    An execution ledger that can be rewritten is not a ledger.

    ONE TABLE FOR EVERY STEP KIND. Tool calls and model generations are step
    TYPES, not separate tables: the Run Detail surface renders a single ordered
    timeline, and three tables would have to be merged and re-sorted on every
    read to produce it.

    THE ECONOMICS COLUMNS ARE TYPED, NOT JSONB. `input_tokens`, `cost_micros` and
    the rest are what "cost per successful task" is computed from, and an
    aggregate over a JSONB field cannot use an index and cannot be summed without
    a cast on every row. They are nullable because they are meaningful only on a
    `model_generation` step.
    """

    __tablename__ = "agent_run_steps"
    __table_args__ = (
        CheckConstraint(_one_of("step_type", STEP_TYPES), name="ck_agent_run_steps_type"),
        CheckConstraint(_one_of("status", STEP_STATUSES), name="ck_agent_run_steps_status"),
        CheckConstraint(
            _nullable_one_of("tool_kind", TOOL_KINDS), name="ck_agent_run_steps_tool_kind"
        ),
        CheckConstraint(
            _nullable_one_of("error_category", ERROR_CATEGORIES),
            name="ck_agent_run_steps_error_category",
        ),
        # Counters are counts. A negative token count would corrupt every
        # aggregate built on this table, silently.
        CheckConstraint(
            "(input_tokens IS NULL OR input_tokens >= 0) AND "
            "(output_tokens IS NULL OR output_tokens >= 0) AND "
            "(cached_tokens IS NULL OR cached_tokens >= 0) AND "
            "(cache_read_tokens IS NULL OR cache_read_tokens >= 0) AND "
            "(cache_write_tokens IS NULL OR cache_write_tokens >= 0) AND "
            "(reasoning_tokens IS NULL OR reasoning_tokens >= 0) AND "
            "(context_items IS NULL OR context_items >= 0) AND "
            "(context_tokens IS NULL OR context_tokens >= 0) AND "
            "(retrieval_count IS NULL OR retrieval_count >= 0) AND "
            "(cost_micros IS NULL OR cost_micros >= 0) AND "
            "(latency_ms IS NULL OR latency_ms >= 0)",
            name="ck_agent_run_steps_counters_non_negative",
        ),
        CheckConstraint(
            _nullable_one_of("cost_basis", COST_BASES), name="ck_agent_run_steps_cost_basis"
        ),
        # A COST MUST SAY WHERE IT CAME FROM. This is the constraint the whole
        # `cost_basis` column exists for: an unattributed currency figure is
        # indistinguishable from a billed one to everybody who reads it later,
        # and the database is the only writer-independent place to hold that.
        # The converse is allowed — a basis with no cost is a caller saying how
        # it WOULD cost something it has not costed, which is harmless.
        CheckConstraint(
            "cost_micros IS NULL OR cost_basis IS NOT NULL",
            name="ck_agent_run_steps_cost_has_basis",
        ),
        CheckConstraint("sequence >= 0", name="ck_agent_run_steps_sequence"),
        # THE ordering guarantee, and the read path. A run's timeline is fetched
        # by run id in sequence order; UNIQUE makes the order deterministic
        # rather than merely usual, so two steps can never share a position and
        # render non-deterministically.
        UniqueConstraint("run_id", "sequence", name="uq_agent_run_steps_run_sequence"),
        Index("ix_agent_run_steps_org_created", "organization_id", sa_text("created_at DESC")),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    #: Position in the timeline, 0-based and dense. Supplied by the service, not
    #: the client, so a caller cannot interleave itself into another's ordering.
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    step_type: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="running", server_default="running"
    )
    #: Short human label for the timeline row: "retrieve billing entity".
    label: Mapped[str] = mapped_column(
        String(MAX_STEP_LABEL), nullable=False, default="", server_default=""
    )

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # ── Tool facts (step_type = tool_call) ──────────────────────────────
    tool_name: Mapped[str | None] = mapped_column(String(MAX_TOOL_NAME), nullable=True)
    tool_kind: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # ── Model facts (step_type = model_generation) ──────────────────────
    #: Provider and model are free-form STRINGS, not enums. Navis does not own
    #: the model, and a closed set here would need a migration every time a
    #: vendor ships a version — which is how a platform ends up unable to record
    #: what its customers are actually running.
    model_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    #: The provider's own id for the generation, where it returns one.
    model_generation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: The LEGACY total: "cached tokens, of whatever kind". Kept because it is
    #: already written and already read, and never reconstructed from the two
    #: columns below — a caller that reports only a total is telling the truth
    #: about what its provider gave it.
    cached_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Cache reads and cache WRITES, split, because they are not the same price.
    #:
    #: This is the column pair the single `cached_tokens` above cannot replace.
    #: Anthropic bills a cache WRITE at ~1.25x the base input rate and a cache
    #: READ at ~0.1x — a ratio of more than twelve to one — and reports them as
    #: two separate usage fields (`cache_creation_input_tokens` and
    #: `cache_read_input_tokens`) precisely because collapsing them loses the
    #: price. A single total makes any cost figure derived from it arithmetic
    #: over an unknown mixture, and "cost per successful task" is the metric this
    #: whole domain exists to produce.
    cache_read_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_write_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Thinking/reasoning tokens where the provider exposes them. Billed as
    #: OUTPUT, and typically already included in `output_tokens` — recorded
    #: separately so "how much of the spend was deliberation" is answerable
    #: without double-counting it into the totals.
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Cost in MILLIONTHS of a currency unit, as an integer. Never a float:
    #: summing floats over millions of rows accumulates error, and money that
    #: does not add up is worse than money that is absent. Currency is carried
    #: alongside because a platform serving many organizations will not have one.
    cost_micros: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    cost_currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    #: WHERE `cost_micros` came from — `estimated` or `provider_reported`.
    #: Required whenever a cost is present, by CHECK below. A provider API
    #: returns token counts, not money, so a currency figure is arithmetic over
    #: a price list until a provider says otherwise; a number that does not say
    #: which it is will eventually be read as an invoice.
    cost_basis: Mapped[str | None] = mapped_column(String(24), nullable=True)
    #: WHICH price list an estimate used, e.g. `anthropic-2026-06-24`. A cost
    #: computed against stale prices is not wrong so much as unattributable, and
    #: without this there is no way to recompute history when prices move.
    pricing_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ── Context facts (step_type = context_fetch) ───────────────────────
    #: How much organizational context this fetch actually supplied.
    #:
    #: TYPED FOR THE SAME REASON THE TOKEN COLUMNS ARE. These are the
    #: INDEPENDENT VARIABLE of the baseline-versus-Navis-context comparison: the
    #: question is whether supplying context changes cost and outcome, and
    #: answering it means summing context tokens per run and grouping by mode.
    #: An aggregate over JSONB cannot use an index and cannot be summed without
    #: a cast on every row.
    #:
    #: METADATA ONLY. The context ITSELF is not copied here — `context_snapshot_
    #: id` on the run is the reference to it. Counting what was supplied is
    #: observability; storing it would make this table a second copy of the
    #: organization's state.
    context_items: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: How many retrievals backed the fetch. Distinct from `context_items`: ten
    #: retrievals yielding three items is a different (and more expensive) shape
    #: than three yielding three.
    retrieval_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Failure ─────────────────────────────────────────────────────────
    error_category: Mapped[str | None] = mapped_column(String(24), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # ── Bounded, scrubbed metadata ──────────────────────────────────────
    #: What went in, what came back, and what context was requested. Each is
    #: capped and passed through `sanitize.py`. Deliberately NOT a transcript
    #: store: raw prompts and completions are the agent's to keep.
    request_metadata: Mapped[dict] = mapped_column(
        _JSONB_DICT, nullable=False, default=dict, server_default=sa_text("'{}'::jsonb")
    )
    response_metadata: Mapped[dict] = mapped_column(
        _JSONB_DICT, nullable=False, default=dict, server_default=sa_text("'{}'::jsonb")
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa_text("now()"), nullable=False
    )


class AgentRunOutcome(UUIDPrimaryKey, Timestamped, Base):
    """What came of a run — separate from whether the run finished.

    A RUN CAN COMPLETE AND STILL BE WRONG. Equating the two makes "was it any
    good?" unaskable, which is exactly the question Learning will need. So the
    outcome carries its own status, and `unknown` is the default: nothing
    evaluates agent outcomes yet, and recording an unjudged result as a success
    would poison the first metric built on this table.

    NOT THE APPLICATION'S `outcomes` TABLE. That one is keyed by a deterministic
    uuid5 of a provider `source_event_id`, carries `execution_id`/`decision_id`
    lineage, and is written by webhooks and the observation service. An agent
    outcome has no source event and none of that lineage; forcing it in would
    mean nullable lineage columns and a synthetic event id on a table the
    application pipeline depends on. The two are joined by Learning later, not by
    a shared primary key now.

    ONE OUTCOME PER RUN, enforced by a unique constraint: a run that produced two
    contradictory verdicts is a bug, not a state to represent.
    """

    __tablename__ = "agent_run_outcomes"
    __table_args__ = (
        CheckConstraint(_one_of("status", OUTCOME_STATUSES), name="ck_agent_run_outcomes_status"),
        CheckConstraint(
            "quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)",
            name="ck_agent_run_outcomes_quality_score",
        ),
        CheckConstraint(
            "human_corrections IS NULL OR human_corrections >= 0",
            name="ck_agent_run_outcomes_human_corrections",
        ),
        UniqueConstraint("run_id", name="uq_agent_run_outcomes_run"),
        Index(
            "ix_agent_run_outcomes_org_created", "organization_id", sa_text("created_at DESC")
        ),
        # "How are this agent's outcomes trending" without touching other agents.
        Index("ix_agent_run_outcomes_agent_status", "agent_id", "status"),
    )

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    #: Denormalized from the run so "this agent's outcomes" is one index scan
    #: rather than a join through runs.
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("service_accounts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="unknown", server_default="unknown"
    )
    #: A short statement of what was produced. Bounded and scrubbed; the artifact
    #: itself lives wherever the agent put it.
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    #: Where the result actually is — a URL, a ticket id, a commit sha. A
    #: reference, because Navis is not the store for the agent's work product.
    result_reference: Mapped[str | None] = mapped_column(String(500), nullable=True)
    #: 0..1, and NULL until something evaluates it. Nullable rather than
    #: defaulted to zero: "not judged" and "judged worthless" are different.
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: How many times a HUMAN had to fix this result.
    #:
    #: NULLABLE, AND THAT IS THE WHOLE DESIGN. `NULL` means nobody has looked;
    #: `0` means somebody looked and it needed nothing. Defaulting to zero would
    #: silently merge "unreviewed" into "perfect", and since this is the
    #: numerator of "human corrections per successful task" it would make an
    #: unreviewed backlog read as flawless work.
    #:
    #: A COUNT rather than a flag because the metric asks how MUCH intervention
    #: a result needed, and one nudge and six rewrites are not the same outcome.
    human_corrections: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Reality time — when the result was produced, which may precede the write.
    produced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    outcome_metadata: Mapped[dict] = mapped_column(
        _JSONB_DICT, nullable=False, default=dict, server_default=sa_text("'{}'::jsonb")
    )
