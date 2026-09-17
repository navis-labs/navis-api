"""BriefOrchestrator — the morning heartbeat.

The pipeline assembly (no business logic of its own):
    State V2 (get_state) -> Reasoning V2 (run_reasoning_v2) -> Decision V2 (decide)

Decision consumes Reasoning only; orchestration is the seam that feeds it.
Exposed both as a coroutine (for the API / tests) and a Celery task (schedule).

It is also the seam that feeds INTELLIGENCE its second input. Intelligence reads
Learning only — its repository imports `navis.learning.models` and nothing else
from the pipeline, pinned by an AST test — so "what is true of this organization
right now" cannot be fetched by Intelligence itself. It is projected here, from
the State and ReasoningSignals this module already holds, and handed in as
primitives. Same shape as the two joins this file already performs: `evidence`
read here and handed to Reasoning, Adaptive policy resolved here and handed to
Decision.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.decisions.base import IntelligenceFact, IntelligenceView
from navis.decisions.models import DailyBrief
from navis.decisions.service import DecisionService
from navis.intelligence.insights import (
    ContextEntityRow,
    ContextLink,
    OrganizationalRealityView,
    RealitySignalRow,
)
from navis.memory.envelope import EntityType
from navis.memory.lineage import classifications_by_entity
from navis.orchestration.runtime_policy import scheduled_work_allowed
from navis.organization.contract import Capability
from navis.platform.database import session_scope
from navis.platform.events import DomainEvent, EventType, event_bus
from navis.platform.monitoring import get_logger, org_context
from navis.platform.monitoring.metrics import metrics
from navis.platform.queue import celery_app, run_async
from navis.reasoning.store import persist_signals
from navis.reasoning.v2 import REASONING_VERSION, run_reasoning_v2
from navis.reasoning.v2.contract import ReasoningSignal
from navis.state.v2 import MorningBrief, StateService, scope_for
from navis.state.v2.models import OrganizationalState
from navis.state.v2.scope import ScopedView

logger = get_logger("orchestration.brief")


# How many memory records / signal events one entity contributes to an insight's
# evidence. `scope.py::_cite` already bounds each entity to its newest 10; an
# insight names up to 20 entities, so a second bound here keeps one JSONB blob to
# a readable size. The newest are kept — a fact about an entity is a description
# of a moment, and the recent moments are the ones being reasoned about.
_MAX_LINEAGE_PER_ENTITY = 3


def _context_rows(view: ScopedView) -> tuple[ContextEntityRow, ...]:
    """Flatten CONTEXT into the primitive rows Intelligence accepts.

    This is the fifth join at this seam and the same shape as the other four:
    read here, flatten to primitives, hand in. `navis/intelligence/**` imports no
    State and gains none — it receives strings.

    WHAT IT CARRIES AND WHY. `scope_for` already ranks the entities relevant to
    the morning-brief intent and already resolves each one's citations from its
    own `provenance` — `memory_ids` and `signal_ids`, the two hops the insight
    provenance chain was missing (`intelligence.service._provenance` documents
    the break in its own docstring). Nothing is re-derived and nothing is
    fetched: this reshapes a view the pipeline was already capable of building
    and had no consumer for.

    Ordered by Context's own rank so the flattening is deterministic and a
    replay of one snapshot produces byte-identical rows.
    """
    def row(entity) -> ContextEntityRow:
        citation = view.citations.get(entity.entity_id)
        return ContextEntityRow(
            entity_type=entity.entity_type,
            entity_id=entity.entity_id,
            rank=entity.rank,
            reason=entity.reason,
            # THE ENTITY GRAPH, flattened one more time. `scope_for` resolves each
            # edge's target type from its own closed role map, so this carries
            # three strings per link and Intelligence still imports no State.
            # Unbounded on purpose: a link list is entity ids only, State caps a
            # relationship's links by that counterparty's real activity, and the
            # situation engine applies its own enumeration bound. Truncating here
            # would silently hide a connection rather than shorten a blob.
            links=tuple(
                ContextLink(
                    role=link.role,
                    entity_type=link.entity_type,
                    entity_id=link.entity_id,
                )
                for link in entity.links
            ),
            # An entity that recorded no provenance keeps an honestly empty
            # tuple rather than a fabricated one.
            memory_ids=tuple(
                (citation.memory_ids or ())[-_MAX_LINEAGE_PER_ENTITY:] if citation else ()
            ),
            signal_event_ids=tuple(
                (citation.signal_ids or ())[-_MAX_LINEAGE_PER_ENTITY:] if citation else ()
            ),
        )

    return tuple(
        row(e)
        for e in sorted(view.entities, key=lambda e: (e.rank, e.entity_type, e.entity_id))
    )


def _reality_view(
    org_id: uuid.UUID,
    state: OrganizationalState,
    signals: list[ReasoningSignal],
    context: ScopedView | None = None,
) -> OrganizationalRealityView:
    """Flatten State + CONTEXT + ReasoningSignals into the contract Intelligence
    accepts.

    PURE — no clock, no I/O, no randomness. `as_of` is State's logical time and
    the watermark is the memory record State actually folded to, so the same
    snapshot always produces the same view.

    ALL signals are carried, not just the decision-worthy ones. The five types
    Decision drops as interpretation (`OPPORTUNITY`, `LEVERAGE_POINT`,
    `ORG_DRIFT`, `PRIORITY_LOAD`, `CONTEXT_SUMMARY`) are exactly the material an
    organizational insight is made of — `decisions/base.py` says so itself:
    "They remain queryable as ReasoningSignals, so an engine or surface that
    wants them as CONTEXT or EVIDENCE still can."

    Nothing is recomputed. Each row carries the engine's own confidence,
    severity and `evidence` verbatim; a signal that recorded no lineage
    contributes none rather than an invented one.
    """
    watermark = state.watermark.last_memory_id
    return OrganizationalRealityView(
        org_id=org_id,
        as_of=state.as_of,
        state_watermark=str(watermark) if watermark else None,
        # The second clock. Without it, a reasoning-version bump over an
        # unchanged snapshot reuses the previous version's insight ids and the
        # corrected conclusions are silently dropped by ON CONFLICT DO NOTHING.
        reasoning_version=REASONING_VERSION,
        entity_counts=state.counts(),
        # Sorted by the deterministic signal id so the same pass always
        # serialises identically, whatever order the engines ran in.
        signals=tuple(
            sorted(
                (
                    RealitySignalRow(
                        signal_id=str(s.signal_id),
                        signal_type=s.signal_type.value,
                        entity_type=s.entity_type,
                        entity_id=s.entity_id,
                        confidence=s.confidence,
                        severity=s.severity,
                        engine_name=s.engine_name,
                        engine_version=s.engine_version,
                        evidence=s.evidence or {},
                    )
                    for s in signals
                ),
                key=lambda r: r.signal_id,
            )
        ),
        # CONTEXT — the relevant slice of State, with its citations. Empty when
        # Context could not be built, which costs an insight lineage depth and
        # never gives it a fabricated id.
        context=_context_rows(context) if context is not None else (),
    )


def _intelligence_view(org_id: uuid.UUID, rows) -> IntelligenceView:
    """Flatten the org's CURRENT insights into the contract Decision accepts.

    PURE — no clock, no I/O, no randomness. `as_of` is the newest logical time
    among the folded insights, so the same stored rows always build the same view.

    `rows` are whatever `IntelligenceRepository.get_insights` returned: the
    current projection, already org-scoped, already deterministically ordered,
    already free of superseded rows. Nothing about that rule is restated here —
    this function reshapes, it does not select.

    Each fact carries a COPY of the row's `evidence`. The ORM hands out the live
    JSONB dict, and a view that shared it would let a consumer dirty a persisted
    insight by touching what looks like a plain dict.

    IT IS ALSO WHERE THE TWO INSIGHT FAMILIES ARE TOLD APART, because this is the
    only layer that may know both vocabularies: `navis/decisions/**` may not
    import `navis.intelligence` (AST-pinned), and `REALITY_INSIGHT_TYPES` is
    Intelligence's own answer to "is this row a description of current State or a
    measurement over past outcomes?". The distinction reaches Decision as a bool
    on each fact — the same shape every other cross-layer input arrives in — and
    decides both which relation a signal match records and whether the fact's
    watermark is a claim about a snapshot at all. See
    `decisions.base.InsightRelation` and `SupportingInsight`.
    """
    from navis.intelligence.insights import REALITY_INSIGHT_TYPES

    reality_types = frozenset(t.value for t in REALITY_INSIGHT_TYPES)
    facts = tuple(
        IntelligenceFact(
            insight_id=str(r.insight_id),
            insight_type=r.insight_type,
            title=r.title,
            key=r.key,
            summary=r.summary,
            confidence=r.confidence,
            sample_size=r.sample_size,
            evidence=dict(r.evidence or {}),
            # Recorded by `intelligence.service._provenance`; sorted there and
            # kept in that order. Absent on a fact that folded no signals, and
            # that emptiness is honest rather than a gap to fill.
            supporting_reasoning_signal_ids=tuple(
                (r.evidence or {}).get("supporting_reasoning_signal_ids") or ()
            ),
            state_watermark=r.state_watermark,
            learning_version=r.learning_version,
            # Reality rows are versioned by `state_watermark`, Learning rows by
            # `learning_version` — `intelligence.service._version_key` keys them
            # on exactly this membership, so reading the same set here cannot
            # drift from how the row was minted.
            describes_state=r.insight_type in reality_types,
        )
        for r in rows
    )
    return IntelligenceView(
        org_id=org_id,
        as_of=max((r.created_at for r in rows if r.created_at is not None), default=None),
        facts=facts,
    )


async def _current_intelligence(db: AsyncSession, org_id: uuid.UUID) -> IntelligenceView:
    """Read the org's current insights and project them for Decision.

    Read through `IntelligenceRepository`, never with a query of our own: that
    boundary owns which projection is current, and a second query here would be
    the third read semantics for one append-only table.

    Called AFTER `_project_intelligence`, deliberately — the projection commits in
    its own session, so reading afterwards gives Decision the insights derived
    from the very snapshot it is about to rank, rather than the previous one.
    """
    from navis.intelligence.repository import IntelligenceRepository

    return _intelligence_view(org_id, await IntelligenceRepository(db).get_insights(org_id))


async def _project_intelligence(
    org_id: uuid.UUID, reality: OrganizationalRealityView
) -> None:
    """Re-project Intelligence with the current reality attached.

    ITS OWN SESSION, deliberately. The projection reads committed `learning_*`
    state and appends to `intelligence_insights`; running it inside the brief's
    transaction would mean a failed insert aborts that transaction and takes the
    just-written `reasoning_signals` and the brief itself down with it. A
    rollback could not rescue them either — it would discard `persist_signals`.
    Isolation is the only way a projection failure costs an insight and never
    the brief. The LEARNING_UPDATED subscriber already opens its own session for
    the same call, so this is the established pattern, not a new one.

    Never fatal, for the same reason: the brief is the product's heartbeat and
    must not stop because a downstream projection did. Failures are logged with
    a metric rather than swallowed silently.
    """
    from navis.intelligence.repository import IntelligenceRepository
    from navis.intelligence.service import IntelligenceService

    try:
        async with session_scope() as db:
            rows = await IntelligenceService(IntelligenceRepository(db)).project(
                org_id, reality=reality
            )
    except Exception:  # noqa: BLE001 — see docstring: never fail the brief
        logger.exception("intelligence projection failed org=%s", org_id)
        metrics.incr("intelligence.projection.failed")
        return
    metrics.incr("intelligence.reality.signals", value=len(reality.signals))
    logger.info(
        "intelligence projected org=%s insights=%d reality_signals=%d watermark=%s",
        org_id, len(rows), len(reality.signals), reality.state_watermark,
    )


async def build_brief(db: AsyncSession, org_id: uuid.UUID) -> DailyBrief:
    """State V2 -> Reasoning V2 -> Decision. Reused by the API and the scheduler.

    Adaptive's ACTIVE policy is resolved HERE and injected into Decision — Decision
    never imports Adaptive. This is where the double loop closes at runtime.
    """
    from navis.orchestration.runtime_policy import resolve_policy

    # incremental_update (not get_state) so the brief's as_of reflects the
    # NEWEST memory_records, not whatever snapshot happens to be cached.
    # get_state serves reads from cache and would freeze as_of at the last
    # rebuild — leaving today's brief permanently stamped "Thursday 9 Jul"
    # even after new signals land. Incremental only folds records past the
    # snapshot watermark, so cost is O(delta) not O(all-memory).
    state = await StateService(db).incremental_update(org_id)
    # Announce that State V2 is current for this org (re-homed from the retired V1
    # state service). Drives the onboarding STATE_READY milestone; State itself stays
    # a pure read model and never touches the event bus.
    await event_bus.publish(
        DomainEvent(type=EventType.STATE_UPDATED, org_id=org_id, source="state.v2",
                    payload=state.counts())
    )
    # Organizational-relevance evidence, read HERE (the composition root) and
    # handed to Reasoning, so Reasoning stays a pure function and performs no
    # I/O. See engines/meta.py: relevance is decided from the accumulated set of
    # signal classifications, never from Memory's per-message `is_automated`.
    evidence = await classifications_by_entity(
        db, org_id, entity_type=EntityType.RELATIONSHIP.value
    )
    signals = run_reasoning_v2(state, evidence=evidence)
    # Persist the provenance BEFORE Decision consumes it. Execution, Observation,
    # Learning and Intelligence all store `reasoning_signal_ids`; without this
    # they were permanent references to rows that never existed, and the ids
    # cannot be recomputed because `compute_signal_id` hashes reasoning_version.
    # Append-only and idempotent, so rebuilding a brief writes nothing new.
    # Written here, not inside Reasoning: the engines stay a pure function of
    # State and never touch a database.
    await persist_signals(db, signals)
    watermark = state.watermark.last_memory_id
    # ── CONTEXT ──────────────────────────────────────────────────────────────
    # Resolved BEFORE Intelligence projects, and that ordering is the whole
    # point: Context is what lets an insight cite the memory records and
    # business-system events behind the entities it names, and it must therefore
    # describe the same snapshot Reasoning just ran over.
    #
    # `resolve_policy` moved up with it because `scope_for` is governed by the
    # same Adaptive policy Decision is (`brief_max_items` caps the per-type
    # selection) — so Context, Reasoning's consumers and Decision all run under
    # one governing snapshot instead of Context silently using code defaults.
    # Purely a reordering of two reads; neither has a side effect.
    #
    # PURE, and State-only: `scope_for` selects and ranks entities State already
    # holds. It reads no provider, performs no I/O and reaches no conclusion.
    policy = await resolve_policy(db, org_id)
    context = scope_for(state, MorningBrief(), policy, as_of=state.as_of)
    # Hand Intelligence the organization's CURRENT reality. Intelligence may not
    # import State or Reasoning (its repository reads Learning only, pinned by an
    # AST test), so the projection is assembled here — the same seam that already
    # reads `evidence` for Reasoning and resolves Adaptive's policy for Decision.
    await _project_intelligence(org_id, _reality_view(org_id, state, signals, context))
    # The fourth join at this seam, and the same shape as the other three: read
    # here, flatten to primitives, hand in. Decision imports no Intelligence.
    intelligence = await _current_intelligence(db, org_id)
    return await DecisionService(db).decide(
        org_id,
        signals,
        reasoning_version=REASONING_VERSION,
        state_watermark=str(watermark) if watermark else None,
        as_of=state.as_of,
        policy=policy,
        intelligence=intelligence,
    )


async def current_state_watermark(db: AsyncSession, org_id: uuid.UUID) -> str | None:
    """The memory record State has folded to, read WITHOUT folding anything.

    `SnapshotRepository.watermark_only` exists for exactly this shape of question
    — it selects the two cursor columns and skips the ~22KB `full_projection`
    JSONB — so asking "has State moved?" costs one indexed row and never a fold.

    None when the org has no snapshot at the current projection version, which is
    a real answer ("State has never been built for this org") and must stay
    distinguishable from "State is at watermark X".
    """
    from navis.state.v2.models import PROJECTION_VERSION
    from navis.state.v2.repository import SnapshotRepository

    cursor = await SnapshotRepository(db).watermark_only(
        org_id, projection_version=PROJECTION_VERSION
    )
    if cursor is None:
        return None
    _timestamp, memory_id = cursor
    return str(memory_id) if memory_id else None


async def brief_for_today(db: AsyncSession, org_id: uuid.UUID) -> tuple[DailyBrief, dict]:
    """Today's brief, REBUILT IF STATE HAS MOVED SINCE IT WAS RANKED.

    THE DEFECT THIS CLOSES. `latest_brief` returns the newest stored brief and
    `/brief/today` built one only when NONE existed — so once a brief existed, the
    endpoint served it forever. Live, that meant Today presented a picture ranked
    at watermark `f8675a7f` (2026-08-22) while State had advanced to `f0102043`,
    with 55 memory records and 13 signal events folded in between. The founder was
    reading a two-day-old organization with nothing on the page saying so, and the
    commitment it led with had aged from 11.85 days to 13.89.

    THE CACHE IS NOT REMOVED, it is given a validity condition. A brief's identity
    is already `(org, reasoning_version, state_watermark)`
    (`decisions.base.compute_brief_id`), so the watermark IS the cache key and
    "the key changed" is the ordinary reason to recompute. When State has not
    moved, this does exactly what it did before: one cheap indexed read, then the
    stored brief.

    NO SECOND SCHEDULER, and no new pipeline. It calls `build_brief` — the same
    function `POST /brief/generate` and the Celery task both call — under a
    condition, rather than introducing another producer. `build_brief` is
    idempotent (deterministic brief id, upsert), so two concurrent readers
    converge on one row instead of racing.

    BOUNDED. A rebuild can only fire after State has actually advanced, which
    requires a sync to have landed; between syncs, repeated polls compare equal
    and do nothing. The fold itself is `incremental_update`, O(delta), and no
    layer on this path calls an LLM.

    Returns the brief and a `freshness` block that reports the comparison
    honestly — including the case where it could NOT be made. Reporting is not
    redundant with rebuilding: a rebuild can legitimately fail to advance the
    watermark (State has no snapshot yet), and the founder must be able to tell
    "current" from "we could not tell".
    """
    brief = await DecisionService(db).latest_brief(org_id)
    state_watermark = await current_state_watermark(db, org_id)

    if brief is None:
        brief = await build_brief(db, org_id)
        return brief, _freshness(brief, state_watermark, rebuilt=True)

    # `None` means State has no snapshot at the current projection version, so
    # there is nothing to compare against — never a reason to discard a brief
    # that exists. Fails closed toward SERVING, and says so in `freshness`.
    stale = state_watermark is not None and brief.state_watermark != state_watermark
    if stale:
        logger.info(
            "brief stale org=%s brief_watermark=%s state_watermark=%s — rebuilding",
            org_id, brief.state_watermark, state_watermark,
        )
        metrics.incr("brief.today.rebuilt_stale")
        brief = await build_brief(db, org_id)
        return brief, _freshness(brief, state_watermark, rebuilt=True)

    return brief, _freshness(brief, state_watermark, rebuilt=False)


def _freshness(brief: DailyBrief, state_watermark: str | None, *, rebuilt: bool) -> dict:
    """Whether this brief describes the State that is currently folded. PURE.

    `current` is TRI-STATE for the same reason `state.v2.models` fields are:

      True  — the brief's watermark IS State's current watermark
      False — State has moved past this brief (only reachable when a rebuild was
              attempted and still could not close the gap)
      None  — no comparison was possible, because State has no snapshot at the
              current projection version. NOT "fresh", and not "stale" either.

    A client that renders `current is False` or `current is None` as "up to date"
    is making a claim this block deliberately refuses to make.
    """
    return {
        "current": (
            None if state_watermark is None
            else brief.state_watermark == state_watermark
        ),
        "brief_watermark": brief.state_watermark,
        "state_watermark": state_watermark,
        # Whether THIS request recomputed the brief. Operationally the difference
        # between "the scheduler is keeping up" and "reads are carrying it".
        "rebuilt": rebuilt,
    }


async def run_brief_for_org(org_id: uuid.UUID) -> dict:
    """The SCHEDULED brief: build it, then propose the actions it implies.

    WHY THE PROPOSE STEP IS HERE.

    `propose_actions` is described in its own docstring as "THE proactive
    producer", every row it writes is stamped `ExecutionOrigin.PROACTIVE`, and
    `GET /actions` filters on exactly that so the Action Center shows things
    "Navis noticed". But nothing scheduled ever called it: Celery beat runs
    `build_all_briefs` -> `build_brief_for_org` -> this function, and this
    function stopped at `build_brief`. The ONLY caller was
    `POST /brief/propose-actions`, a manual endpoint.

    So the proactive queue was only ever populated when a client asked for it.
    Navis did not notice an unanswered email and raise it; it noticed only when
    something else prompted it to look. That is the difference between the
    product being proactive and being on-demand, and it was a wiring gap rather
    than a decision — nothing documented the omission and the bridge was built
    for precisely this call site.

    SAFE TO RUN UNATTENDED, which is why it belongs on the schedule at all:

      * it creates REQUESTED rows only. Approval is still the single edge into
        APPROVED, so nothing here can produce a side effect;
      * it is idempotent by construction — execution ids are deterministic and
        the obligation index is unique, so re-running the brief converges on the
        same rows rather than accumulating duplicates (proven in
        `test_re_running_the_brief_reuses_the_same_execution_identity`);
      * a failure to propose must not lose the brief that was just built, so it
        is caught: the brief is the founder-visible artefact and the next run
        re-proposes from it.
    """
    # Tenant log context for the whole pass (T-21). `org_id` is the task's only
    # argument, but it is not taken on trust for anything that MATTERS: the gate
    # below resolves entitlements from persisted org state, and every read inside
    # `build_brief` is filtered on this same id. Binding it here is what makes the
    # brief pipeline's logs attributable — previously every line carried "-".
    with org_context(org_id):
        async with session_scope() as db:
            # ── ENTITLEMENT GATE (T-14) ──────────────────────────────────────
            # `POST /brief/generate` has always required Capability.DECISION.
            # This task produces the SAME artefact on a timer and required
            # nothing, so a suspended or free-tier organization kept getting
            # briefs built — and kept getting REQUESTED execution rows proposed
            # from them. Asked before any work because building a brief folds the
            # org's whole state and calls the LLM: the point is not to discard
            # the result, it is not to spend it.
            if not await scheduled_work_allowed(
                db, org_id, Capability.DECISION, task="build_brief_for_org"
            ):
                return {"org_id": str(org_id), "skipped": "not_entitled"}

            brief = await build_brief(db, org_id)
            result = {
                "org_id": str(org_id),
                "brief_id": str(brief.id),
                "decisions": len(brief.decisions),
            }
            try:
                result["proposed"] = await propose_actions(db, org_id)
            except Exception:  # noqa: BLE001 — see docstring: never lose the brief
                logger.exception(
                    "propose_actions failed for org=%s after building the brief", org_id
                )
                result["proposed"] = 0
            return result


async def propose_actions(db: AsyncSession, org_id: uuid.UUID) -> int:
    """Decision -> Execution bridge: turn the latest brief's decisions into
    REQUESTED ExecutionRequests (idempotent; human approval still required).

    THIS IS ALSO WHERE THE ACCOUNT IS RECORDED (T-11).

    `execution_requests.integration_id` says which of the organization's
    connected accounts an action must be performed AS. It was added by migration
    0042 and read all the way down — `worker._Lease` copies it, `__ctx__` carries
    it, `resolve_integration` pins on it — but NO producer ever wrote one, so the
    column was uniformly NULL and every send fell back to resolving by
    (organization, provider).

    That fallback is fail-closed, and that is exactly why the gap mattered: the
    org-scoped resolvers refuse a multi-account organization with "recreate the
    action from the message it answers". With no producer recording an account,
    recreating it produced another NULL — so the refusal had no remedy and every
    proactive reply for such an organization was permanently unsendable.

    A reply is the case the column was built for and the one that can answer it
    honestly: the thread it answers was captured on exactly one account, and
    `memory_records.source_integration_id` has recorded which since the first
    migration. So the account is RE-DERIVED from persisted lineage here, never
    accepted from a payload — the same rule the worker applies to the tenant.

    A thread whose lineage is missing or ambiguous contributes no id and the row
    stays NULL, which is the honest "not recorded" the model documents. Nothing
    is guessed, and nothing that already worked changes: a single-account
    organization resolves to the same credential either way.
    """
    from navis.decisions.service import DecisionService
    from navis.execution.models import ExecutionOrigin
    from navis.execution.proposer import propose_from_decisions
    from navis.execution.service import ExecutionService
    from navis.memory.lineage import integrations_for_threads

    brief = await DecisionService(db).latest_brief(org_id)
    if brief is None:
        return 0
    proposed = propose_from_decisions(brief.decisions or [], org_id)
    # ONE lookup for the whole brief, before the loop — see the lineage helper.
    by_thread = await integrations_for_threads(
        db, org_id, [str(p.payload.get("thread_id") or "") for p in proposed]
    )
    svc = ExecutionService(db)
    created = 0
    for p in proposed:
        integration_id = by_thread.get(str(p.payload.get("thread_id") or ""))
        _req, was_created = await svc.request(
            p.org_id, p.decision_id, p.action_type, p.payload,
            reasoning_signal_ids=p.reasoning_signal_ids,
            reasoning_version=p.reasoning_version,
            state_watermark=p.state_watermark,
            # The connected account this reply must be sent FROM, re-derived
            # above from the thread's own memory lineage. None when the thread
            # names no single account; the credential boundary then refuses a
            # multi-account org rather than picking one.
            integration_id=integration_id,
            # THE proactive producer: these come from reasoning over signals the
            # founder has not seen, so the Action Center owns reviewing them.
            origin=ExecutionOrigin.PROACTIVE,
        )
        created += int(was_created)
    return created


async def _dispatch_all() -> int:
    """Fan-out: enqueue ONE per-org task per organization (keyset-paginated).

    Avoids a single task sequentially looping every org and avoids loading 1M ids
    into memory — the heavy per-org work parallelizes across the worker fleet.
    """
    from navis.identity.models import Organization

    dispatched, after, page = 0, None, 1000
    while True:
        async with session_scope() as db:
            stmt = select(Organization.id).order_by(Organization.id).limit(page)
            if after is not None:
                stmt = stmt.where(Organization.id > after)
            ids = (await db.scalars(stmt)).all()
        if not ids:
            break
        for oid in ids:
            build_brief_for_org.delay(str(oid))
        dispatched += len(ids)
        after = ids[-1]
        if len(ids) < page:
            break
    return dispatched


@celery_app.task(name="navis.orchestration.brief.build_brief_for_org")
def build_brief_for_org(org_id: str) -> dict:
    return run_async(run_brief_for_org(uuid.UUID(org_id)))


@celery_app.task(name="navis.orchestration.brief.build_all_briefs")
def build_all_briefs() -> dict:
    return {"dispatched": run_async(_dispatch_all())}
