"""The Developer API — `/api/v1/developer/*`.

MOUNTED ON THE EXISTING APPLICATION. This router is included by
`navis.api.router.api_router`, which `main.create_app` mounts under
`settings.api_v1_prefix`. There is no second FastAPI app, no second OpenAPI
document, and no separate process: a Developer route appears in the same schema,
behind the same middleware, with the same correlation id, as every other route.

EVERY DATA ROUTE DELEGATES. The handlers below call the web edge's own handler
coroutines — `context_router.context_bundle`, `knowledge_router.knowledge_entity`,
`observation.router.list_outcomes`, `learning.router.learning_summary` — as plain
functions with the dependencies this route already resolved. That is deliberate
and it is the single most important property of this module: there is exactly ONE
implementation of each read, so the Developer API and the web app cannot disagree
about what an organization's context, entities, outcomes or learning are. The
delegated function's own `Depends(...)` defaults are inert when it is called
directly, which is why each route re-declares its gates explicitly below rather
than inheriting them by accident.

WHERE A HANDLER COULD NOT BE CALLED, THE SERVICE IS. Three routes reuse the
domain service and its serializer instead of a web handler, and each has a
reason it could not do otherwise:

  * `/insights`  — `intelligence.router.list_insights` resolves `CurrentContext`
    rather than a `Principal`, so it cannot be handed this route's dependencies.
    It calls `IntelligenceService.get_insights` and Intelligence's own
    `_serialize`, which is the same pair that handler calls.
  * `/decisions` — `/brief/today` REBUILDS a brief whose watermark has moved,
    writing `reasoning_signals`, `intelligence_insights` and `daily_briefs`.
    Delegating to it would make a Developer read mutate four tables, so this
    composes the two READ halves the web edge also uses: `latest_brief` and
    `brief_router._with_executions`.
  * `/reasoning` — there was no handler to delegate to. Reasoning owns no router
    on either edge, so this is the layer's first reader; it reads through
    `reasoning.store`, which already owned every read of those two tables.

None of the three reimplements a domain rule, and none of them is a second
opinion about what the data means.

THE ONE CHAIN, FOR EVERY ROUTE:

    credential -> get_principal -> Principal(org from membership/api_keys row)
               -> require_permission  (403 — may this credential?)
               -> require_capability  (402 — is this tenant entitled?)
               -> rate_limit          (429 — per authenticated organization)
               -> DeveloperContext    -> existing domain service -> repository
               -> repository tenant predicate on ctx.organization_id

APPLICATION API AUTHORIZATION != DEVELOPER API AUTHORIZATION. This is a deliberate,
permanent asymmetry, not a migration in progress.

The web edge's read routes (`/context`, `/knowledge/*`, `/outcomes`,
`/learning/summary`) gate on a CAPABILITY only. That is coherent for a human
session: the caller is a member of the organization, the UI decides what to render
from `/auth/me`'s permission list, and an `employee` who holds neither
`view_state` nor `view_memory` still legitimately sees the product. Adding a
permission gate there would revoke access from real people as a side effect of
building a machine boundary, so those routes are UNCHANGED and stay that way until
someone deliberately does the role migration.

A machine credential has no UI, no session and no human judgment behind it, and it
is minted with an explicit scope list precisely so that its reach can be bounded.
So every Developer data route requires a permission AND a capability. The
practical consequence, and the reason this boundary is worth having: a key minted
with `scopes=[]` reads the four application routes above and reads NOTHING here.

Both halves are pinned by tests — `tests/test_architecture.py` structurally, and
`tests/integration/test_developer_boundary_postgres.py` at the wire, including a
characterisation test that fails the day the application routes change.

NO ROUTE TAKES AN IDENTITY PARAMETER. There is no `organization_id`, `agent_id`,
`principal_id` or `user_id` anywhere in this file, in any form — path, query,
header or body. `/context` on the web edge still accepts a `user_id` for frontend
compatibility and ignores it; the Developer boundary does not accept it at all,
because an ignored identity parameter is a parameter a client will eventually
believe in.

READ-ONLY, BY CONSTRUCTION. Every handler is a GET and every domain call it makes
is a read. `StateService.incremental_update` folds new memory into the snapshot
cache — the same call `/context` already makes on the web edge — and is
idempotent; it advances a projection, it does not create a fact.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from navis.developer.gateway import (
    AGENT_MANAGEMENT_PERMISSIONS,
    DeveloperContext,
    get_developer_context,
    require_developer,
    require_developer_any,
)
from navis.developer.schemas import (
    DEVELOPER_ERRORS,
    DEVELOPER_ERRORS_GATED,
    DeveloperContextOut,
    DeveloperDecisionsOut,
    DeveloperEntityOut,
    DeveloperInsightsOut,
    DeveloperLearningOut,
    DeveloperMeOut,
    DeveloperMemoryPageOut,
    DeveloperPermissionCatalogOut,
    DeveloperPermissionOut,
    DeveloperOrganizationOut,
    DeveloperOutcomesOut,
    DeveloperPrincipalOut,
    DeveloperReasoningOut,
    DeveloperStateOut,
)
from navis.identity.models import Organization
from navis.identity.permissions import (
    AGENT_ASSIGNABLE_PERMISSIONS,
    PERMISSION_CATALOG,
    Permission,
)
from navis.identity.throttle import rate_limit
from navis.organization.contract import Capability
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger

logger = get_logger("developer")

router = APIRouter(prefix="/developer", tags=["developer"])

# Every read route is throttled per authenticated ORGANIZATION on the shared
# "read" bucket — the same bucket and the same limit the web edge's read routes
# use. A separate Developer bucket would let one tenant's agent fleet spend a
# budget its humans are not counted against, which is the tenant-isolation
# property the limiter exists to defend.
_READ_LIMIT = Depends(rate_limit("read", "rate_limit_read_per_window"))

# Page-size ceiling for the memory log. The log is append-only and grows with
# every signal, so an unbounded read is a read that gets slower forever.
_MAX_MEMORY_LIMIT = 200

# The same ceiling for the reasoning log, and for the same reason: every brief
# appends the conclusions it reached, so the table grows without bound. It is
# lower than Memory's because each row is joined against its current observation
# — one extra indexed read for the page, sized so that read stays cheap.
_MAX_REASONING_LIMIT = 200


@router.get(
    "/me",
    response_model=DeveloperMeOut,
    responses=DEVELOPER_ERRORS,
    dependencies=[_READ_LIMIT],
    summary="Who this credential is, and what it may reach",
)
async def developer_me(
    ctx: DeveloperContext = Depends(get_developer_context),
    db: AsyncSession = Depends(get_session),
) -> DeveloperMeOut:
    """The caller's own identity — the first call any MCP server, SDK or CLI makes.

    IDENTITY IS NOT SELECTABLE. This route accepts no parameters at all. There is
    no `organization_id` to pick a tenant with, no `agent_id` to impersonate a
    service account with, and no `principal_id` to name a subject with. Everything
    reported here is derived from the presented credential: the organization comes
    from the `memberships` row (JWT) or the `api_keys` row (API key), and the
    service account comes from `api_keys.service_account_id`. A caller cannot ask
    this endpoint about anyone but itself.

    NO SECRET IS REPORTED. `credential_id` is the `api_keys` row id, not the key.
    The plaintext exists exactly once, in the response to `POST
    /identity/api-keys`, and is never stored — `api_keys.hashed_secret` holds only
    a SHA-256 — so there is nothing here for this route to leak even by mistake.

    AUTHENTICATION ONLY. Unlike every other Developer route this one names no
    permission: a credential that cannot discover what it is cannot report a
    misconfiguration, and the answer tells a caller nothing it could not learn
    from the 403s it would collect by trying the other routes.
    """
    from navis.organization.service import OrganizationService

    org = await db.get(Organization, ctx.organization_id)
    # `capabilities` is the org's ENTITLEMENT snapshot, one round-trip for all of
    # them (`EntitlementService.snapshot`), and it is the same authority
    # `require_capability` consults per request — so a client can predict a 402
    # instead of discovering it. It is emphatically not an authorization answer:
    # a true capability with a missing permission is still a 403.
    capabilities = await OrganizationService(db).entitlements.snapshot(ctx.organization_id)
    return DeveloperMeOut(
        organization=DeveloperOrganizationOut(
            id=ctx.organization_id,
            # An organization row is guaranteed present for a resolved principal
            # (both `memberships` and `api_keys` carry a FK to it), but reporting
            # a placeholder would be a lie if that ever stopped being true.
            name=org.name if org else "",
            slug=org.slug if org else "",
        ),
        principal=DeveloperPrincipalOut(
            kind=ctx.kind,
            subject_id=ctx.subject_id,
            service_account_id=ctx.service_account_id,
            credential_id=ctx.credential_id,
            permissions=sorted(ctx.permissions),
        ),
        capabilities=capabilities,
    )


@router.get(
    "/permissions",
    response_model=DeveloperPermissionCatalogOut,
    responses=DEVELOPER_ERRORS_GATED,
    dependencies=[_READ_LIMIT],
    summary="The permission vocabulary, and which of it this caller may grant",
)
async def developer_permissions(
    ctx: DeveloperContext = Depends(
        require_developer_any(AGENT_MANAGEMENT_PERMISSIONS, Capability.STATE)
    ),
) -> DeveloperPermissionCatalogOut:
    """The canonical catalog, served from the module that DEFINES the vocabulary.

    WHY THIS ROUTE EXISTS. The Developer Console shipped its own hand-written
    list of four agent permissions while the Developer boundary had grown to
    gate on nine. Nothing detected the drift, because the list and the routes
    lived in different languages in different repositories — so a permission
    the API accepted could not be selected in the UI whose entire job is to
    grant it. A management surface must read its vocabulary from the authority
    that enforces it, or it eventually describes a system that no longer exists.

    IT DECIDES NOTHING. `grantable` mirrors the rule `ensure_grantable_scopes`
    applies on write, and `assignable_to_agents` is a console default, not a
    server rule — the API still accepts any valid permission the caller holds.
    A client that ignored both fields entirely could not escalate past the Core,
    because the Core re-checks the grant itself.

    GATED ON THE SAME PAIR AS THE AGENT ROUTES — `manage_service_accounts` or
    `manage_own_agents` (P1-7) — matching the agent surface it serves. When it
    required the organization-wide permission alone, a self-serve developer was
    allowed to register an agent and handed an empty catalog to choose its
    permissions from, so the form they could submit had nothing to tick. The
    shared `AGENT_MANAGEMENT_PERMISSIONS` is what keeps the two in step.

    It reports no organizational data at all — a fixed vocabulary, plus a
    projection of the caller's own permissions, which `/developer/me` already
    returns to that same caller.
    """
    held = set(ctx.permissions)
    return DeveloperPermissionCatalogOut(
        permissions=[
            DeveloperPermissionOut(
                permission=name,
                kind=info.kind,
                summary=info.summary,
                sensitive=info.sensitive,
                assignable_to_agents=name in AGENT_ASSIGNABLE_PERMISSIONS,
                grantable=name in held,
            )
            # Sorted so the console renders a stable order without imposing one
            # of its own — a list that reshuffles between reads reads as a
            # change to what the organization may do.
            for name, info in sorted(PERMISSION_CATALOG.items())
        ]
    )


@router.get(
    "/context",
    response_model=DeveloperContextOut,
    responses=DEVELOPER_ERRORS_GATED,
    dependencies=[_READ_LIMIT],
    summary="The intent-scoped view over current State",
)
async def developer_context(
    ctx: DeveloperContext = Depends(
        require_developer(Permission.VIEW_STATE, Capability.DECISION)
    ),
    db: AsyncSession = Depends(get_session),
) -> Any:
    """Exactly what `GET /context` serves, to a machine caller, plus event time.

    Delegates to `context_router.context_bundle`, so the scoped view, the policy
    resolution, the state fold and the audit log line are the web edge's own —
    one implementation, one answer.

    `user_id` is NOT accepted. The web edge takes it for frontend compatibility
    and ignores it (Phase 1 scoping is org-wide); repeating an ignored identity
    parameter on a machine contract would invite a client to depend on it.

    ── WHY THIS ROUTE ADDS ONE FIELD THE WEB EDGE DOES NOT ──────────────────
    `ScopedView` publishes `as_of`, which is OBSERVATION time — when Navis last
    folded something. On its own it reads as "this is current", and it is not:
    the newest fact a projection holds can be weeks older than the moment it
    was folded. A person reading the console has the organization around the
    answer and notices; an agent has the answer alone and does not, and an
    agent that presents a three-week-old organization as current makes
    decisions on it.

    `POST /developer/context` has published `newest_fact_at` beside `as_of`
    since the freshness contract was written. This route did not, so the two
    Developer context routes disagreed about what freshness means — and this
    is the one an MCP client reaches first.

    ── THE VALUE IS READ, NEVER COMPUTED ───────────────────────────────────
    `state.fold_frontier.last_event_timestamp` — the same field, from the same
    Core service, that `orchestration.context.state_groundings` reads for the
    POST route. State has always maintained it for its own correctness (it is
    how an out-of-order record is detected and forced into a rebuild). There is
    no second freshness calculation here and no new source: nothing is derived,
    no connector is consulted, and no synchronization is triggered.

    ── READ-ONLY, AND ONE STATE READ MORE THAN BEFORE ──────────────────────
    `read_only_state` never folds, never rebuilds and never disposes — the same
    read the POST route takes, for the same reason. `context_bundle` has just
    saved and cached this projection, so this resolves on `StateService`'s warm
    path rather than re-deriving anything.

    ── A MISMATCHED PAIR IS REPORTED AS ABSENT ─────────────────────────────
    If State moved between the scoping read and this one, the frontier belongs
    to a different projection than `as_of` does. Publishing it anyway would
    understate the gap — the one direction that matters, because the whole
    point of the field is to expose staleness. So the watermarks are compared
    and a mismatch answers null, which the contract already defines as unknown.
    """
    from navis.api.context_router import context_bundle
    from navis.state.v2 import StateService

    body = await context_bundle(user_id=None, p=ctx.principal, db=db)

    state = await StateService(db).read_only_state(ctx.organization_id)
    newest_fact_at = None
    if state is not None:
        watermark = (
            str(state.watermark.last_memory_id)
            if state.watermark.last_memory_id
            else None
        )
        # `scope_for` builds `state_watermark` from this exact field, so equality
        # here means the two values describe one projection rather than two.
        if watermark == body.get("state_watermark"):
            frontier = state.fold_frontier.last_event_timestamp
            newest_fact_at = frontier.isoformat() if frontier else None

    # ADDITIVE. Every key `context_bundle` returned is untouched and in the same
    # place; the dict it hands back is this route's to extend, and the web edge
    # returns its own — the Application's `GET /context` is not reached by this.
    return {**body, "newest_fact_at": newest_fact_at}


@router.get(
    "/state",
    response_model=DeveloperStateOut,
    responses=DEVELOPER_ERRORS_GATED,
    dependencies=[_READ_LIMIT],
    summary="The current State V2 projection",
)
async def developer_state(
    ctx: DeveloperContext = Depends(
        require_developer(Permission.VIEW_STATE, Capability.STATE)
    ),
    db: AsyncSession = Depends(get_session),
) -> Any:
    """The organization's current projection, as State V2 itself serializes it.

    `StateService` IS the sole interface to State (see `navis/state/v2/__init__`),
    and `incremental_update` is the same entry `/context` and `/knowledge/home`
    call — it folds any memory that landed since the last snapshot and returns the
    projection. `OrganizationalState.to_dict` is State's own deterministic
    serializer; nothing is reshaped here.

    GATED ON `Capability.STATE`, not `DECISION`. This is the raw projection rather
    than the brief-scoped view, so the honest entitlement is the one that owns the
    layer being read. It is narrower than `/developer/context` in permission terms
    and broader in plan terms — `STATE` is included from FREE upward.

    SIZE IS THE CALLER'S TO MANAGE. The full projection is returned unpaged, the
    same as `/context` on the web edge. `record_count` and `counts` describe how
    large it is; a paged State read is a separate contract and is not invented here.
    """
    from navis.state.v2 import StateService

    state = await StateService(db).incremental_update(ctx.organization_id)
    return state.to_dict()


@router.get(
    "/memory",
    response_model=DeveloperMemoryPageOut,
    responses=DEVELOPER_ERRORS_GATED,
    dependencies=[_READ_LIMIT],
    summary="The append-only memory log",
)
async def developer_memory(
    entity_type: str | None = Query(
        default=None,
        description=(
            "Canonical `EntityType` value to filter on (person, company, thread, "
            "commitment, ...). Omit for every type, newest first."
        ),
    ),
    limit: int = Query(default=100, description=f"1-{_MAX_MEMORY_LIMIT}."),
    ctx: DeveloperContext = Depends(
        require_developer(Permission.VIEW_MEMORY, Capability.MEMORY)
    ),
    db: AsyncSession = Depends(get_session),
) -> Any:
    """Immutable facts from `memory_records`, exactly as Memory stored them.

    THE CANONICAL LOG, NOT A DERIVED VIEW. This is `MemoryRepository`, which is
    the only reader and writer of `memory_records`, and each row is published
    with its full provenance and no interpretation. It deliberately does NOT
    reuse `/memory/people` or `/decision-memory`: those two re-derive, in the API
    layer, things State V2 already holds, and add scores State does not hold at
    all (a response-likelihood bucket, an escalation risk). Publishing that on a
    machine contract would harden a second organizational model into an SDK.

    An unrecognised `entity_type` returns an EMPTY page rather than a 422. The
    vocabulary is Memory's and it grows; a client pinned to an older enum should
    see "no facts of that kind" instead of a hard failure. Filtering with a
    tenant predicate on top of it means an unknown type cannot widen the scope.
    """
    from navis.memory.envelope import EntityType
    from navis.memory.repository import MemoryRepository

    # Clamped rather than declared with `Query(ge=…, le=…)`, matching every other
    # read handler in this codebase: those are called directly in tests, where a
    # `Query` default arrives as a `Query` object instead of an int.
    limit = max(1, min(int(limit), _MAX_MEMORY_LIMIT))
    repo = MemoryRepository(db)

    if entity_type:
        try:
            typed = EntityType(entity_type)
        except ValueError:
            rows: list[Any] = []
        else:
            rows = list(await repo.list_by_entity_type(
                ctx.organization_id, typed, limit=limit
            ))
    else:
        rows = list(await repo.recent(ctx.organization_id, limit=limit))

    return {
        "records": [
            {
                "memory_id": r.memory_id,
                "entity_type": r.entity_type,
                "entity_id": r.entity_id,
                "source_signal_event_id": r.source_signal_event_id,
                "source_integration_id": r.source_integration_id,
                "extractor_name": r.extractor_name,
                "extractor_version": r.extractor_version,
                "schema_version": r.schema_version,
                "confidence": r.confidence,
                "event_timestamp": r.event_timestamp.isoformat() if r.event_timestamp else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "dedup_key": r.dedup_key,
                "payload": r.payload or {},
            }
            for r in rows
        ],
        "count": len(rows),
        "limit": limit,
        "entity_type": entity_type,
        "truncated": len(rows) >= limit,
        "order": "event_timestamp.desc",
    }


@router.get(
    "/entities/{entity_type}/{entity_id}",
    response_model=DeveloperEntityOut,
    responses={
        **DEVELOPER_ERRORS_GATED,
        404: {"description": "No entity of that type and id in this organization's State."},
    },
    dependencies=[_READ_LIMIT],
    summary="One canonical entity, its edges, and what the pipeline said about it",
)
async def developer_entity(
    entity_type: str,
    entity_id: str,
    ctx: DeveloperContext = Depends(
        require_developer(Permission.VIEW_STATE, Capability.DECISION)
    ),
    db: AsyncSession = Depends(get_session),
) -> Any:
    """Exactly what `GET /knowledge/entity/{type}/{id}` serves.

    Delegates to `knowledge_router.knowledge_entity`, which resolves the entity
    from State, traverses its persisted `EntityLink` edges (it never reconstructs
    a relationship from matching text), and quotes what Decision and Intelligence
    stored about that exact canonical id.

    THE 404 IS TENANT-SCOPED AND THAT IS THE POINT. The lookup runs against
    `StateService.incremental_update(ctx.organization_id)`, so an entity id that
    genuinely exists in another organization is simply absent from this
    projection and answers 404 — indistinguishable, to the caller, from an id
    that exists nowhere. A cross-tenant probe learns nothing.
    """
    from navis.api.knowledge_router import knowledge_entity

    return await knowledge_entity(
        entity_type=entity_type, entity_id=entity_id, p=ctx.principal, db=db
    )


@router.get(
    "/outcomes",
    response_model=DeveloperOutcomesOut,
    responses=DEVELOPER_ERRORS_GATED,
    dependencies=[_READ_LIMIT],
    summary="What actually happened after Navis acted",
)
async def developer_outcomes(
    decision_id: uuid.UUID | None = None,
    execution_id: uuid.UUID | None = None,
    limit: int = 200,
    ctx: DeveloperContext = Depends(
        require_developer(Permission.VIEW_OBSERVATION, Capability.OBSERVATION)
    ),
    db: AsyncSession = Depends(get_session),
) -> Any:
    """Exactly what `GET /outcomes` serves. Delegates to `observation.router`.

    `decision_id` and `execution_id` are FILTERS, never scope. They are applied on
    top of the organization predicate inside `OutcomeRepository`, so an id
    belonging to another tenant matches nothing rather than reaching its rows.

    Gated on `VIEW_OBSERVATION`, which now exists. It briefly gated on
    `VIEW_DECISIONS` because the vocabulary had no observation permission; that
    was a borrowed word, and borrowing it would have widened the least-privileged
    role's meaning by way of a machine edge. See `identity.permissions`.
    """
    from navis.observation.router import list_outcomes

    return await list_outcomes(
        decision_id=decision_id,
        execution_id=execution_id,
        limit=limit,
        p=ctx.principal,
        db=db,
    )


@router.get(
    "/learning",
    response_model=DeveloperLearningOut,
    responses=DEVELOPER_ERRORS_GATED,
    dependencies=[_READ_LIMIT],
    summary="Everything Learning is willing to say about this organization",
)
async def developer_learning(
    ctx: DeveloperContext = Depends(
        require_developer(Permission.VIEW_LEARNING, Capability.LEARNING)
    ),
    db: AsyncSession = Depends(get_session),
) -> Any:
    """Exactly what `GET /learning/summary` serves. Delegates to `learning.router`.

    Reads Learning's own read contract (`learning.metrics.learning_snapshot`), so
    every rate and grouping here is Learning's derivation rather than a second
    opinion computed at this edge.

    Gated on `VIEW_LEARNING`, added to the ONE permission vocabulary rather than
    minted as a developer-only scope. There is no `developer_permissions` table
    and there must never be one: an agent and a human are authorized by the same
    words, so a grant means one thing in this system.
    """
    from navis.learning.router import learning_summary

    return await learning_summary(p=ctx.principal, db=db)


@router.get(
    "/insights",
    response_model=DeveloperInsightsOut,
    responses=DEVELOPER_ERRORS_GATED,
    dependencies=[_READ_LIMIT],
    summary="What this organization has established about itself",
)
async def developer_insights(
    ctx: DeveloperContext = Depends(
        require_developer(Permission.VIEW_INTELLIGENCE, Capability.INTELLIGENCE)
    ),
    db: AsyncSession = Depends(get_session),
) -> Any:
    """The organization's CURRENT insight projection — what `/intelligence/insights` serves.

    Reads through `IntelligenceService.get_insights` and serializes with
    Intelligence's own `_serialize`, so the Developer API and the web app cannot
    disagree about what an insight is or which generation of it is current. The
    repository owns "current" — superseded rows are excluded there, and nothing
    is re-decided at this edge.

    THE GAP THIS CLOSES. An insight was previously reachable only through
    `/developer/entities/{type}/{id}`, which requires already knowing the entity
    id to ask about. So an agent could confirm an insight it had somehow guessed
    at, and could never ask "what does this organization currently believe?".

    FACTS, NOT ADVICE. Intelligence publishes what is true of the organization
    with a confidence and its evidence; the recommendation layer is Decision, and
    it is a different route. Nothing here is rescored or reranked.

    Gated on `VIEW_INTELLIGENCE`, which already existed in the ONE vocabulary and
    had no route using it.
    """
    from navis.intelligence.repository import IntelligenceRepository
    from navis.intelligence.router import _serialize as serialize_insight
    from navis.intelligence.service import IntelligenceService

    rows = await IntelligenceService(IntelligenceRepository(db)).get_insights(
        ctx.organization_id
    )
    insights = [serialize_insight(r) for r in rows]
    return {"insights": insights, "count": len(insights)}


@router.get(
    "/decisions",
    response_model=DeveloperDecisionsOut,
    responses=DEVELOPER_ERRORS_GATED,
    dependencies=[_READ_LIMIT],
    summary="What Navis decided matters, and whether it still describes now",
)
async def developer_decisions(
    ctx: DeveloperContext = Depends(
        require_developer(Permission.VIEW_DECISIONS, Capability.DECISION)
    ),
    db: AsyncSession = Depends(get_session),
) -> Any:
    """The organization's latest brief, enriched exactly as the web edge enriches it.

    DELIBERATELY NOT `GET /brief/today`, and this is the whole design of the
    route. That handler REBUILDS a brief whose watermark has moved — it calls
    `build_brief`, which folds State, persists `reasoning_signals`, projects
    `intelligence_insights` and upserts a `daily_briefs` row. Delegating to it
    would mean a Developer READ mutates four tables and spends the pipeline,
    which breaks this module's read-only contract in the most expensive way
    available.

    So it composes the two READ halves instead: `DecisionService.latest_brief`
    (one indexed select) and `brief_router._with_executions` (the same
    obligations, provenance, action, execution and explanation enrichment the
    dashboard renders). Both are the web edge's own; neither writes.

    STALENESS IS REPORTED, NEVER REPAIRED. `current_state_watermark` reads the
    two cursor columns without folding anything, so the caller is told whether
    this brief still describes the organization. `stale` is TRI-STATE for the
    same reason `brief_router._freshness` is: null means the comparison could not
    be made, which is not the same as fresh. An agent that renders a stale brief
    as current is making a claim this route refuses to make.

    NO BRIEF IS BUILT ON READ. An organization that has never had one answers
    `brief: null` with an empty decision list — "nothing has been decided yet",
    which is a real state and must stay distinguishable from an empty brief that
    a read manufactured.
    """
    from navis.api.brief_router import _with_executions
    from navis.decisions.service import DecisionService
    from navis.orchestration.brief import current_state_watermark

    brief = await DecisionService(db).latest_brief(ctx.organization_id)
    current = await current_state_watermark(db, ctx.organization_id)
    if brief is None:
        return {
            "brief": None,
            "decisions": [],
            "count": 0,
            "as_of": None,
            "state_watermark": None,
            "current_state_watermark": current,
            "stale": None,
            "reasoning_version": None,
        }

    body = await _with_executions(db, brief)
    decisions = list(body.get("decisions") or [])
    return {
        "brief": body,
        "decisions": decisions,
        "count": len(decisions),
        "as_of": body.get("as_of"),
        "state_watermark": body.get("state_watermark"),
        "current_state_watermark": current,
        # Same tri-state rule as `_freshness.current`, inverted to name the
        # problem rather than the absence of one. None when State holds no
        # snapshot: no comparison was possible, and that is not "fresh".
        "stale": (
            None if current is None else body.get("state_watermark") != current
        ),
        "reasoning_version": body.get("reasoning_version"),
    }


@router.get(
    "/reasoning",
    response_model=DeveloperReasoningOut,
    responses=DEVELOPER_ERRORS_GATED,
    dependencies=[_READ_LIMIT],
    summary="The conclusions Navis has reached, and what they look like now",
)
async def developer_reasoning(
    signal_type: str | None = Query(
        default=None,
        description=(
            "Canonical `SignalType` value to filter on (commitment_slipping, "
            "relationship_cooling, opportunity, ...). Omit for every type."
        ),
    ),
    limit: int = Query(default=100, description=f"1-{_MAX_REASONING_LIMIT}."),
    ctx: DeveloperContext = Depends(
        require_developer(Permission.VIEW_REASONING, Capability.REASONING)
    ),
    db: AsyncSession = Depends(get_session),
) -> Any:
    """Reasoning's stored conclusions — the layer that had no reader anywhere.

    THE ONE STAGE WITH NO HTTP SURFACE. `navis/reasoning/` owns no router, so
    before this route a conclusion reached a client only as three fields buried
    inside a decision. The engines' `why`, `evidence`, `severity` and
    `confidence` — the explanation substrate an agent needs to justify acting —
    were unreachable as a set on BOTH edges.

    TWO SETS OF VALUES, REPORTED APART. `reasoning_signals` keeps each
    conclusion's FIRST observation; `reasoning_signal_observations` keeps what it
    looks like at a given snapshot. `store.list_signals` returns the identity
    rows and `store.observations_at` resolves the current ones, and this route
    publishes both rather than choosing — because choosing wrongly is a live
    defect the codebase already documents (a decision reporting 11.85 days
    against a cited signal recording 7.16).

    FAILS CLOSED ON THE JOIN. A conclusion with no observation at the current
    watermark reports `current: null`. It is never filled from an older
    observation, the newest one, or the identity row — "we did not record this
    conclusion at that snapshot" is the honest answer, and substituting a
    neighbour would rebuild the defect one layer up.

    Ranked by CURRENT severity, descending, so the caller reads the most pressing
    live conclusion first — and conclusions with no current observation sort
    last rather than competing on a stale number. The stored order
    (`state_as_of.desc`) breaks ties.

    Gated on `VIEW_REASONING` + `Capability.REASONING`, both of which already
    existed and neither of which had a route.
    """
    from navis.orchestration.brief import current_state_watermark
    from navis.reasoning.store import list_signals, observations_at

    limit = max(1, min(int(limit), _MAX_REASONING_LIMIT))
    rows = list(
        await list_signals(db, ctx.organization_id, signal_type=signal_type, limit=limit)
    )
    watermark = await current_state_watermark(db, ctx.organization_id)
    observed = await observations_at(
        db,
        ctx.organization_id,
        [r.signal_id for r in rows],
        state_watermark=watermark,
    )

    def out(row: Any) -> dict[str, Any]:
        obs = observed.get(str(row.signal_id))
        return {
            "signal_id": str(row.signal_id),
            "signal_type": row.signal_type,
            "entity_type": row.entity_type,
            "entity_id": row.entity_id,
            "title": row.title,
            "engine_name": row.engine_name,
            "engine_version": row.engine_version,
            "reasoning_version": row.reasoning_version,
            "first_observed": {
                "why": row.why,
                "confidence": row.confidence,
                "severity": row.severity,
                "evidence": row.evidence or {},
                "state_as_of": row.state_as_of.isoformat() if row.state_as_of else None,
                "state_watermark": row.state_watermark,
            },
            "current": None if obs is None else {
                "why": obs.why,
                "confidence": obs.confidence,
                "severity": obs.severity,
                "evidence": obs.evidence or {},
                "state_as_of": obs.state_as_of.isoformat() if obs.state_as_of else None,
                "state_watermark": obs.state_watermark,
            },
        }

    signals = [out(r) for r in rows]
    # Current severity ranks; an unobserved conclusion sorts last on -1.0 rather
    # than on the stale severity its identity row happens to carry. The stored
    # order is already deterministic, so `sorted` (stable) keeps it as the tiebreak.
    signals.sort(
        key=lambda s: -(s["current"]["severity"] if s["current"] else -1.0)
    )
    return {
        "signals": signals,
        "count": len(signals),
        "limit": limit,
        "signal_type": signal_type,
        "truncated": len(signals) >= limit,
        "state_watermark": watermark,
        "order": "current.severity.desc",
    }
