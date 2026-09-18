"""Task-aware organizational context — `POST /api/v1/developer/context`.

THE ROUTE THIS PLATFORM EXISTS FOR. Every other Developer read publishes one
layer and leaves the caller to assemble an organization out of them: fetch the
context projection, then State, then memory, then one entity per id it cares
about, and even then it cannot see what the organization concluded. This one
answers the question an agent actually has — "I am about to do X; what does this
company already know, believe and consider important about it?" — and answers it
from the same composer the Navis application uses.

    /chat                      -> build_grounded_context(...) -> LLM -> prose
    POST /developer/context     -> build_grounded_context(...) -> the bundle

ITS OWN MODULE, AND THAT IS DELIBERATE. `developer/router.py` states that every
handler in it is a GET, which is what makes its read-only guarantee auditable at
a glance; `agents_router` and `runs_router` are separate for the same reason.
This is a POST because its input is a task in prose — too large and too
structured for a query string — and NOT because it writes. It writes nothing.

READ-ONLY, PROVEN RATHER THAN ASSERTED. `read_only=True` sends the State read
through `StateService.read_only_state`, which never folds, never rebuilds and
never disposes. That matters because the alternatives do: `incremental_update`
(what `GET /developer/context` calls) saves a snapshot on every request, and
`get_state` (what `/chat` calls) deletes and rebuilds a corrupt or absent one. A
machine asking what an organization knows must not be able to trigger a full
fold of its memory by asking, so this route declines to — and reports the
absence instead of repairing it.

NO MODEL IS CALLED. The bundle IS the answer. Handing an agent generated prose
would put a Navis model between it and its own organization's facts, and the
agent already has a better one — it IS one. Navis grounds; the caller reasons.

AUTHORIZATION IS A FLOOR PLUS A FILTER, not one gate. See `_layers_for`.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from navis.developer.gateway import DeveloperContext, require_developer
from navis.developer.schemas import DEVELOPER_ERRORS_GATED, DeveloperContextBundleOut
from navis.identity.permissions import Permission
from navis.identity.throttle import rate_limit
from navis.organization.contract import Capability
from navis.orchestration.context import (
    DEVELOPER_STATE_TIMEOUT_S,
    ContextLayers,
    build_grounded_context,
)
from navis.platform.database import AsyncSessionLocal, get_session
from navis.platform.monitoring import get_logger

logger = get_logger("developer.context")

router = APIRouter(prefix="/developer", tags=["developer"])

# The most expensive Developer request: it embeds the task, runs a vector search
# over the organization's corpus and folds nothing but reads State. Counted on
# the "chat" bucket rather than "read", because that is the bucket sized for
# exactly this shape of work on the web edge.
_CHAT_LIMIT = Depends(rate_limit("chat", "rate_limit_chat_per_window"))

#: Upper bound on the task text. Long enough for a real instruction with its
#: constraints; short enough that the string being embedded is a task rather
#: than a pasted document. The retriever embeds this verbatim.
_MAX_TASK_CHARS = 2000


class DeveloperContextIn(BaseModel):
    """What an agent is about to do.

    ONE REQUIRED FIELD, and no identity anywhere. There is no `organization_id`,
    no `user_id`, no `agent_id` — the tenant is the credential's and cannot be
    named by a caller. `tests/test_architecture.py` walks this model's fields and
    fails the build if one is ever added.

    The optional fields exist only because the Core already accepts them:
    `since`/`until` narrow the retriever's candidate window, and
    `token_budget` is the bound `ContextBuilder` already enforces. Nothing here
    was invented to make the API look richer.
    """

    task: str = Field(
        min_length=1,
        max_length=_MAX_TASK_CHARS,
        description=(
            "What you are about to do, in your own words. This is embedded and "
            "searched against the organization's memory, so a sentence naming "
            "the system, the people and the change retrieves better than a "
            "keyword."
        ),
        examples=[
            "I am about to modify the billing integration. What organizational "
            "context should I know before making this change?"
        ],
    )
    since: datetime | None = Field(
        default=None,
        description="Only consider facts that happened at or after this instant.",
    )
    until: datetime | None = Field(
        default=None,
        description="Only consider facts that happened before this instant.",
    )
    token_budget: int | None = Field(
        default=None,
        ge=200,
        le=8000,
        description=(
            "Bound on the assembled bundle. Facts are admitted in relevance "
            "order until it is reached and the tail is dropped — never "
            "summarised, so a citation is never lost to make room. Defaults to "
            "the same budget the Navis application uses."
        ),
    )


#: Each optional layer, with the permission and capability that open it.
#:
#: Memory is absent because it is the FLOOR, not a layer — see the gate below.
_OPTIONAL_LAYERS: tuple[tuple[str, str, Capability], ...] = (
    ("state", Permission.VIEW_STATE, Capability.STATE),
    ("reasoning", Permission.VIEW_REASONING, Capability.REASONING),
    ("intelligence", Permission.VIEW_INTELLIGENCE, Capability.INTELLIGENCE),
    ("learning", Permission.VIEW_LEARNING, Capability.LEARNING),
)


async def _layers_for(
    ctx: DeveloperContext, db: AsyncSession
) -> tuple[ContextLayers, list[dict[str, str]]]:
    """Which layers this credential may receive, and why the rest were withheld.

    THE AUTHORIZATION MODEL, AND WHY IT IS NOT ONE GATE. Every other Developer
    route reads exactly one layer, so one permission answers it. This route
    composes five, and the two obvious designs are both wrong:

      * ONE PERMISSION for the whole route would let a credential holding only
        `view_memory` read reasoning, intelligence and decisions through a side
        door — the permission vocabulary would stop meaning anything;
      * ALL FIVE required would 403 every credential that exists today. On the
        live tenant not one key holds more than four of them, so the route the
        platform exists for would be unreachable.

    So: a FLOOR that is genuinely required, and a FILTER above it. The floor is
    memory — a bundle with no retrieved memory is not a degraded context, it is
    the projection `GET /developer/context` already serves, and there is no
    honest version of this route without it. Each further layer is included only
    if the credential holds its permission AND the organization is entitled to
    its capability.

    WITHHOLDING IS REPORTED, NEVER SILENT. A thin bundle and a quiet
    organization look identical on the wire, and an agent that cannot tell them
    apart will report "your company has no open risks" when what happened is
    that its key could not read them. Every exclusion comes back with the reason
    and the permission that would fix it.

    Fails CLOSED on an unreadable entitlement: if the capability snapshot cannot
    be read, no optional layer is included. A layer withheld costs the bundle
    some depth; a layer wrongly included is a permission that did not hold.
    """
    from navis.organization.service import OrganizationService

    try:
        capabilities = await OrganizationService(db).entitlements.snapshot(
            ctx.organization_id
        )
    except Exception:  # noqa: BLE001 — fail closed, and say so
        logger.exception(
            "developer context: entitlements unreadable org=%s", ctx.organization_id
        )
        capabilities = {}

    enabled: dict[str, bool] = {}
    withheld: list[dict[str, str]] = []
    for name, permission, capability in _OPTIONAL_LAYERS:
        has_permission = permission in ctx.permissions
        has_capability = bool(capabilities.get(capability.value))
        enabled[name] = has_permission and has_capability
        if enabled[name]:
            continue
        withheld.append({
            "layer": name,
            # The credential is the caller's to fix; the plan is the
            # organization's. Reporting them apart is the same 403-vs-402
            # distinction the gate chain makes, carried into a partial answer.
            "reason": "permission" if not has_permission else "capability",
            "requires": permission if not has_permission else capability.value,
        })
    return ContextLayers(**enabled), withheld


async def _connected_sources(db: AsyncSession, org_id: uuid.UUID) -> list[Any]:
    """The organization's integrations WITH THEIR HEALTH. Best-effort.

    Reuses `workspace_fallback.source_health` — the Navis application's own
    answer to "what is connected, and is it working?" — rather than querying
    `integrations` here. Two readers of that table would be two answers to one
    question, and this one would be the one nobody noticed drifting.
    `connected_source_labels` is derived from the same function, so the labels
    this route publishes and the ones the application shows a founder cannot
    disagree.

    Degrades to `[]` rather than failing the bundle. An empty list is
    indistinguishable from "nothing connected", which is the safe direction:
    an agent that under-trusts an absence is right more often than one that
    assumes Navis was watching a system it cannot see. Note that degrading here
    also empties `sources.detail`, which is correct for the same reason — an
    unknown health is not a healthy one, and no source at all claims nothing.
    """
    from navis.api.workspace_fallback import source_health

    try:
        return list(await source_health(db, org_id))
    except Exception:  # noqa: BLE001 — never fail the bundle over a label
        logger.exception("developer context: source health unavailable org=%s", org_id)
        return []


async def _connected_sources_isolated(org_id: uuid.UUID) -> list[Any]:
    """`_connected_sources` on a session of its own, for the concurrent path.

    The request's `db` belongs to the composer while this runs, and an
    `AsyncSession` cannot serve two coroutines at once. Opening one here — and
    closing it before returning — keeps this branch from touching it at all.

    Degrades to `[]` on any failure, including a failure to open the session,
    on exactly the terms `_connected_sources` documents: the labels explain an
    answer, they are never the answer, and no bundle is worth failing over one.
    """
    try:
        async with AsyncSessionLocal() as db:
            return await _connected_sources(db, org_id)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — never fail the bundle over a label
        logger.exception("developer context: source session unavailable org=%s", org_id)
        return []


@router.post(
    "/context",
    response_model=DeveloperContextBundleOut,
    responses=DEVELOPER_ERRORS_GATED,
    dependencies=[_CHAT_LIMIT],
    summary="What this organization knows that bears on a task",
)
async def developer_context_for_task(
    body: DeveloperContextIn,
    # THE FLOOR. `view_memory` + `memory`, matching what `/chat` gates on: this
    # route answers out of the organization's Memory, and `memory` is granted by
    # every plan including FREE, so the capability denies suspended and churned
    # tenants and nothing else. Every richer layer is filtered above.
    ctx: DeveloperContext = Depends(
        require_developer(Permission.VIEW_MEMORY, Capability.MEMORY)
    ),
    db: AsyncSession = Depends(get_session),
) -> Any:
    """Assemble the organizational context this task requires.

    Calls `orchestration.context.build_grounded_context` — the same function
    `/chat` calls, with the same retrieval, the same State fold, the same
    reasoning pass and the same token budget. A founder asking Navis and an
    agent asking Navis are grounded in one organization; only the presentation
    differs.

    WHAT COMES BACK. Cited facts, each carrying the layer that produced it
    (`memory` | `state` | `reasoning` | `intelligence` | `learning`), its source
    id and its drill-down lineage; the snapshot the answer describes; and an
    explicit account of which layers were consulted and which this credential
    could not see. No prose, no ranking of Navis's own invention, no model.

    FRESHNESS IS NOT OPTIONAL HERE. `as_of` and `state_watermark` say which
    moment the state layer describes, and `state_available: false` says the
    organization has no usable snapshot rather than nothing in it. An agent that
    presents a two-week-old picture as current is making a claim this route
    refuses to make on its behalf.
    """
    # FIRST, AND ALONE: the composer's inputs depend on it. `layers` decides
    # which truth layers are even fetched, so nothing below can start until this
    # credential's reach is known.
    layers, withheld = await _layers_for(ctx, db)
    # One wall-clock reading, shared by the time-window filter and the bundle's
    # own anchor, so "this week" and the dates on the facts cannot disagree.
    now = datetime.now(timezone.utc)

    # THEN EVERYTHING ELSE AT ONCE. The source labels and the bundle share no
    # data — one names the integrations that are connected, the other reads what
    # they produced — but they ran in series, and against a database a round trip
    # away that series WAS the latency: one query for three provider names cost
    # 2.6s of a request that had a 15s budget to meet.
    #
    # `AsyncSessionLocal` rather than this request's `db`: an `AsyncSession` is
    # not concurrency-safe, so each branch takes its own. The composition root
    # is the right place to make that choice — `navis.orchestration` and
    # `navis.ml` are handed the factory and never import one.
    sources_task = asyncio.create_task(_connected_sources_isolated(ctx.organization_id))
    try:
        composed = await build_grounded_context(
            db,
            ctx.organization_id,
            body.task,
            # THE READ-ONLY GUARANTEE. Never folds, never rebuilds, never disposes.
            # It is also what makes the concurrency below safe — see
            # `build_grounded_context`, which gates on exactly this flag.
            read_only=True,
            # NOT AN INTERACTIVE TURN, so not bounded like one. The chat bound
            # (4.0s) was sized when a warm read measured p95 1.6s; on this
            # deployment the same read is p50 3.3s with excursions past 4.0s,
            # which is what made this boundary shed the State and Reasoning
            # layers roughly one call in six. An agent request already runs
            # 17-25s end to end, so waiting for the projection is strictly
            # better for it than a fast, thinner answer. A genuinely cold fold
            # is still refused and still degrades explicitly.
            state_timeout_s=DEVELOPER_STATE_TIMEOUT_S,
            # DATA MINIMIZATION FOR AN EXTERNAL AGENT. The memory layer is
            # re-rendered from the stored record without raw message bodies:
            # what is right to EMBED for similarity is not automatically right
            # to hand a model running outside Navis. Holding `view_memory` on
            # this organization authorizes the organizational fact — the subject
            # line, the participants, the commitment — not the inbox. `/chat`
            # does not set this; see `minimize_memory_facts` for why the
            # asymmetry is deliberate.
            minimize_memory=True,
            layers=layers,
            since=body.since,
            until=body.until,
            now=now,
            sessions=AsyncSessionLocal,
            **({"token_budget": body.token_budget} if body.token_budget else {}),
        )
    except BaseException:
        # The bundle failed, so nothing will read the labels. Cancel rather than
        # leave a task holding a pooled connection until it finishes alone.
        sources_task.cancel()
        raise
    # WHICH SYSTEMS ARE ACTUALLY LOOKING, AND WHETHER THEY CAN STILL SEE. Read
    # through the same helper the Navis application uses to explain an empty
    # answer to a founder, so an agent and a human are told the same thing about
    # the same organization.
    #
    # An agent reasoning about an ABSENCE needs both halves. Navis ingests Gmail,
    # Calendar, Drive, Slack and Notion, so silence about a repository means
    # nothing was ever watching one — and silence about EMAIL means either a
    # quiet week or a revoked Google grant, which `auth_state` and
    # `last_success_at` are the only way to tell apart.
    health = await sources_task
    context = composed.context

    facts = [
        {
            "kind": f.kind,
            "source": f.source,
            "text": f.text,
            "occurred_at": f.occurred_at.isoformat() if f.occurred_at else None,
            # Prefixed, self-describing ids ("memory:<uuid>", "signal:<uuid>") —
            # the drill-down for a fact that was DERIVED rather than retrieved.
            # Empty for a fact whose single `source` already IS its provenance.
            "lineage": list(f.lineage),
            # WHICH entity the claim is about, when the producing layer knows.
            # Emitted as null — not omitted, and never guessed — when it does
            # not: an org-wide conclusion has no subject, and inventing one from
            # the claim text would be a fabricated attribution.
            "subject": (
                {"entity_type": f.subject_type, "entity_id": f.subject_id}
                if f.subject_type is not None and f.subject_id is not None
                else None
            ),
        }
        for f in context.facts
    ]
    # `unhealthy` is logged as a COUNT, not a list of providers: an operator
    # grepping for a degraded tenant needs to know that one exists, and the
    # provider names are already one query away in `integrations`.
    logger.info(
        "developer context org=%s facts=%d tokens=%d state=%s withheld=%s unhealthy=%d",
        ctx.organization_id, len(facts), context.token_estimate,
        composed.state_available, [w["layer"] for w in withheld],
        sum(1 for s in health if s.connected and s.auth_state != "ok"),
    )
    return {
        "task": body.task,
        "facts": facts,
        "count": len(facts),
        "token_estimate": context.token_estimate,
        "freshness": {
            # OBSERVATION time. Unchanged in meaning, unchanged in name, and
            # unchanged in value — every existing consumer reads what it always
            # read.
            "as_of": composed.as_of.isoformat() if composed.as_of else None,
            # EVENT time, additive. The gap between this and `as_of` is how a
            # caller tells "Navis is up to date" from "Navis is still folding,
            # but nothing new has arrived in eighteen days".
            "newest_fact_at": (
                composed.newest_fact_at.isoformat() if composed.newest_fact_at else None
            ),
            "state_watermark": composed.state_watermark,
            "state_available": composed.state_available,
            # WHY the state layer is missing, when it is: "absent" (no
            # snapshot, cold organization), "timeout" (a snapshot exists but
            # the read exceeded this caller's bound — retrying may succeed),
            # "failed" (the read raised), "withheld" (this credential may not
            # read it; also in `layers.withheld`). Null when state IS
            # available. Reported so a reader stops having to guess which of
            # four situations a bare `false` meant.
            "state_degraded_reason": composed.state_degraded_reason,
        },
        "sources": {
            # Intent, exactly as before — same labels, same order, same rule.
            "connected": [s.label for s in health if s.connected],
            # Health, additive. See `DeveloperContextSourcesOut`.
            "detail": [
                {
                    "source": s.source,
                    "label": s.label,
                    "connected": s.connected,
                    "auth_state": s.auth_state,
                    "last_attempt_at": (
                        s.last_attempt_at.isoformat() if s.last_attempt_at else None
                    ),
                    "last_success_at": (
                        s.last_success_at.isoformat() if s.last_success_at else None
                    ),
                    "last_failure_reason": s.last_failure_reason,
                }
                for s in health
            ],
        },
        "layers": {
            "included": sorted(
                [name for name, on in vars_of(composed.layers).items() if on]
                + ["memory"]
            ),
            "withheld": withheld,
        },
        "subject": (
            None if composed.subject is None or composed.subject.match is None
            else {
                "query": composed.subject.query,
                "resolved": True,
                "expanded": composed.expanded,
            }
        ),
    }


def vars_of(layers: ContextLayers) -> dict[str, bool]:
    """`ContextLayers` as a plain mapping. `slots=True` has no `__dict__`."""
    return {name: getattr(layers, name) for name in ("state", "reasoning", "intelligence", "learning")}
