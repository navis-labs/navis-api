"""Explicit wire contracts for the Developer API.

WHY THESE EXIST AS MODELS. The Developer boundary is the contract MCP, SDK and
CLI clients will be generated from, so every route declares a `response_model`
and the generated OpenAPI schema carries a named component for it. A handler
returning a bare `dict[str, Any]` documents nothing, and a client generator
emits `object` for it.

WHY THE BODIES ARE THE DOMAINS' OWN SHAPES. Each data model below mirrors, field
for field, what the existing domain serializer already returns — `ScopedView
.to_dict`, `OrganizationalState.to_dict`, `observation.router.serialize_outcome`,
`learning.router.serialize_snapshot`, `knowledge_router.knowledge_entity`. A
Developer-specific reshaping would be a second wire vocabulary for one set of
facts, and the two would drift the first time a domain added a field. The nested
values stay `dict[str, Any]` / `list[dict[str, Any]]` for exactly that reason: a
re-declaration of a nested domain shape here is a copy that can go stale
silently, while the top-level keys are the part the Developer API guarantees.

NOTHING HERE IS A REQUEST MODEL. The foundation is read-only, and every route's
input is a path or query parameter declared on the handler. There is deliberately
no model carrying `organization_id`, `agent_id` or `principal_id` — see the
`gateway` module note on why identity is never an input.
"""
from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class DeveloperPrincipalOut(BaseModel):
    """The caller, as the Developer boundary resolved it server-side."""

    kind: str = Field(description="user | api_key | service_account")
    subject_id: uuid.UUID = Field(
        description="The acting subject: a user, a service account, or a bare API key."
    )
    service_account_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "The machine identity acting, when one is bound to the credential. "
            "Null for a human session and for an API key with no service account."
        ),
    )
    credential_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "The api_keys row that authenticated this request — the handle for "
            "attribution, rotation and revocation. Null for a JWT session. Never "
            "a secret: this is a row id, and the key's plaintext is not stored."
        ),
    )
    permissions: list[str] = Field(
        description="Effective permissions, resolved server-side. Sorted."
    )


class DeveloperOrganizationOut(BaseModel):
    """The authenticated tenant. Derived from the credential, never from input."""

    id: uuid.UUID
    name: str
    slug: str


class DeveloperMeOut(BaseModel):
    """Identity, tenant, and what this credential may reach."""

    organization: DeveloperOrganizationOut
    principal: DeveloperPrincipalOut
    capabilities: dict[str, bool] = Field(
        description=(
            "This organization's entitlement snapshot — capability -> enabled. "
            "An entitlement, not an authorization: a capability may be true here "
            "while the credential still lacks the permission to use it."
        )
    )


class DeveloperPermissionOut(BaseModel):
    """One permission, described — and whether THIS caller may grant it.

    `grantable` is the only per-caller field, and it is computed from the
    same rule `ensure_grantable_scopes` enforces on write: a principal may
    not grant a permission it does not itself hold. It is published so a
    management screen can grey out what it cannot offer, NOT so the screen
    can decide anything — the Core re-checks every grant and refuses with
    its own 403 regardless of what was rendered.
    """

    permission: str
    kind: str = Field(description="`read`, `write` or `admin` — what holding it lets a principal do.")
    summary: str = Field(description="One sentence an operator can act on. May be empty.")
    sensitive: bool = Field(
        description=(
            "Worth a second look before granting. Advisory only — no gate "
            "consults this."
        )
    )
    assignable_to_agents: bool = Field(
        description=(
            "Whether the Developer boundary consumes this permission, and so "
            "whether granting it to an agent changes anything. A console "
            "default, not a server rule: the API accepts any valid permission "
            "the caller holds."
        )
    )
    grantable: bool = Field(
        description="Whether the calling principal holds this permission and may therefore grant it."
    )


class DeveloperPermissionCatalogOut(BaseModel):
    """The canonical permission vocabulary, as the Core defines it.

    THE SINGLE SOURCE OF TRUTH FOR EVERY MANAGEMENT SURFACE. It exists
    because the Developer Console was maintaining a hand-written copy that
    had drifted five permissions behind the boundary it described.
    """

    permissions: list[DeveloperPermissionOut]


class DeveloperMemoryRecordOut(BaseModel):
    """One immutable fact from `memory_records`, exactly as stored.

    Every field is either the fact or its provenance. Memory computes no
    interpretation and this publishes none — no score, no ranking, no verdict.
    """

    memory_id: uuid.UUID
    entity_type: str
    entity_id: uuid.UUID
    source_signal_event_id: uuid.UUID
    source_integration_id: uuid.UUID | None = None
    extractor_name: str
    extractor_version: str
    schema_version: int
    confidence: float
    event_timestamp: str | None = Field(
        default=None, description="When the fact HAPPENED (logical time)."
    )
    created_at: str | None = Field(
        default=None, description="When Navis LEARNED it (ingest time)."
    )
    dedup_key: str
    payload: dict[str, Any]


class DeveloperMemoryPageOut(BaseModel):
    records: list[DeveloperMemoryRecordOut]
    count: int
    limit: int
    entity_type: str | None = Field(
        default=None, description="The filter applied, or null for every type."
    )
    truncated: bool = Field(
        description=(
            "Whether this page is the whole answer. A client that cannot tell a "
            "full result from a truncated one will report the page size as a count."
        )
    )
    order: str = "event_timestamp.desc"


class DeveloperStateOut(BaseModel):
    """The current State V2 projection — `OrganizationalState.to_dict`, verbatim."""

    org_id: uuid.UUID
    as_of: str | None
    schema_version: int
    projection_version: int
    watermark: dict[str, Any]
    fold_frontier: dict[str, Any]
    record_count: int
    entities: dict[str, dict[str, Any]]
    counts: dict[str, int]


class DeveloperContextOut(BaseModel):
    """The intent-scoped view over State — `ScopedView.to_dict`, verbatim."""

    intent: str
    as_of: str | None
    state_watermark: str | None
    policy_version: int | None
    counts: dict[str, int]
    entities: list[dict[str, Any]]
    citations: dict[str, Any]


class DeveloperEntityOut(BaseModel):
    """One canonical entity, its persisted edges, and what the pipeline said."""

    as_of: str | None
    state_watermark: str | None
    entity: dict[str, Any]
    neighbours: list[dict[str, Any]] = Field(
        description="Resolved `EntityLink` rows. Traversed, never reconstructed from text."
    )
    decisions: list[dict[str, Any]]
    insights: list[dict[str, Any]]


class DeveloperOutcomesOut(BaseModel):
    """The append-only outcome log — `observation.router.serialize_outcome` rows."""

    outcomes: list[dict[str, Any]]
    total: int
    truncated: bool
    order: str


class DeveloperLearningOut(BaseModel):
    """Learning's read contract — `learning.router.serialize_snapshot`, verbatim.

    `totals.success_rate` is null, never 0.0, until something decisive has been
    recorded: "nothing decided yet" and "everything failed" are different facts.
    """

    org_id: uuid.UUID
    learning_version: int
    as_of: str | None
    state_watermark: str | None
    reasoning_version: int | None
    totals: dict[str, Any]
    by_surface: dict[str, Any]
    by_outcome_type: list[dict[str, Any]]
    by_signal: list[dict[str, Any]]
    patterns: list[dict[str, Any]]


class DeveloperInsightOut(BaseModel):
    """One organizational insight — `intelligence.router._serialize`, verbatim.

    A FACT the organization has established about itself, never a recommendation.
    `confidence` and `sample_size` are Intelligence's own; nothing is rescored
    here.
    """

    insight_id: str
    insight_type: str
    key: str
    title: str
    summary: str | None = None
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="The insight's own evidence, as Intelligence stored it.",
    )
    sample_size: int | None = None
    confidence: float | None = None
    supporting_learning_event_ids: list[str] = Field(default_factory=list)
    learning_version: int | None = None
    created_at: str | None = None


class DeveloperInsightsOut(BaseModel):
    """The organization's CURRENT insight projection.

    Superseded rows are already excluded by `IntelligenceRepository.get_insights`
    — this publishes that projection rather than re-deciding which generation is
    current.
    """

    insights: list[DeveloperInsightOut]
    count: int


class DeveloperReasoningSignalOut(BaseModel):
    """One stored conclusion, and what it looks like NOW.

    THE TWO SETS OF VALUES ARE REPORTED SEPARATELY AND DELIBERATELY.
    `reasoning_signals` keeps the FIRST observation of a conclusion; what it
    looks like at the current snapshot lives in `reasoning_signal_observations`.
    Collapsing them is how a live decision came to report 11.85 days against a
    cited signal recording 7.16, so this contract refuses to.

    `current` is null when no observation of this conclusion exists at the
    organization's current state watermark — an honest "we did not record this
    at that snapshot", never a substituted neighbouring observation.
    """

    signal_id: str
    signal_type: str
    entity_type: str | None = None
    entity_id: str | None = None
    title: str
    engine_name: str
    engine_version: str
    reasoning_version: int
    first_observed: dict[str, Any] = Field(
        description=(
            "What the engine concluded when it FIRST reached this conclusion: "
            "why, confidence, severity, evidence, state_as_of, state_watermark."
        )
    )
    current: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The same values at the organization's current state watermark, from "
            "`reasoning_signal_observations`. Null when this conclusion was not "
            "observed at that snapshot."
        ),
    )


class DeveloperReasoningOut(BaseModel):
    """The organization's stored reasoning, newest conclusion first."""

    signals: list[DeveloperReasoningSignalOut]
    count: int
    limit: int
    signal_type: str | None = Field(
        default=None, description="The filter applied, or null for every type."
    )
    truncated: bool
    state_watermark: str | None = Field(
        default=None,
        description=(
            "The watermark `current` was resolved at — the memory record State "
            "has folded to. Null when State holds no snapshot, in which case "
            "every `current` is null too."
        ),
    )
    order: str = "state_as_of.desc"


class DeveloperDecisionsOut(BaseModel):
    """The organization's latest brief, exactly as the web edge serializes it.

    READ-ONLY, and that is the difference from `GET /brief/today`: this never
    calls `build_brief`, so it cannot fold State, persist reasoning, project
    Intelligence or upsert a brief as a side effect of being read.

    `brief` is null when the organization has never had one built — an honest
    "nothing has been decided yet" rather than a brief manufactured by the read.
    """

    brief: dict[str, Any] | None = None
    decisions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "The brief's decisions, each with its execution, action, provenance "
            "and explanation — `brief_router._with_executions`, verbatim."
        ),
    )
    count: int = 0
    as_of: str | None = None
    state_watermark: str | None = Field(
        default=None, description="The snapshot this brief was RANKED at."
    )
    current_state_watermark: str | None = Field(
        default=None,
        description=(
            "The snapshot State has folded to NOW. When it differs from "
            "`state_watermark`, this brief describes an older organization and "
            "`stale` says so. Read WITHOUT folding anything."
        ),
    )
    stale: bool | None = Field(
        default=None,
        description=(
            "TRI-STATE, like the web edge's `freshness.current`. False: this "
            "brief describes the current snapshot. True: State has moved past "
            "it. Null: no comparison was possible because State holds no "
            "snapshot — NOT 'fresh'."
        ),
    )
    reasoning_version: int | None = None


class DeveloperGroundedFactOut(BaseModel):
    """One cited fact, exactly as `ml.rag.grounding.Grounding` carries it.

    `kind` names the LAYER that produced it, which is what lets a reader judge
    the claim: a retrieved email, a State entity, a Reasoning conclusion, a
    persisted insight and a Learning measurement are five different kinds of
    assertion and must not be flattened into one.
    """

    kind: str = Field(
        description="memory | state | reasoning | intelligence | learning"
    )
    source: str = Field(
        description="The lineage pointer — a memory_id, entity_id, signal_id or insight key."
    )
    text: str
    occurred_at: str | None = Field(
        default=None, description="When the underlying event happened, when known."
    )
    lineage: list[str] = Field(
        default_factory=list,
        description=(
            "Deeper provenance for a DERIVED fact, as prefixed ids "
            "(`memory:<uuid>`, `signal:<uuid>`, `integration:<uuid>`). A State "
            "entity is a fold over many records, so its single `source` cannot "
            "express what it was built from. Empty for a retrieved fact, whose "
            "`source` already is its provenance."
        ),
    )


class DeveloperContextFreshnessOut(BaseModel):
    """Which moment this bundle's State layer describes.

    TWO CLOCKS, AND READING ONLY ONE OF THEM IS THE MISTAKE THIS TYPE EXISTS TO
    PREVENT:

        as_of           when Navis last LEARNED something (observation time)
        newest_fact_at  when the newest thing it knows about HAPPENED (event time)

    They are both true and they are routinely far apart. Re-extraction, a
    backfill or a replay writes memory rows today that carry last month's event
    timestamps — which advances `as_of` to now while the organization's actual
    knowledge stands still. A reader given only `as_of` concludes the picture is
    current; the honest reading is `as_of` for "how recently did Navis do any
    work" and `newest_fact_at` for "how recently did anything happen".

    `as_of` MOVING WHILE `newest_fact_at` DOES NOT is the signature of a broken
    ingestion pipeline, and it is the single most useful thing on this object.
    Cross-reference `sources` before concluding an organization is merely quiet.

    `state_available: false` means the organization has no usable snapshot and
    this route DECLINED TO BUILD ONE — a read must not trigger a full fold of an
    organization's memory. It is not the same as an organization with nothing in
    it, and an agent that reports it that way is wrong.
    """

    as_of: str | None = Field(
        default=None,
        description=(
            "OBSERVATION time: when Navis last folded something into State. NOT "
            "how recent the underlying facts are — see `newest_fact_at`."
        ),
    )
    newest_fact_at: str | None = Field(
        default=None,
        description=(
            "EVENT time: when the most recent fact in this projection actually "
            "happened. Compare with `as_of` — a large gap means Navis is still "
            "working but nothing new is arriving. Null when State is "
            "unavailable or nothing has been folded yet."
        ),
    )
    state_watermark: str | None = Field(
        default=None, description="The memory record State had folded to."
    )
    state_available: bool = Field(
        description="Whether the State and Reasoning layers reached this bundle at all."
    )


class DeveloperWithheldLayerOut(BaseModel):
    """A layer this credential could not see, and what would open it."""

    layer: str
    reason: str = Field(
        description=(
            "`permission` — this credential lacks it. `capability` — this "
            "organization is not entitled to it. The same distinction 403 and "
            "402 make, carried into a partial answer."
        )
    )
    requires: str = Field(description="The permission or capability that would grant it.")


class DeveloperContextLayersOut(BaseModel):
    """What went into the bundle, and what was left out.

    WITHOUT THIS THE BUNDLE CANNOT BE READ HONESTLY. A thin answer and a quiet
    organization look identical, so a narrowly-scoped credential would report
    "no open risks" when what happened is that it may not read them.
    """

    included: list[str]
    withheld: list[DeveloperWithheldLayerOut]


class DeveloperContextSubjectOut(BaseModel):
    """Who the task was resolved to, when the identity graph found exactly one."""

    query: str | None = Field(
        default=None, description="The token that resolved, e.g. a name or an address."
    )
    resolved: bool
    expanded: bool = Field(
        description=(
            "Whether retrieval found too little about this person and the one "
            "targeted fetch-by-id hop ran."
        )
    )


class DeveloperSourceHealthOut(BaseModel):
    """One business system's intent, credential health and ingestion recency.

    THREE INDEPENDENT FACTS. They were one field (`is_active`) until 0050, which
    is why a revoked Google grant reported itself as a healthy connection for as
    long as nobody looked at the raw data.
    """

    source: str = Field(
        description="Provider id, e.g. `gmail`, `google_calendar`, `drive`, `slack`, `notion`."
    )
    label: str = Field(description="The same human-readable name used in `connected`.")
    connected: bool = Field(
        description=(
            "THE USER'S INTENT — whether this organization wants the source "
            "connected. True here with a failing `auth_state` is a real and "
            "common state: they want it, the credential stopped working."
        )
    )
    auth_state: str = Field(
        description=(
            "`ok` — the credential works. `reconsent_required` — the grant was "
            "rejected and a person must re-authorise; nothing is arriving and "
            "nothing will until they do. `revoked` — the provider reported the "
            "token as gone."
        )
    )
    last_attempt_at: str | None = Field(
        default=None,
        description=(
            "When a sync was last TRIED. Null means never. Distinguishes a "
            "healthy quiet source from one nothing is even reaching."
        ),
    )
    last_success_at: str | None = Field(
        default=None,
        description=(
            "When data last actually ARRIVED — the only field staleness should "
            "be computed from. Null means this source has never synced "
            "successfully, so its silence carries no information at all."
        ),
    )
    last_failure_reason: str | None = Field(
        default=None,
        description=(
            "What went wrong last, as a provider error code — never a token. "
            "Present alongside a healthy `auth_state` when attempts are failing "
            "for reasons other than the credential."
        ),
    )


class DeveloperContextSourcesOut(BaseModel):
    """Which business systems actually feed this organization's Navis.

    WHY AN AGENT NEEDS THIS. A coding agent asking "what does this company know
    about the payments service?" cannot tell a quiet answer from a blind one.
    If no code host, issue tracker or observability system is connected — and
    today none can be, because Navis ingests Gmail, Calendar, Drive, Slack and
    Notion and nothing else — then silence about a repository is not evidence
    that nothing was said about it. It is evidence that Navis was never looking.

    `connected` is the COMPLETE set of systems behind every fact in this
    bundle, read through the same `connected_source_labels` the Navis
    application uses to explain an empty answer to a founder. An agent that
    reasons about what is absent should read this first.

    CONNECTED IS NOT HEALTHY, and `detail` is what makes the difference
    readable. `connected` answers "did somebody set this up?" — it is the user's
    intent and nothing more. A Google grant whose refresh token has been revoked
    stays connected forever while ingesting nothing, so an agent reading only
    `connected` will treat a blind pipeline as a quiet company. `detail` carries
    the auth state and the last successful sync per source, which is what
    actually separates:

        connected + auth_state ok + recent last_success_at   HEALTHY
        connected + auth_state ok + old/absent success       STALE
        connected + auth_state reconsent_required            BROKEN
        not connected                                        DISCONNECTED

    HEALTHY + NO RESULT and BROKEN + NO RESULT are different answers to the same
    question, and only the second one means "do not trust this silence".
    """

    connected: list[str] = Field(
        description=(
            "Human-readable names of this organization's active integrations, "
            "e.g. ['Gmail', 'Google Calendar', 'Google Drive']. Empty means "
            "nothing is connected and every layer below will be thin. THIS IS "
            "INTENT, NOT HEALTH — a source listed here may be failing to sync; "
            "read `detail` before treating an absence as evidence."
        )
    )
    detail: list[DeveloperSourceHealthOut] = Field(
        default_factory=list,
        description=(
            "Per-source auth and ingestion health, including sources the "
            "organization has disconnected — 'you turned Slack off' and 'you "
            "never connected Slack' are different explanations for the same "
            "silence."
        ),
    )


class DeveloperContextBundleOut(BaseModel):
    """Task-aware organizational grounding — the composer's output, verbatim.

    NOT AN ANSWER, and that is the product boundary. No model was called: these
    are the facts Navis holds that bear on the task, each cited, so the CALLER's
    model does the reasoning. A generated narrative here would put a Navis model
    between an agent and its own organization's facts.
    """

    task: str = Field(description="Echoed so a stored bundle is self-describing.")
    facts: list[DeveloperGroundedFactOut]
    count: int
    token_estimate: int = Field(
        description=(
            "What this bundle costs to render, counted the way the budget counts "
            "it — the whole cited line per fact, not the text alone."
        )
    )
    freshness: DeveloperContextFreshnessOut
    layers: DeveloperContextLayersOut
    sources: DeveloperContextSourcesOut
    subject: DeveloperContextSubjectOut | None = Field(
        default=None,
        description=(
            "Null when the task named nobody the identity graph knows, which is "
            "the common case for an infrastructure task and is not a failure."
        ),
    )


class DeveloperErrorOut(BaseModel):
    """FastAPI's error envelope, named so the OpenAPI schema documents it."""

    detail: str


# The error responses EVERY Developer route can return, declared once and spread
# into each route's `responses=`. Documented rather than implied, because a
# generated client that does not know 402 exists will report an entitlement
# problem as an unexpected failure.
DEVELOPER_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {
        "model": DeveloperErrorOut,
        "description": (
            "No credential, an unparseable one, an expired or revoked JWT, an "
            "invalid or revoked API key, or a key whose service account was revoked. "
            "Also returned when both an API key and a bearer token are presented."
        ),
    },
    403: {
        "model": DeveloperErrorOut,
        "description": (
            "Authenticated, but this credential does not hold the permission the "
            "route requires — or the user has no membership in the organization "
            "the token names."
        ),
    },
    429: {
        "model": DeveloperErrorOut,
        "description": (
            "Per-organization rate limit exceeded. `Retry-After` names the seconds "
            "until the window rolls."
        ),
    },
}

# Routes that reach a gated domain can additionally answer 402.
DEVELOPER_ERRORS_GATED: dict[int | str, dict[str, Any]] = {
    **DEVELOPER_ERRORS,
    402: {
        "model": DeveloperErrorOut,
        "description": (
            "This organization is not entitled to the capability behind the route "
            "— it is not on a plan that includes it, a kill switch is thrown, or "
            "its lifecycle is suspended/churned. Distinct from 403, which is about "
            "the credential rather than the tenant."
        ),
    },
}
