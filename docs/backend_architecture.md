# Navis Backend — Complete High-Level Architecture

> **Scope:** Full inventory of the Python/FastAPI backend under `navis/`. Layered
> pipeline, per-module responsibilities, external dependencies, cross-cutting
> patterns, migrations, tests, and configuration.
>
> **Sister docs:**
> - `docs/ARCHITECTURE.md` — thesis + design philosophy (why the pipeline exists)
> - `docs/frontend_api_contract.md` — HTTP request/response shapes
> - `docs/frontend_backend_flow_guide.md` — frontend architecture + flows
> - `docs/IDENTITY_DEPENDENCY_GRAPH.md` — identity module deep dive

---

## 1. System at a glance

Navis is an **organizational intelligence** pipeline. Raw provider signals are
captured immutably, distilled into a shared memory, projected into a current
state, reasoned over deterministically, and turned into a ranked brief. The
founder approves actions; provider outcomes flow back and drive learning,
insights, and adaptive policy — closing the double loop.

```
                    ┌─────────────────────────────────────────────────┐
                    │              HTTP EDGE  (navis/api)              │
                    │  auth · org · signals · brief · execution ·      │
                    │  intelligence · adaptive · webhooks · scim       │
                    └─────────────────────────────────────────────────┘
                                        │
                    ┌─────────────────────────────────────────────────┐
                    │        ORCHESTRATION  (composition only)         │
                    │  brief · sync · org_provisioning · runtime_policy│
                    └─────────────────────────────────────────────────┘
                                        │
   ┌────────────────────────────────────┴────────────────────────────────────┐
   │                          DOMAIN PIPELINE                                 │
   │                                                                          │
   │  SIGNALS ──► MEMORY ──► STATE ──► REASONING ──► DECISIONS ──► EXECUTION │
   │    (1)        (2)       (3)         (4)          (5)            (6)     │
   │                                                                          │
   │                                                          OBSERVATION ◄──┤
   │                                                             (7)          │
   │                                                              │           │
   │                                                              ▼           │
   │                                                         LEARNING (8)     │
   │                                                              │           │
   │                                                              ▼           │
   │                                                       INTELLIGENCE (9)   │
   │                                                              │           │
   │                                            ADAPTIVE ◄────────┘           │
   │                                        (policy proposals,                │
   │                                         human-gated, feeds back          │
   │                                         into DECISIONS at runtime)       │
   └──────────────────────────────────────────────────────────────────────────┘
                                        │
   ┌────────────────────────────────────┴────────────────────────────────────┐
   │  CROSS-CUTTING                                                           │
   │  identity/      — auth, sessions, integrations, RBAC                     │
   │  organization/  — profile, entitlements, lifecycle, onboarding, flags   │
   │  ml/            — LLM providers, embeddings, RAG (AI isolated here)     │
   │                                                                          │
   │  PLATFORM (infra only, no business logic)                                │
   │  database · cache · events · security · config · monitoring · queue     │
   └──────────────────────────────────────────────────────────────────────────┘
```

**Golden rule:** dependencies point **down** the pipeline and **into**
platform. Later layers may import earlier ones; earlier layers reach later
layers **only through the event bus**. `ml/` is a leaf everyone can call but
that imports no domain.

---

## 2. Layered pipeline (data flow)

```
 SIGNALS (1)                MEMORY (2)                 STATE (3)
 ───────────                ──────────                 ───────────
 SignalEvent                MemoryRecord               OrganizationalState
 ├─ event_id (UUIDv5)       ├─ memory_id (UUIDv5)      ├─ entities dict
 ├─ dedup_key               ├─ entity_type/id          ├─ watermark
 ├─ payload (encrypted)     ├─ source_signal_event_id  │  (last_ts, last_memory_id)
 ├─ external_version        ├─ provenance              ├─ projection_version
 └─ Outbox pointer          └─ confidence              └─ SnapshotRepository
                                                          (monotonic upsert,
                                                           incremental rebuild)


 REASONING (4)              DECISIONS (5)              EXECUTION (6)
 ─────────────              ──────────────             ──────────────
 ReasoningSignal[]          DailyBrief                 ExecutionRequest
 ├─ signal_id (UUIDv5)      ├─ brief_id (UUIDv5)       ├─ execution_id (UUIDv5)
 ├─ severity + confidence   ├─ decisions[]             ├─ status machine
 ├─ engine_name/version     ├─ as_of                   ├─ idempotency_key
 ├─ state_watermark         ├─ state_watermark         ├─ provider_reference
 └─ pluggable engines       └─ Adaptive policy         └─ Provider dispatch
    (Meta, Risk, ...)          injected at runtime        (Gmail, Cal, Slack)


 OBSERVATION (7)            LEARNING (8)               INTELLIGENCE (9)
 ────────────────           ──────────────             ──────────────────
 Outcome                    LearningEvent              Insight
 ├─ outcome_id (UUIDv5)     ├─ attribution             ├─ insight_type
 ├─ outcome_type            │   (reasoning_signal_ids) ├─ evidence[]
 ├─ execution_id            ├─ Aggregates (counts)     ├─ sample_size
 ├─ decision_id             ├─ Patterns                ├─ confidence
 ├─ reasoning_signal_ids    └─ LEARNING_UPDATED        └─ supporting_learning_ids
 └─ payload
                                                       Cross-cutting: ADAPTIVE
                                                       ├─ Proposal (human-gated)
                                                       └─ PolicySnapshot
                                                          (ACTIVE / DRAFT /
                                                           ROLLED_BACK)
                                                          → injected at
                                                            DECISION time
```

---

## 3. Per-module inventory

### 3.1 `navis/main.py` + `navis/registry.py` — Bootstrap & DI

| Aspect | Details |
|---|---|
| Responsibility | App factory, ORM model registration, event subscriber wiring |
| Key symbols | `create_app()`, `lifespan` async context manager |
| Public interface | `app` (FastAPI); mounts every router under `/api/v1` |
| Depends on | Every domain (via `registry.py` import side-effects) |
| External systems | None |

`registry.py` imports every domain's models (so Alembic + `Base.metadata` see
them) and every event subscriber module (so `@subscribe` decorators run). No
IoC container; the explicit import graph *is* the DI.

---

### 3.2 `navis/api/` — HTTP edge

| Aspect | Details |
|---|---|
| Responsibility | Aggregates domain routers; enforces auth/capabilities at the edge |
| Key routers | `identity_router`, `organization_router`, `signals_router`, `brief_router`, `execution_router`, `intelligence_router`, `adaptive_router`, `observation_webhooks_router`, `scim_router` |
| Notable routes | `POST /auth/signup`, `POST /auth/signin`, `GET /auth/google/callback`, `POST /auth/refresh`, `GET/PUT /org/profile`, `POST /org/onboarding`, `GET /signals/connectors`, `POST /webhooks/{provider}`, `POST /brief/generate`, `GET /brief/today`, `POST /brief/propose-actions`, `POST /execution/{id}/approve`, `GET /intelligence/insights`, `POST /adaptive/proposals` |
| Depends on | Every domain service, `Principal` dependency, capability guards |
| External systems | None (delegates only) |

---

### 3.3 `navis/identity/` — Auth, users, sessions, integrations, SSO

| Aspect | Details |
|---|---|
| Responsibility | Signup/login (password + Google SSO), sessions, refresh tokens, RBAC, audit trail, OAuth integration lifecycle |
| Key services | `IdentityService`, `GoogleAuthService`, `AuthorizationService` |
| Key models | `User`, `Organization`, `Membership`, `Integration`, `Session` |
| Public interfaces | `signup`, `authenticate`, `google_signin`, `exchange_code`, `verify_id_token`, `require_permission`, `require_capability` (FastAPI deps) |
| Depends on | `platform.security` (JWT, hashing), `platform.database`, `organization` (capabilities) |
| Depended on by | Every route (via `Principal`); `organization` bootstrap; `orchestration` for token access |
| External systems | Google OAuth 2.0 (auth URL, code exchange, ID-token verification) |

`token_valid_from` gives a bulk revocation lever (invalidates all tokens issued
before a timestamp). `google_auth.py` uses a pooled `httpx.AsyncClient`.

---

### 3.4 `navis/organization/` — Company profile, entitlements, lifecycle

| Aspect | Details |
|---|---|
| Responsibility | Org lifecycle (TRIAL → ACTIVE → SUSPENDED/CHURNED), capability entitlements, plan management, per-org feature flags, onboarding stage tracking (TTFI) |
| Key services | `OrganizationService` (front door + capability authority), `EntitlementService`, `LifecycleService`, `OnboardingService`, `FeatureFlagService`, `BillingService` |
| Key models | `Organization`, `OrganizationProfile`, `OrganizationLifecycle`, `OrganizationOnboarding`, `OrganizationEntitlement`, `FeatureFlag`, `Metric` |
| Capabilities enum | `SIGNAL_CAPTURE`, `MEMORY`, `STATE`, `REASONING`, `DECISION`, `EXECUTION`, `LEARNING`, `INTELLIGENCE` |
| Public interfaces | `can_use(org_id, capability)`, `bootstrap(org_id, plan)`, `update_profile`, `get_profile` |
| Depends on | `identity` (org creation at signup), `platform.database`, event bus |
| Depended on by | Every domain (capability gates); `orchestration.org_provisioning` |
| External systems | None (future: Stripe/billing) |

---

### 3.5 `navis/signals/` — Reality capture (production-hardened)

| Aspect | Details |
|---|---|
| Responsibility | Immutable, versioned, dedup'd, encrypted-at-rest ingestion from Gmail/Calendar/Drive/Docs/Meet/Slack; DLQ + replay; retention + GDPR erasure; transactional outbox |
| Key services | `SignalRepository.ingest`, `DeadLetterRepository`, per-provider connectors |
| Key models | `SignalEventRecord` (PK=event_id UUIDv5), `SignalDeadLetter`, `SignalOutbox` |
| Providers | `gmail/`, `calendar/`, `drive/`, `meet/` (derived from calendar), `slack/`, `google/` (shared oauth/pubsub/transport) |
| Dedup key | `(org_id, integration_id, source, external_id, external_version)` → deterministic UUIDv5 → idempotent upsert |
| Invariants | Capture-only (no LLM/no inference), event-sourced replay-safe, PII-minimized, `content_trust_level` tagged as untrusted by default |
| Depends on | `identity` (Integration OAuth tokens), `platform.database`, `platform.security` (Fernet), `platform.events` (outbox relay) |
| Depended on by | `memory` (via SignalPointerEvent outbox), `observation.webhooks` (push receivers), orchestration |
| External systems | Google APIs (Gmail messages/watch, Calendar events/watch, Drive files/watch, Docs); Google Pub/Sub (push endpoint verification); Slack Events API + REST |
| Encryption | Fernet field cipher for payload content; keys mandatory in production; graceful decrypt-fail handling |
| Governance | Retention windows + per-subject tombstones (never hard-delete); audit trail preserved |

---

### 3.6 `navis/memory/` — Facts extracted from signals

| Aspect | Details |
|---|---|
| Responsibility | Pure, deterministic transform of `SignalEvent` → `MemoryRecord[]` via per-provider extractors; append-only fact store with full provenance |
| Key symbols | `build_memory_records()` (pure), `MemoryRepository.append`, `MemoryService` (event subscriber), extractor plugins |
| Key models | `MemoryRecord` (PK=memory_id UUIDv5, `entity_type`, `entity_id`, `source_signal_event_id`, provenance, `payload` JSONB, `schema_version`, `confidence`) |
| Entity types | Person, Commitment, Project, Company, Customer, Document, Message, Relationship, Meeting |
| Extractors | Email, Calendar, Docs, Drive, Meet, Slack (all under `memory/extractors/`) |
| Depends on | `signals` (SignalEvent input), `platform.database`, `platform.security` (decrypt) |
| Depended on by | `state.v2` (streams memory in deterministic order), `ml.embeddings` |
| External systems | None (pure transform) |

`build_memory_records` is unit-testable without a database. `schema_version`
on every record enables long-horizon DLQ replay after schema evolution.

---

### 3.7 `navis/state/v2/` — Current-truth projection

| Aspect | Details |
|---|---|
| Responsibility | Deterministic, snapshot-cached projection of `OrganizationalState` from memory records; incremental updates by watermark; the single source of truth for downstream layers |
| Key services | `StateService` (rebuild_org, incremental_update, get_state), `SnapshotRepository` (monotonic upsert), `StateBuilder` |
| Key models | `OrganizationalState` (root aggregate with `entities` dict, `watermark`, `projection_version`, `schema_version`, `counts`), `StateSnapshot` |
| Typed views | `PersonState`, `CommitmentState`, `ProjectState`, `CompanyState` (thin views over raw dicts) |
| Depends on | `memory` (streamed input), `platform.database` (advisory locking, snapshot table) |
| Depended on by | `reasoning.v2`, `decisions`, `orchestration.brief`, `intelligence`, `adaptive` |
| Hardening | Streaming rebuild (O(entities) memory), snapshot validation + auto-dispose on corruption, per-org advisory lock (non-blocking), structured metrics |
| External systems | None |

**No wall-clock time in projection** — `as_of` comes from the memory tail.
Same memory tail → byte-identical state.

---

### 3.8 `navis/reasoning/v2/` — Deterministic signal generation

| Aspect | Details |
|---|---|
| Responsibility | Pure pass over `OrganizationalState` → `ReasoningSignal[]`; no LLM, no clock, no RNG; per-engine isolation (one failure doesn't blank the pass) |
| Key symbols | `run_reasoning_v2(state)`, `ReasoningEngine` interface, concrete engines (`MetaEngine`, `RiskEngine`) |
| Key models | `ReasoningSignal` (`signal_id` UUIDv5, `signal_type`, `entity_type/id`, `severity`, `confidence`, `why`, `evidence`, `engine_name`, `engine_version`, `state_watermark`, `reasoning_version`, `created_at=state.as_of`) |
| Deterministic order | `(-severity, signal_id)` — stable across runs |
| Depends on | `state.v2`, `reasoning.v2.contract` |
| Depended on by | `decisions`, `orchestration.brief` |
| External systems | None |

---

### 3.9 `navis/decisions/` — What-matters engine

| Aspect | Details |
|---|---|
| Responsibility | Convert `ReasoningSignal[]` + runtime adaptive policy → ranked `DailyBrief`; idempotent upsert; emit `BRIEF_READY` |
| Key services | `DecisionService.decide()`, `WhatMattersTodayEngine` |
| Key models | `DailyBrief` (`brief_id` = UUIDv5 of `(org, reasoning_version, state_watermark)`, `decisions[]`, `as_of`, `state_watermark`) |
| Policy injection | Runtime policy from adaptive (`brief_max_items`, `brief_min_confidence`) overrides engine defaults; `policy=None` → engine defaults |
| Depends on | `reasoning.v2`, `platform.events`, `orchestration.runtime_policy` (`resolve_policy`) |
| Depended on by | `execution` (via `propose_actions` bridge), `orchestration.brief`, `api.brief_router` |
| External systems | None |

---

### 3.10 `navis/execution/` — Action execution & provider dispatch

| Aspect | Details |
|---|---|
| Responsibility | Persist approved requests, dispatch to providers, retry with backoff, DLQ + telemetry, transactional outbox for outcome events |
| Key services | `ExecutionService` (request/approve/reject/begin/complete/fail), `execute_with_retry`, execution worker (`execute_pass`, `relay_pass`) |
| Providers | Gmail (send/draft), Calendar (create event), Slack (post message), Tasks (create) |
| Key models | `ExecutionRequest` (`execution_id` = UUIDv5 of `(org, decision_id, action_type, payload_hash)`, `status`, `idempotency_key`, `provider_reference`, `attempts`, `approved_by/at`) |
| Status machine | `REQUESTED → (APPROVED / REJECTED) → (EXECUTING → EXECUTED / FAILED / CANCELLED)` |
| Idempotency | Same `(org, decision_id, action_type, payload)` → same `execution_id` → upsert returns `created=False` on retry |
| Depends on | `platform.database`, `platform.security`, `platform.queue` (Celery), `orchestration.brief`, provider APIs |
| Depended on by | `observation` (outcome callbacks), `api.brief_router` |
| External systems | Google APIs (Gmail sendMessage, Calendar insertEvent, Tasks insert), Slack Web API (chat.postMessage) |
| Hardening | Per-provider rate-limit gates, DLQ for poison signals, max-attempt cap, `FOR UPDATE SKIP LOCKED` for horizontal scaling |

---

### 3.11 `navis/observation/` — Outcome recording

| Aspect | Details |
|---|---|
| Responsibility | Append-only, idempotent recording of outcomes with full lineage; emits `OUTCOME_OBSERVED` |
| Key services | `ObservationService.record()`, per-provider webhook handlers |
| Key models | `Outcome` (`outcome_id` = UUIDv5 of `(source_event_id, outcome_type)`, `outcome_type`, `execution_id`, `decision_id`, `reasoning_signal_ids[]`, `reasoning_version`, `state_watermark`, `source_event_id`, `occurred_at`, `recorded_at`, `payload`) |
| Sources | Gmail push, Calendar push, Slack Events API (reactions, replies, mentions); manual observation via API |
| Depends on | `platform.database`, `platform.events`, `execution` (lineage) |
| Depended on by | `learning` (subscriber), `observation.webhooks` |
| External systems | Gmail Pub/Sub push, Calendar Pub/Sub push, Slack Events API |

Lineage (`reasoning_signal_ids`) is the seam that makes learning attribution
possible: an outcome can be traced back through decision → reasoning signals.

---

### 3.12 `navis/learning/` — Feedback loop & patterns

| Aspect | Details |
|---|---|
| Responsibility | Consume `OUTCOME_OBSERVED`; attribute to signals; maintain aggregate counts (by signal/decision/outcome_type); derive patterns; emit `LEARNING_UPDATED` |
| Key services | `LearningService.record_from_outcome()`, `derive_signal_patterns()` |
| Key models | `LearningEvent`, `Aggregate` (`dimension`, `key`, `count`, `last_seen`), `Pattern` (`signal_id`, `success_rate`, `confidence`) |
| Determinism | Pure count aggregation — order-agnostic, idempotent; no wall-clock, no EMA, no RNG |
| Depends on | `observation`, `platform.events`, `platform.database` |
| Depended on by | `intelligence`, `adaptive` |
| External systems | None |

---

### 3.13 `navis/intelligence/` — Insights & scoring

| Aspect | Details |
|---|---|
| Responsibility | Deterministic projection of learning aggregates → insights; append-only |
| Key services | `IntelligenceService.project()`, `project_view()` |
| Key models | `Insight` (`insight_id` = UUIDv5 of `(org, insight_type, key, learning_version)`, `insight_type`, `title`, `summary`, `evidence[]`, `sample_size`, `confidence`, `supporting_learning_event_ids`, `learning_version`) |
| Depends on | `learning`, `platform.database` |
| Depended on by | `adaptive`, API (`GET /intelligence/insights`) |
| External systems | None |

---

### 3.14 `navis/adaptive/` — Policy proposals + human governance

| Aspect | Details |
|---|---|
| Responsibility | Deterministic projection of insights → policy proposals (`brief_max_items`, `brief_min_confidence`); human approval/rejection; active policy snapshots injected into decisions (double-loop closure) |
| Key services | `AdaptiveService.propose()` (deterministic), `.approve()`, `.reject()`, `.rollback()` (human-gated), `ProposalDraft`, policy validators |
| Key models | `Proposal` (`status ∈ {PENDING_APPROVAL, APPROVED, REJECTED}`, `proposed_changes[]`, `proposed_payload`), `PolicySnapshot` (`status ∈ {ACTIVE, DRAFT, ROLLED_BACK}`, `payload`, `created_at`) |
| Depends on | `intelligence`, `platform.database`, `identity` (approver role) |
| Depended on by | `decisions` (via `orchestration.runtime_policy.resolve_policy` reading the ACTIVE snapshot) |
| External systems | None |

Approval is explicitly time-stamped, but no wall-clock in the *logic* itself
— governance validates a proposal against the current active policy before
promoting it.

---

### 3.15 `navis/ml/` — LLM, embeddings, retrieval (AI isolated here)

| Aspect | Details |
|---|---|
| Responsibility | All AI/ML logic lives here; domains request from `ml`, they don't compute intelligence directly |
| Submodules | `providers/` (Claude, Gemini, Azure, Deterministic), `embeddings/`, `generation/`, `retrieval/`, `rag/`, `evals/` |
| Provider selection | Config-driven (API-key presence); defaults to `deterministic` (offline-testable) |
| `EmbeddingService` | Reads `MemoryRecord`, embeds via provider, stores in `memory_embeddings`; model + version pinned separately |
| Depends on | `memory`, `platform.config`, `platform.database` |
| Depended on by | `memory` (if extractors are LLM-enhanced), `ml.generation`, `ml.retrieval` |
| External systems | Anthropic API (Claude), Google Generative AI (Gemini), Azure OpenAI |

Provider is injectable at test time; embedding model version is pinned for
stability; all LLM calls are async.

---

### 3.16 `navis/orchestration/` — Composition, not business logic

| Aspect | Details |
|---|---|
| Responsibility | Assemble the pipeline; coordinate signal sync; handle org provisioning; resolve runtime policy — no domain logic |
| Modules | `brief.py` (build_brief, propose_actions), `sync.py` (signal fetching), `org_provisioning.py` (ORG_CREATED subscriber), `runtime_policy.py` (`resolve_policy`) |
| Key functions | `build_brief(db, org_id)` = State → Reasoning → Decision; `propose_actions(db, org_id)` = Decision → Execution bridge; `run_brief_for_org` (Celery wrapper) |
| Depends on | Every domain, `organization` (capabilities), `platform.database` |
| Depended on by | `api.brief_router`, `platform.queue` (Celery scheduler), event bus |
| External systems | None |

---

### 3.17 `navis/platform/` — Infrastructure only

| Submodule | Responsibility | Details |
|---|---|---|
| `database/` | Async SQLAlchemy 2.0 sessions | asyncpg driver; Neon pooler detection + prepared-statement cache disable; `pool_recycle < idle_timeout`; statement timeout via `DB_STATEMENT_TIMEOUT_SECONDS` |
| `cache/` | Redis client | Pooled; feature-flag reads, webhook dedup, token cache |
| `events/` | Async event bus | In-process concurrent handlers (per-handler timeout 30s default); one failure never blocks others; mirrors to Redis stream for out-of-process consumers; dead-letters on handler failure; Prometheus counters |
| `security/` | Crypto + JWT | Fernet (field cipher for content, token cipher for OAuth); HS256 (symmetric) or RS256/ES256 (PEM); access token 60m, refresh 30d (defaults) |
| `config/` | Pydantic v2 `Settings` | Reads `.env`; every domain imports `from navis.platform.config import settings` — single source of truth |
| `monitoring/` | Observability | `structlog`, Prometheus `/metrics`, Sentry (optional), health checks, `RequestContextMiddleware` (correlation IDs) |
| `queue/` | Celery | Broker + result backend; task registry; scheduler for periodic jobs |
| `storage/` | S3 stub | Reserved for future large-blob storage |

**No business logic** — every domain imports platform, platform imports no
domain.

---

## 4. External systems

| System | Used by | Purpose | Type |
|---|---|---|---|
| PostgreSQL (Neon) | `platform.database` (all domains) | Persistent ORM storage for every table | Database |
| asyncpg | `platform.database` | Async Postgres driver | Driver |
| Redis | `platform.cache`, `platform.events`, `signals.outbox` | Sessions, feature flags, event stream mirror, webhook idempotency, Celery broker | Cache + MQ |
| Celery | `platform.queue`, `execution.worker`, `orchestration.brief` | Background tasks + periodic scheduling | Task queue |
| Google OAuth 2.0 | `identity.google_auth`, `signals.google.oauth` | SSO + connector consent | Auth |
| Google APIs | `signals.{gmail,calendar,drive}`, `execution.providers.{gmail,calendar,tasks}` | Gmail messages/watch, Calendar events/watch, Drive files/watch, Docs, Tasks | API |
| Google Pub/Sub | `signals.google.pubsub_auth`, `observation.webhooks.{gmail,calendar}` | Push notification delivery + endpoint verification | Messaging |
| Slack OAuth 2.0 | `signals.slack.oauth` | App install + token lifecycle | Auth |
| Slack Events API + Web API | `signals.slack.{events,client}`, `observation.webhooks.slack` | Webhook events (message, reaction, mention); REST backfill | API + Webhooks |
| Slack signing secret | `signals.slack.signature` | HMAC-SHA256 webhook verification | Security |
| Anthropic Claude API | `ml.providers.claude` | LLM (optional; key-gated) | LLM |
| Google Generative AI (Gemini) | `ml.providers.gemini` | LLM (optional; key-gated) | LLM |
| Azure OpenAI | `ml.providers.azure` | LLM + embeddings (optional; endpoint+key gated) | LLM + Embeddings |
| Sentry | `platform.monitoring.sentry` | Error tracking (optional) | Monitoring |
| Prometheus | `platform.monitoring.metrics` (`/metrics`) | App metrics | Metrics |
| httpx | `identity.google_auth`, `signals.google.{oauth,transport}` | Async HTTP client | HTTP |
| cryptography (Fernet) | `platform.security.crypto`, `signals.crypto` | Symmetric encryption at rest | Encryption |

---

## 5. Cross-cutting patterns

### 5.1 Repository pattern (data access isolation)
Every domain has its own `Repository` class. Repositories are injectable
(testable), not singletons — services accept them via constructor.

### 5.2 Registry via import side-effects
`navis/registry.py` imports every ORM model + subscriber. ORM imports register
tables on `Base.metadata`; subscriber imports fire `@subscribe` decorators. The
import graph *is* the DI. `main.py` imports `registry` once during startup.

### 5.3 Async event bus with in-process handlers + Redis mirror
Every state change emits a `DomainEvent`. Handlers are decorated with
`@subscribe(EventType.X)` and run concurrently under a per-handler timeout
(30s default). Failures are logged + dead-lettered — one bad handler never
blocks the others. Events also mirror to a Redis stream for out-of-process
consumers.

Canonical event chain:
```
SIGNAL_CAPTURED → MEMORY_RECORDED → STATE_UPDATED → BRIEF_READY
                                                     → ACTION_REQUESTED
                                                        → ACTION_EXECUTED
                                                           → OUTCOME_OBSERVED
                                                              → LEARNING_UPDATED
                                                                 → INSIGHT_PROJECTED
                                                                    → PROPOSAL_READY
```

### 5.4 Deterministic IDs (UUIDv5 everywhere)
Every fact/event has an ID derived from its identity tuple:

| Layer | UUIDv5 identity tuple |
|---|---|
| SignalEvent | `(org, integration, source, external_id, external_version)` |
| MemoryRecord | `(org, signal_event, extractor, entity_type, dedup_key)` |
| ReasoningSignal | `(org, reasoning_version, engine, entity_id, dedup_key)` |
| ExecutionRequest | `(org, decision_id, action_type, payload_hash)` |
| Outcome | `(source_event_id, outcome_type)` |
| Insight | `(org, insight_type, key, learning_version)` |

Same input → same ID → upsert-safe replay.

### 5.5 Append-only immutability
`signal_events`, `memory_records`, `outcomes`, `learning_events`, `insights`,
`proposals`, `policy_snapshots` are all append-only. Retention/GDPR uses
tombstones (skeleton preserved, sensitive fields erased) — never hard-delete.

### 5.6 Encryption at rest
Fernet symmetric ciphers:
- **Field cipher** — signal payload content (email bodies, Slack messages)
- **Token cipher** — Integration OAuth tokens

Keys via `FIELD_ENCRYPTION_KEYS` / `SIGNALS_TOKEN_KEYS` env vars, comma-separated,
newest first for rotation. Failed decrypt → log warning + empty string, don't
crash ingestion.

### 5.7 Transactional outbox
Every state change that publishes an event also writes to a `*_outbox` row in
the same transaction. A relay worker publishes `WHERE published_at IS NULL`
then marks them delivered. Consumers dedupe by aggregate ID. **Guarantee:**
no write without notification; at-least-once delivery.

### 5.8 One-way pipeline coupling
Domain A imports domain B **only if B is upstream** of A. There are no
cycles. Reverse coupling happens only through the event bus. `orchestration/`
is the glue that composes upstream domains without business logic.

### 5.9 Capability entitlements at the request edge
Every domain-defining route is gated:
```python
p: Principal = Depends(require_capability(Capability.DECISION))
```
`Capability` enum lives in `navis/organization/contract.py`. Plans map to
entitlements at signup.

### 5.10 Dependency injection for testing
Services accept injectable collaborators:
- `ExecutionService(db=None, *, repo=None, clock=_utcnow)`
- `StateService(db=None, *, reader=None, snapshots=None)`
- `LearningService(db=None, *, repo=None, publisher=None)`

Enables unit tests without a database or event bus.

### 5.11 Idempotent upsert (`ON CONFLICT DO UPDATE`)
Deterministic ID + `pg_insert(...).on_conflict_do_update(...)` = replay-safe
writes. Reasoning replay updates the brief without duplicating rows.

### 5.12 Structured logs + correlation IDs
`RequestContextMiddleware` injects a `correlation_id` into `structlog`
context. Event bus logs it alongside `event_type`, `org_id`, `event_id`. One
request → one correlation ID → all logs and events it spawned traceable.

---

## 6. Schema & migrations

**Alembic:** `migrations/versions/` — 30 migrations, forward-only in prod,
each reversible in dev. ORM tables auto-registered via `registry.py`; Alembic's
`autogenerate` picks them up.

**Table families:**
- Identity — `users`, `sessions`, `memberships`, `integrations`
- Organization — `organizations`, `organization_profiles`, `organization_lifecycles`, `organization_onboarding`, `organization_entitlements`, `feature_flags`
- Signals — `signal_events`, `signal_dead_letters`, `signal_outbox`
- Memory — `memory_records`, `memory_embeddings`
- State — `state_snapshots`
- Decisions — `decisions`
- Execution — `execution_requests`
- Observation — `outcomes`
- Learning — `learning_events`, `aggregates`, `patterns`
- Intelligence — `insights`
- Adaptive — `proposals`, `policy_snapshots`
- Audit — identity audit tables

---

## 7. Test structure

```
tests/
  adaptive/          — ProposalDraft logic, governance
  decisions/         — WhatMattersTodayEngine ranking + filtering
  execution/         — state machine, provider dispatch
  identity/          — signup, auth, refresh, SSO
  integration/       — end-to-end signals → memory → state → reasoning → decision
  intelligence/      — insight projection
  learning/          — aggregation + pattern derivation
  memory/            — extractors (pure SignalView → MemoryRecord[])
  ml/                — provider selection, embedding service
  observation/       — outcome idempotency, webhook parsing
  organization/      — capability, entitlements, onboarding
  platform/          — session, event bus, crypto, JWT
  reasoning/         — engine output determinism
  signals/           — dedup, connector mocks, DLQ replay
  state/             — projection determinism, snapshot caching, watermarks
  test_architecture.py
  test_pipeline.py
```

**Strategy:** in-memory fakes, pytest-asyncio fixtures, determinism assertions
(same input → same output), no mandatory external services.

---

## 8. Configuration (env-driven)

`navis/platform/config/settings.py` — Pydantic v2, reads `.env`, case-insensitive.

| Group | Keys (partial) |
|---|---|
| App | `APP_NAME`, `VERSION`, `DEBUG`, `SECRET_KEY`, `API_V1_PREFIX`, `CORS_ORIGINS` |
| Database | `DATABASE_URL` (async postgresql://), `POOL_SIZE`, `MAX_OVERFLOW`, `POOL_RECYCLE`, `DB_STATEMENT_TIMEOUT_SECONDS` |
| Redis | `REDIS_URL`, `REDIS_CACHE_URL`, `REDIS_EVENTS_URL` |
| Security | `ACCESS_TOKEN_EXPIRE_MINUTES` (60), `REFRESH_TOKEN_EXPIRE_DAYS` (30), `JWT_ALGORITHM` (HS256/RS256), `JWT_PRIVATE_KEY`, `JWT_PUBLIC_KEY`, `FIELD_ENCRYPTION_KEYS`, `SIGNALS_TOKEN_KEYS` |
| ML | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_AI_API_KEY`, `AZURE_OPENAI_*` |
| OAuth | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, `SLACK_CLIENT_ID`, `SLACK_CLIENT_SECRET`, `SLACK_SIGNING_SECRET`, `SLACK_SIGNING_SECRETS` (rotation) |
| Signal ingest | `SIGNALS_INGEST_BATCH_SIZE` (500), `SIGNALS_SYNC_TIMEOUT_SECONDS` (120), `SIGNALS_DLQ_MAX_REPLAY_ATTEMPTS` (5), `SIGNALS_REJECT_LEGACY_PLAINTEXT` |
| Provider limits | `CALENDAR_FULL_SYNC_WINDOW_DAYS` (60), `CALENDAR_MAX_SYNC_PAGES` (1000), `DRIVE_ONBOARD_FILE_CAP` (2000), `GMAIL_ONBOARD_MESSAGE_CAP` (500), `SLACK_BACKFILL_MESSAGE_CAP` (1000) |
| Celery | `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` |
| Monitoring | `LOG_LEVEL`, `SENTRY_DSN`, `METRICS_*` |

---

## 9. Request → brief, end-to-end

1. `POST /api/v1/integrations/seed` (or a real Gmail sync) ingests a `SignalEvent`.
2. `signals` persists it (encrypted payload, dedup'd event_id) and publishes `SIGNAL_CAPTURED` via outbox.
3. `memory` subscriber extracts people/commitments → `MemoryRecord[]` appended → `MEMORY_RECORDED` published.
4. `orchestration.wiring` (or the hourly Celery beat) triggers a State incremental update → `STATE_UPDATED`.
5. `GET /api/v1/brief/today` calls `orchestration.build_brief(db, org)` — reads state, runs `run_reasoning_v2`, calls `WhatMattersTodayEngine` (with adaptive policy injected), upserts `DailyBrief`, emits `BRIEF_READY`.
6. Founder approves via `POST /api/v1/brief/propose-actions` → `execution` upserts `ExecutionRequest` → worker dispatches to Gmail/Calendar/Slack/Tasks → emits `ACTION_EXECUTED`.
7. Provider webhook fires → `observation.webhooks.*` records `Outcome` with lineage → `OUTCOME_OBSERVED`.
8. `learning` subscriber increments aggregates + derives patterns → `LEARNING_UPDATED`.
9. `intelligence` projects insights; `adaptive` synthesizes proposals from insights (deterministic) and holds them pending human approval.
10. Approved proposal → new `PolicySnapshot ACTIVE` → next brief run reads it via `resolve_policy` → **the loop closes**.

---

## 10. Extension points

- **New signal source** — add `signals/<source>/connector.py` implementing `SignalConnector`, register in `orchestration/sync.py`. Nothing downstream changes.
- **New reasoning engine** — add `reasoning/v2/engines/<x>.py` implementing the engine interface, register in `reasoning/v2/engines/__init__.py`. It flows into the brief automatically.
- **Real action** — replace a `NoopHandler` in `execution/service.py` with a provider-backed `ActionHandler`. Approval flow unchanged.
- **Real extraction** — have `memory` call `navis.ml.llm` to upgrade heuristic commitment extraction. Heuristics remain the fallback when no API key is set.
- **New LLM provider** — add `ml/providers/<name>.py`, register in the provider registry. Provider selection is config-driven.
- **New capability gate** — add a value to `organization/contract.Capability`, wire into plan → entitlements bootstrap, decorate the route with `require_capability(...)`.

---

## 11. Architectural strengths

1. **Strict one-way pipeline** — no cycles; each layer's contract is independently testable.
2. **Deterministic everywhere** — UUIDv5 dedup keys, no wall-clock in projections, pure fold functions → safe replay.
3. **Append-only + tombstones** — auditability + GDPR without contortions.
4. **Encryption by default** — Fernet field + token ciphers; graceful decrypt-fail.
5. **Loose coupling via event bus** — domains evolve independently; async handlers isolated.
6. **Observable** — structured logs, correlation IDs, Prometheus, Sentry, health checks.
7. **Testable** — in-memory fakes, injectable deps, deterministic fallback ML provider.
8. **AI isolated** — `ml/` is the single seam; the rest of the system is pure Python + SQL.

---

## 12. Known follow-ups

1. **Cross-process event delivery** — current bus is in-process; a durable queue (Kafka, NATS) unlocks multi-region scaling.
2. **Vector DB** — embeddings currently live in Postgres; a dedicated vector store (Pinecone, Weaviate) would scale retrieval.
3. **Circuit breakers** — external calls (Google, Slack, LLM) would benefit from explicit breakers.
4. **Global rate limiter** — coordinate across providers, not per-provider.
5. **OpenTelemetry** — correlation IDs exist; OTel would surface service dependencies.
6. **Formal schema registry** — signal `schema_version` is per-event; a central registry would help evolution.
7. **Secrets rotation** — key rotation via CSV env vars works, but Vault/KMS would be smoother.
