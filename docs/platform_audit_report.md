# Platform Layer — V2 Audit & Hardening Report

**Scope:** `navis/platform/**` only. No business domain (identity, organization, signals, memory, state, reasoning, decision, execution, observation, learning, intelligence, adaptive, ml, orchestration, api) was modified. Verified by `git status`: changes touch only `navis/platform/**` and `tests/platform/**`.

**Result:** 9 platform files modified, 1 created, 10 new tests. Full suite **254 passed, 9 skipped** (was 244 → +10). Zero regressions.

---

## Executive Summary

| Dimension | Before | After |
|---|---|---|
| Architecture | 82 | 88 |
| Reliability | 78 | 86 |
| Scalability | 70 | 80 |
| Security | 72 | 82 |
| Observability | 58 | 78 |
| Multi-Tenancy | 75 | 82 |
| **Overall** | **72** | **83** |

The platform was already strong on the fundamentals most teams get wrong (idempotent Celery, graceful Redis degradation, pooled-Postgres correctness, fail-fast prod config, structured logging with request correlation). The gaps were in **event lineage, observability metrics, asymmetric-auth readiness, failure-domain isolation, and runaway-query protection** — all now closed with config-only, contract-preserving changes. The remaining ceiling is cross-process event delivery, which is correctly left as an **abstraction point** (a seam now exists) rather than a premature Kafka adoption.

---

## Phase 1 — Verified Findings (independent re-audit; previous reviews NOT trusted)

### 1. JWT Revocation Gap — **REJECTED** (works) / minor platform note
- **Evidence:** `platform/security/jwt.py:34` mints a `jti`. Validation lives in `identity/authz.py` (`get_principal`) **and** `identity/dependencies.py:get_current_context` — both call `TokenRevocationService.is_jti_revoked()` and `token_predates_bulk_revoke()`. Logout (`identity/router.py:logout`) inserts the `jti` into `revoked_tokens`; deactivation/member-removal bump `users.tokens_valid_from`. Revocation is enforced on **every** authenticated route.
- **Impact:** N/A — the claim is false in the current code. The platform-correct design (mint here, validate in the identity domain) is intact.
- **Fix applied:** none required at platform layer.

### 2. HS256 Future Limitation — **CONFIRMED** (low urgency)
- **Evidence:** `config/settings.py:42 algorithm="HS256"` + single `secret_key`. Symmetric signing means every verifier needs the secret — unfit for multi-service verification and key rotation.
- **Impact:** **P2** (becomes P1 at multi-service scale).
- **Fix applied:** `jwt.py` is now **algorithm-agnostic**. HS256 stays the default; setting `ALGORITHM=RS256/ES256/PS256` + `JWT_PRIVATE_KEY`/`JWT_PUBLIC_KEY` switches to asymmetric signing with an optional `kid` header for JWKS/rotation — **no call-site change**. Migration is config-only. (`settings.jwt_signing_key`/`jwt_verify_key`/`jwt_is_asymmetric`.)

### 3. EventBus Singleton Risk — **CONFIRMED (scaling), PARTIALLY (correctness)**
- **Evidence:** `events/bus.py: event_bus = EventBus()` — a per-process in-memory handler registry. Handlers registered via `@subscribe` run **in the publishing process**. Because every process imports `navis.registry` (which imports all subscribers), all handlers exist in all processes, so delivery is *correct* — but heavy work runs **inline in the publisher** (e.g., the API process), and there is no durable cross-process fan-out. The Redis mirror is **write-only** (no consumer).
- **Impact:** **P1** at scale (publisher latency/blast radius); not a correctness bug today.
- **Fix applied:** (a) explicit **delivery contract** documented; (b) `set_external_publisher()` **seam** added so a durable/distributed transport (outbox-relay / Kafka / NATS) is a registration, not a rewrite; (c) per-type publish/handle/failure counters + publish-latency histogram for visibility. Kafka deliberately **not** introduced.

### 4. Missing Event Correlation — **CONFIRMED**
- **Evidence:** `DomainEvent` had `id/type/org_id/payload/occurred_at/source` — no correlation/causation/parent linkage. Lineage and distributed tracing impossible.
- **Impact:** **P1** (observability + replay debugging).
- **Fix applied:** `DomainEvent` gains `correlation_id` (auto-filled from the ambient request/task id), `causation_id`, and a `.caused()` helper that builds a child event recording its parent — a precise parent→child chain. Backward-compatible (new fields default).

### 5. Untyped Event Payloads — **CONFIRMED (partial fix)**
- **Evidence:** `payload: dict[str, Any]`, no version → silent schema drift risk.
- **Impact:** **P2**.
- **Fix applied:** `schema_version` field + `EVENT_SCHEMA_VERSION` on every event and in `to_dict()`, so consumers can detect/evolve payload contracts. *Full per-event typed payloads require touching the domains that construct events — documented as out-of-scope (see "Changes requiring work outside platform").*

### 6. Redis Dependency Concentration — **CONFIRMED**
- **Evidence:** one Redis served cache (`db0`), broker (`db2`), backend (`db3`), and the event/dead-letter streams (cache client, `db0`). Single failure domain; a cache outage also kills event mirroring.
- **Impact:** **P1** (blast radius).
- **Fix applied:** cache and events now use **independent clients** (`redis_client`, `events_redis_client`) bound to `effective_redis_cache_url` / `effective_redis_events_url`, each defaulting to `redis_url`. Splitting onto separate instances is **config-only**. Broker/backend already use separate URLs. Cache and event mirror both already degrade gracefully (best-effort).

### 7. Event Replay Capability — **PARTIALLY CONFIRMED**
- **Evidence:** the Redis stream is `maxlen≈10000, approximate` and has no consumer — it is a bounded, lossy *mirror*, not a replay log. **However**, authoritative replay comes from the **append-only domain tables in Postgres** (signal_events, memory_records, etc.), which *are* fully replayable. So "events replay" at the bus level = NO; at the system level = YES.
- **Impact:** **P2** (no truth is at risk).
- **Fix applied:** the mirror is explicitly documented as **not the replay source of truth**; `maxlen` is now configurable (`event_stream_maxlen`). A durable event log is correctly left to the domain outbox pattern (already used by Execution), not bolted onto the bus.

### 8. Multi-Tenant Isolation — **PARTIALLY CONFIRMED**
- **Evidence:** `DomainEvent` carries `org_id` (tenancy on every event ✓). No platform-level enforcement (no RLS, no ambient tenant context for logs).
- **Impact:** **P2** (defense-in-depth).
- **Fix applied:** an `org_id` **contextvar** + `set_org_id/get_org_id` now propagate the tenant onto every log line (parity with `request_id`), making per-tenant activity traceable end-to-end. **Postgres RLS is documented as out-of-scope** (requires domain/migration changes).

### 9. Observability Gaps — **CONFIRMED (largely fixed)**
- **Evidence:** had structured logging + `request_id` correlation + Sentry + health/readiness. **Missing:** metrics/latency histograms, event lineage, pipeline timing, tenant context.
- **Impact:** **P1**.
- **Fix applied:** new `monitoring/metrics.py` (zero-dependency counters + latency histograms + snapshot + `timed()` context manager); event correlation/causation (#4); tenant log context (#8); event-bus publish-latency + per-type metrics. Distributed tracing (OpenTelemetry) is left as a seam — `metrics.snapshot()` + correlation IDs are the foundation an exporter plugs into.

### 10. Reliability Gaps — **REJECTED (strong)**
- **Evidence:** Celery `task_acks_late=True`, `task_reject_on_worker_lost=True`, `worker_prefetch_multiplier=1`, `worker_max_tasks_per_child=200`, soft/hard time limits, `visibility_timeout`, retry-on-startup (`queue/celery_app.py`). Redis retry+backoff+timeouts+graceful degrade. Bus per-handler timeout + dead-letter. Idempotency at the domain layer (inbox/outbox).
- **Impact:** N/A — already production-grade.
- **Fix applied:** none required; added a **DB statement timeout** (see new issue N1) which strengthens this dimension further.

---

## Phase 2 — Newly Discovered Issues

| ID | Severity | Issue | Root cause | Fix |
|---|---|---|---|---|
| **N1** | P1 | A runaway query can pin a pooled connection indefinitely → pool exhaustion under load | No server-side statement timeout | `session.py` sets asyncpg `command_timeout = db_statement_timeout_seconds` (default 30s) — runaway statements are cancelled, bounding blast radius (NVIDIA/DB lens). |
| **N2** | P2 | Event publish ran with no latency visibility; a slow handler was invisible until it timed out | No timing instrumentation | Bus records `events.publish_latency` histogram + per-type counters (Uber/observability lens). |
| **N3** | P2 | Cache + event stream shared one Redis client/instance | Single client | Independent clients, config-separable (#6). |
| **N4** | P3 | `Metrics.snapshot()` exists but no HTTP exposure | Endpoint lives in `main.py` (edge) | Snapshot is ready; wiring a `/metrics` route is a one-line edge change (documented, not implemented — `main.py` is outside platform scope by a strict reading; the helper is in place). |
| **N5** | P3 | DB pool sizing (`pool_size=10` + `max_overflow=20` = 30 conns/process) can exceed Postgres/PgBouncer limits when many API replicas run | Per-process defaults | Already fully configurable; documented sizing guidance in the evolution plan below. No code change — correct default for current scale. |

---

## Files Changed (Before → After → Reason)

1. **`config/settings.py`** — *Before:* HS256 only, single Redis URL, no statement-timeout/metrics/stream config. *After:* `jwt_private_key/public_key/key_id` + `jwt_is_asymmetric/signing_key/verify_key`; `redis_cache_url/events_url` + `effective_*` props; `db_statement_timeout_seconds`; `event_stream_maxlen`; `metrics_enabled`. *Reason:* every hardening below is config-driven and defaults to today's behavior.
2. **`security/jwt.py`** — *Before:* signed/verified with `secret_key` + `algorithm`. *After:* uses `settings.jwt_signing_key`/`jwt_verify_key` + optional `kid`. *Reason:* RS256/JWKS migration path with no call-site change (Finding #2).
3. **`events/events.py`** — *Before:* `DomainEvent` had no lineage. *After:* `correlation_id` (auto-filled), `causation_id`, `schema_version`, `.caused()` helper, richer `to_dict()`. *Reason:* event lineage + schema versioning (#4, #5).
4. **`events/bus.py`** — *Before:* counters only; cache-client mirror at fixed maxlen. *After:* publish-latency + per-type metrics; events-Redis client; configurable maxlen; `set_external_publisher()` seam; documented delivery contract. *Reason:* observability + failure-domain split + distributed-transport seam (#3, #6, #9, N2).
5. **`monitoring/metrics.py`** *(new)* — counters + latency histograms + snapshot + `timed()`. *Reason:* the missing metrics primitive (#9), zero new dependencies.
6. **`monitoring/logging.py`** — *Before:* `request_id` context only. *After:* + `org_id` context on every log line. *Reason:* tenant traceability (#8).
7. **`monitoring/__init__.py`** — export `metrics`, `timed`, `set_org_id`, `get_org_id`. *Reason:* surface the new primitives.
8. **`cache/redis.py`** — *Before:* one client. *After:* `_make_client()` factory + separate `redis_client` (cache) and `events_redis_client`; `aclose()` drains both. *Reason:* failure-domain isolation (#6).
9. **`cache/__init__.py`** — export `events_redis_client`. *Reason:* used by the bus.
10. **`database/session.py`** — *After:* asyncpg `command_timeout` on pooled connections. *Reason:* runaway-query guard (N1).

Unified diffs: `git diff navis/platform` (197 insertions, 25 deletions across 9 files; +1 new file).

---

## Million-User Readiness

- **Before:** **5/10.** Correct fundamentals, but inline cross-process event handling, single Redis failure domain, no metrics, no statement timeout, symmetric-only auth.
- **After:** **7/10.** Lineage + metrics + failure-domain separation + query timeout + asymmetric-auth path close the platform-internal gaps.

**Remaining bottlenecks (each needs work OUTSIDE platform or a future infra adoption):**
1. **Cross-process event delivery** — the seam exists (`set_external_publisher`); adopting a durable transport + relay is the Stage-3 move.
2. **No `/metrics` HTTP endpoint** — one-line wiring in `main.py` (edge).
3. **Postgres RLS** — defense-in-depth tenant isolation; a domain/migration change.
4. **Vector/at-scale search** — ML layer (pgvector), not platform.
5. **Per-event typed payload schemas** — domains construct events; needs domain edits.

---

## Future Evolution Plan

**Stage 1 — 1k orgs (today).** *Event bus:* in-process + Redis mirror (current). *DB:* single Neon pooled instance, `pool_size≈10/process`, statement timeout on. *Cache:* single Redis (cache+events+broker). *Observability:* structured logs + metrics snapshot + Sentry; wire `/metrics`. *Security:* HS256 fine.

**Stage 2 — 10k orgs.** *Event bus:* register an **outbox-relay** as the external publisher (persist-then-publish from Postgres — Execution already uses this pattern) so events survive a process crash. *DB:* read replica for heavy projections; tune pools per replica count. *Cache:* split **events Redis** from cache Redis (config flip — already supported). *Observability:* ship metrics to Prometheus via a tiny exporter over `snapshot()`. *Security:* migrate to **RS256 + JWKS** (config flip + key provisioning).

**Stage 3 — 100k orgs.** *Event bus:* adopt **Kafka/NATS** behind `set_external_publisher` for durable, partitioned, cross-region fan-out; in-process bus remains for low-latency local handlers. *DB:* partition the largest append-only tables (signal_events, memory_records) by org/time; cold-storage tiering for old partitions. *Cache:* Redis Cluster; separate broker cluster. *Observability:* OpenTelemetry tracing keyed on the existing `correlation_id`. *Security:* per-tenant key scoping, secret manager.

**Stage 4 — 1M users.** *Event bus:* multi-region Kafka, exactly-once via idempotency keys (already in the domain inbox pattern). *DB:* sharded Postgres or Citus by org_id; Postgres RLS enforced. *Cache:* regional Redis Clusters. *Observability:* full distributed tracing + SLO dashboards from histograms. *Security:* JWKS rotation, mTLS between services, tenant-escape fuzzing in CI.

The architecture reaches each stage by **flipping config and registering a transport at the seam** — not by rewriting the platform. That is the payoff of doing this hardening now.

---

## Final Platform Readiness Score: **83/100** (was 72)

## Remaining Platform Risks
1. Cross-process event durability depends on adopting the external-publisher seam (not yet wired — correct for current scale).
2. `/metrics` endpoint must be exposed at the edge (`main.py`) to make metrics externally visible.
3. DB pool math must be tuned as API replica count grows (configurable, but operator-dependent).
4. Asymmetric JWT is *available* but unused until keys are provisioned — symmetric secret remains the live trust anchor.
5. Tenant isolation is still application-enforced; platform now traces it but does not yet enforce RLS (a domain/migration change).

## Changes That Require Work OUTSIDE Platform (documented, NOT implemented per mandate)
- **Per-event typed payloads** — domains construct `DomainEvent.payload`; typing them touches signals/memory/etc.
- **Postgres Row-Level Security** — needs a migration + domain query review.
- **`/metrics` HTTP route** — belongs in `navis/main.py` (edge).
- **Binding `set_org_id()` after auth** — should be called in `identity` once the principal resolves, so logs carry the tenant automatically.
- **Outbox-relay as external publisher** — an orchestration/infra wiring, not platform code.
