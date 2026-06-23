# Navis — Organizational Intelligence Infrastructure

> Companies already have information. They do not have intelligence.
> Navis turns information into understanding, understanding into decisions,
> and decisions into organizational intelligence.

This is the **v2 backend**, architected as a set of bounded contexts that mirror
the organizational-intelligence pipeline:

```
Signals → Memory → State → Reasoning → Decision → Execution → Observation → Learning → Intelligence
```

## The Golden Rule

Every module must improve at least one of the five primitives:

1. **Memory** — what the company knows
2. **State** — a living model of the company right now
3. **Reasoning** — turning state into signals
4. **Decision quality** — what matters, what to ignore, what creates leverage
5. **Learning loops** — every outcome improves future decisions

If a folder improves none of these, it does not belong in core Navis.

## Architecture (bounded contexts)

| Domain | Layer | Responsibility |
|---|---|---|
| `platform/` | 0 | Shared infrastructure only — **never** business logic |
| `identity/` | — | Users, organizations, teams, auth, tenancy, integrations |
| `signals/` | 1 | Reality capture. Emits `SignalEvent`. No memory, no reasoning. |
| `memory/` | 2 | The company's brain — People, Projects, Commitments, Decisions... |
| `state/` | 3 | `OrganizationalState` — the central moat. Built every morning. |
| `reasoning/` | 4 | Engines that turn state into `ReasoningSignal`s (risk, priority...) |
| `decisions/` | 5 | `WhatMattersTodayEngine` → the `DailyBrief` the founder sees |
| `execution/` | 6 | Human-approved actions (email, tasks, meetings) |
| `observation/` | 7 | Did it work? Reply? Booked? Commitment completed? |
| `learning/` | 8 | Founder/team/decision patterns; adjusts confidence over time |
| `intelligence/` | — | Organizational Intelligence Score; adaptive recommendations |
| `ml/` | — | All LLM/embedding/extraction isolated here. Domains *ask* ml. |
| `orchestration/` | — | Workflow control only (onboarding, sync, brief). No business logic. |

Everything flows through the **event bus** (`platform/events`):

```
GmailReceived → PersonDetected / CommitmentDetected → StateUpdated
  → ReasoningGenerated → DecisionGenerated → ActionExecuted
  → OutcomeObserved → LearningUpdated
```

## Core invariants

These hold across the whole pipeline and are enforced (and tested) in the Signal layer today:

- **Capture is reality only** — `signals/` never summarizes, classifies, infers, or runs LLM
  logic. Interpretation lives downstream.
- **Deterministic identity** — `event_id = uuid5(org : integration : source : external_id : version)`.
  Re-capture and replay are idempotent; integration-scoping prevents cross-org / cross-user /
  shared-channel collisions.
- **Event-sourced & replay-safe** — append-only log; retention and GDPR erasure *tombstone*
  content, never delete the event skeleton.
- **Privacy by design** — sensitive content (email bodies, Slack messages, file/channel names)
  is encrypted at rest; identity PII is minimized; external/third-party PII is dropped.
- **Untrusted by default** — captured external content is tagged `content_trust_level=untrusted`
  / `trusted_for_execution=false`; future agents must never execute instructions found in it.

## Signal Layer (Layer 1) — production-ready

The most mature layer. The Signal → Memory contract is **frozen**.

| Capability | Where |
|---|---|
| Immutable, version-aware `SignalEvent` + integration-scoped dedupe | `signals/base.py`, `signals/models.py` |
| Transactional **outbox** + relay (crash-safe at-least-once delivery) | `signals/outbox.py` |
| **DLQ** with replay accounting + poison-loop guard | `signals/reliability/` |
| Content **encryption at rest** (Fernet; prod-mandatory keys) | `platform/security/crypto.py`, `signals/crypto.py` |
| **Governance**: retention enforcement + GDPR per-subject erasure (tombstone, audited) | `signals/governance.py` |
| Google connectors — Gmail / Calendar / Drive / Docs / Meet | `signals/{gmail,calendar,drive,meet,google}/` |
| Slack connector — Events API webhook (HMAC-verified) + REST backfill, one mapper | `signals/slack/` |
| Resilient Google transport + OAuth lifecycle (refresh, rotation, re-consent) | `signals/google/` |
| Per-integration advisory locking | `signals/locking.py` |

Memory and the layers above are in progress.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # fill in real values (see below)
alembic upgrade head        # apply migrations  (or: python init_db.py)
uvicorn navis.main:app --reload
```

Open http://localhost:8000/docs

### Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres (Neon pooled URLs supported) |
| `REDIS_URL` | cache · event stream · webhook dedupe |
| `SECRET_KEY` | JWT signing + keyed governance handles (≥32 chars in prod) |
| `FIELD_ENCRYPTION_KEYS` | **content** encryption at rest — *production boot fails if missing* |
| `SIGNALS_TOKEN_KEYS` | OAuth **token** encryption at rest (prod-mandatory) |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Google OAuth |
| `GOOGLE_PUBSUB_AUDIENCE` | Gmail push (Pub/Sub) verification |
| `SLACK_CLIENT_ID` / `SLACK_CLIENT_SECRET` / `SLACK_SIGNING_SECRET` | Slack OAuth + webhook HMAC |

Generate a Fernet key for the encryption vars:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Development

```bash
PYTHONPATH=. .venv/bin/pytest -q        # tests (in-memory fakes; no live services needed)
npx pyright                             # type checking (target: 0 errors)
alembic revision -m "describe change"   # new migration
```

Conventions: **reuse existing infrastructure over building new**; the capture layer stays free
of interpretation; every change keeps the test suite green and pyright clean.

Stack: FastAPI · async SQLAlchemy 2.0 + asyncpg · PostgreSQL (Neon) · Alembic · Pydantic v2 ·
Redis · Celery · pytest (`asyncio_mode=auto`) · pyright.

## Why State is a first-class domain

The biggest architectural mistake to avoid: skipping `state/`. Without it,
the decision layer becomes a god object that has to recompute the world on every
request. With a real `OrganizationalState` aggregate in the center, every domain
evolves independently and the pipeline stays clean.
