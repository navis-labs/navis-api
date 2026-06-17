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

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # fill in real values
python init_db.py           # create tables
uvicorn navis.main:app --reload
```

Open http://localhost:8000/docs

## Why State is a first-class domain

The biggest architectural mistake to avoid: skipping `state/`. Without it,
the decision layer becomes a god object that has to recompute the world on every
request. With a real `OrganizationalState` aggregate in the center, every domain
evolves independently and the pipeline stays clean.
