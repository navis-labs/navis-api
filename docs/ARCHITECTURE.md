# Navis v2 — Architecture

## The thesis

The bottleneck is no longer model intelligence — it's **organizational intelligence**:
memory, state, decision continuity, and learning loops. Navis is the
infrastructure that turns fragmented information into understanding, understanding
into decisions, and decisions into intelligence.

## The pipeline

```
Reality
  └─ signals/      Layer 1  capture → SignalEvent
       └─ memory/      Layer 2  entities (People, Commitments, Decisions...)
            └─ state/       Layer 3  OrganizationalState (the moat)
                 └─ reasoning/   Layer 4  ReasoningSignal (risk, priority, opportunity)
                      └─ decisions/   Layer 5  WhatMattersToday → DailyBrief
                           └─ execution/   Layer 6  human-approved Actions
                                └─ observation/ Layer 7  OutcomeObservation
                                     └─ learning/    Layer 8  confidence + patterns
                                          └─ intelligence/  Organizational Intelligence Score
```

Cross-cutting: `platform/` (infra), `identity/` (tenancy/auth), `ml/` (isolated
LLM access), `orchestration/` (workflow control), `analytics/`, `search/`.

## Why `state/` is a first-class domain

The single biggest architectural mistake is skipping State. Without it, the
decision service has to recompute the whole world on every request and becomes a
god object. With a real `OrganizationalState` aggregate in the center:

- reasoning reads **one** coherent snapshot, not raw tables;
- every domain evolves independently behind events;
- State gets history (snapshots), so we can ask "what changed since yesterday?".

## Event bus (the spine)

Every domain communicates through `platform/events`. No domain imports another
domain's service across a pipeline boundary; it emits an event and the next
context reacts.

```
GmailReceived
  → (memory)      CommitmentDetected / PersonDetected
      → (state)       StateUpdated
          → (decisions, on demand) BriefReady
              → (execution)  ActionRequested → ActionExecuted
                  → (observation) OutcomeObserved
                      → (learning)    LearningUpdated
```

Live wiring today (`orchestration/wiring.py`):
`GmailReceived → memory extracts → CommitmentDetected → state rebuild`.
Reasoning + decisions run on demand via the **BriefOrchestrator** (and the
hourly Celery beat), which keeps the morning brief cheap and deterministic.

## The five primitives (the golden rule)

Every module must improve at least one of:

1. **Memory** — what the company knows (`memory/`)
2. **State** — the living model (`state/`)
3. **Reasoning** — state → signals (`reasoning/`)
4. **Decision quality** — what matters (`decisions/`)
5. **Learning loops** — outcomes → better confidence (`observation/` + `learning/`)

If a proposed folder improves none of these, it does not belong in core Navis.

## Dependency direction

Dependencies point **down** the pipeline and **into** platform. A domain may
import the contracts of an earlier layer (decisions imports `reasoning`, which
imports `state`), never a later one. Backward coupling happens only through
events. `ml/` is a leaf every domain may call but which imports no domain.

## Request → brief, end to end

1. `POST /api/v1/integrations/seed` (or a real Gmail sync) ingests a `SignalEvent`.
2. `signals` persists it and publishes `GmailReceived`.
3. `memory.on_gmail_received` extracts people + commitments, publishes `CommitmentDetected`.
4. `orchestration.wiring` rebuilds `OrganizationalState`; State publishes `StateUpdated`.
5. `GET /api/v1/brief/today` builds state → runs reasoning engines → ranks via
   `WhatMattersTodayEngine` → returns the `DailyBrief`.
6. Founder approves an `Action`; `observation` records the outcome; `learning`
   adjusts confidence; `intelligence` re-scores.

## Local extension points

- **New signal source**: add `signals/<source>/connector.py` implementing
  `SignalConnector`, register it in `orchestration/sync.py`. Nothing downstream
  changes.
- **New risk engine**: add `reasoning/engines/<x>.py` implementing
  `ReasoningEngine`, register in `reasoning/engines/__init__.py`. It automatically
  flows into the brief.
- **Real action**: replace a `NoopHandler` in `execution/service.py` with a
  provider-backed `ActionHandler`. The approval flow is unchanged.
- **Real extraction**: have `memory` call `navis.ml.llm` to upgrade heuristic
  commitment extraction. Heuristics remain the fallback when no key is set.
