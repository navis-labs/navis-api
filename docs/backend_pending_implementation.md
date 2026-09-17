# Backend — Pending Implementation Tracker

Rolling doc for backend work that is identified but not yet implemented.
Add new entries at the bottom using the template. Update the status field as
work progresses so this stays useful.

**Status legend:** `PENDING` (not started) · `IN_PROGRESS` · `BLOCKED` · `DONE`

---

## Index

| # | Title | Status | Priority | Owner |
|---|-------|--------|----------|-------|
| 1 | Gmail Pub/Sub real-time push wiring | PENDING | High | — |
| 2 | Calendar watch + webhook receiver | PENDING | High | — |
| 3 | Refresh-token audit + re-consent debug | PENDING | Medium | — |
| 4 | Multi-email self-detection (viewer alter-egos) | PENDING | Low | — |
| 5 | Parser bug — malformed emails leaking into memory | PENDING | Low | — |
| 6 | Reply latency computation for People cards | PENDING | Medium | — |
| 7 | Team domain detection (roleHint="team") | PENDING | Medium | — |
| 8 | LLM hybrid in reasoning engines | PENDING | Roadmap | — |
| 9 | RAG expansion beyond `/chat` (briefs + context) | PENDING | Roadmap | — |
| 10 | Reasoning engine protocol refactor | PENDING | Roadmap | — |
| 11 | User dismissal tracking → eval dataset | PENDING | Roadmap | — |

---

## 1. Gmail Pub/Sub real-time push wiring

**Status:** PENDING
**Priority:** High
**Discovered:** 2026-07-26

### Problem
New emails do not appear on the dashboard in real time. User has to
manually re-sync (or wait for the 5-minute Celery beat cycle) to see
newly arrived mail.

### Root cause
`navis/signals/gmail/connector.py:175-181` reads the Pub/Sub topic
directly from `os.environ.get("GOOGLE_PUBSUB_TOPIC")`. The variable is:
- NOT declared in `navis/platform/config/settings.py`
- NOT present in `.env` or `.env.example`

Result: `create_watch()` silently returns `{}` and logs a warning. No
Gmail watch channel is ever registered with Google, so Google never
sends push notifications to `POST /webhooks/google/pubsub`
(`navis/signals/router.py:351-397` — receiver is correct, just never
invoked).

### Files to change
- `navis/platform/config/settings.py` — add typed setting
- `navis/signals/gmail/connector.py:176` — read from `settings` instead of `os.environ`
- `.env.example` — document the variable
- `docs/backend_architecture.md` (optional) — mention Pub/Sub dependency

### Implementation steps
1. **Add setting.** In `settings.py`:
   ```python
   google_pubsub_topic: str | None = None
   # Fully-qualified GCP topic, e.g.
   # projects/<gcp-project>/topics/<topic-name>
   ```
2. **Switch connector to settings.** Replace
   `os.environ.get("GOOGLE_PUBSUB_TOPIC")` with
   `settings.google_pubsub_topic`. Keep the "skip if unset" fallback
   for local dev, but consider surfacing the missing config on
   application startup so it can't be silently forgotten in prod.
3. **Document in `.env.example`** with a comment explaining that
   Pub/Sub is required for real-time Gmail push and how to obtain the
   topic name.
4. **Startup check (optional but recommended):** in a startup hook,
   if the app is running with `ENV=production` and
   `google_pubsub_topic` is `None`, log a loud warning (or fail fast).

### GCP setup (outside code)
Real-time push will not work until the following is done in Google
Cloud, regardless of code changes:
1. Create a Pub/Sub **topic** in the GCP project used by the OAuth
   client. Name it e.g. `gmail-push`.
2. Grant Gmail permission to publish: add
   `gmail-api-push@system.gserviceaccount.com` as **Pub/Sub Publisher**
   on the topic.
3. Create a **push subscription** on the topic whose push endpoint is
   the public URL of `/webhooks/google/pubsub` (use ngrok /
   cloudflared for local dev).
4. Set `GOOGLE_PUBSUB_TOPIC=projects/<project>/topics/gmail-push` in
   the environment.
5. Existing connected Gmail integrations need `create_watch()` called
   once so Google starts pushing. The 6-hour renewal beat
   (`navis/orchestration/sync.py` — watch renewal path) will handle
   re-registration going forward.

### Verification
- Connect Gmail with the env var set → logs should show the watch
  being registered (no `"gmail watch skipped"` warning).
- Send an email to the connected account → within seconds,
  `/webhooks/google/pubsub` should receive a POST; a sync task should
  be enqueued; the dashboard should show the new mail without manual
  refresh.
- Check `Integration.sync_state["watch"]` in DB — should contain
  `historyId`, `expiration`, and `topic`.

### Related references
- `navis/signals/router.py:351-397` — Pub/Sub webhook receiver
- `navis/signals/google/pubsub_auth.py` — receiver authentication
- `navis/orchestration/sync.py` — watch renewal beat
- Google docs: <https://developers.google.com/gmail/api/guides/push>

---

## 2. Calendar watch + webhook receiver

**Status:** PENDING
**Priority:** High
**Discovered:** 2026-07-26

### Problem
New / updated calendar events do not appear on the dashboard in real
time. Same symptom as Gmail, different root cause.

### Root cause
`CalendarConnector` (`navis/signals/calendar/connector.py`) inherits
the no-op `create_watch()` from the base class
(`navis/signals/base.py:269-271`) which returns `{}`. Unlike Gmail,
calendar has **no watch implementation at all**, even though
`CalendarClient.watch()` exists (`navis/signals/calendar/client.py:46-51`).

There is also no webhook receiver route for Google Calendar push
channels; the only Google receiver today is the Gmail Pub/Sub one.

### Files to change
- `navis/signals/calendar/connector.py` — implement `create_watch()`
- `navis/signals/router.py` — add `POST /webhooks/google/calendar`
  receiver
- `navis/platform/config/settings.py` — add
  `google_calendar_webhook_url: str | None = None` (target URL Google
  will POST to; needs to be publicly reachable)
- `navis/orchestration/sync.py` — ensure the existing watch renewal
  loop picks up calendar as well (Google Calendar channels expire
  after ~7 days)
- `.env.example` — document

### Implementation notes
Calendar push uses a different mechanism than Gmail:
- It is a per-channel HTTP callback (not Pub/Sub).
- Each channel has a UUID `channel_id` and an optional `token` for
  verification; both must be stored so the webhook receiver can
  authenticate incoming POSTs.
- Google sends `X-Goog-Channel-ID`, `X-Goog-Resource-State`,
  `X-Goog-Resource-ID` headers. The receiver should match
  `X-Goog-Channel-ID` back to an `Integration` and enqueue a sync
  task (do NOT trust the body — Google may send an empty body; use
  the headers as the signal).
- The `sync` header value `X-Goog-Resource-State: sync` is the
  initial handshake and can be ack'd without doing anything.

### Verification
- After connecting Google Calendar, `Integration.sync_state["watch"]`
  should contain channel_id, resource_id, expiration.
- Create / modify a calendar event → within seconds, the webhook
  receiver should log the incoming POST, enqueue sync, and the event
  should appear on the dashboard without manual refresh.
- Renewal beat should re-register expiring channels before 7 days.

### Related references
- `navis/signals/calendar/client.py:46` — `watch()` client method
- `navis/signals/reliability/watch_recovery.py` — renewal path used
  by Gmail; should be extensible to calendar
- Google docs: <https://developers.google.com/calendar/api/guides/push>

---

## 3. Refresh-token audit + re-consent debug

**Status:** PENDING
**Priority:** Medium
**Discovered:** 2026-07-26

### Problem
User reports that pressing "sync" occasionally redirects them to
Google's OAuth consent screen instead of running a silent sync.

### What was verified as correct
The backend OAuth chain is functioning as designed:
- `navis/signals/google/oauth.py:70,72` — `access_type=offline` +
  `prompt=consent` are both set on the authorize URL.
- `navis/signals/router.py:299,326` — refresh_token is encrypted and
  persisted with a `coalesce()` fallback so re-consent that omits
  `refresh_token` doesn't null out the stored one.
- `navis/signals/google/transport.py:111-115` — 401 triggers a silent
  refresh + single retry.
- `navis/signals/crypto.py:76-80` — `encrypt(None)` returns `None`,
  so the coalesce path is safe.

### Likely real causes (still to confirm)
1. **DB refresh_token is NULL** for the affected integration row —
   e.g., the account was previously authorized under a Google project
   that shared consent, so Google returned `null` for
   `refresh_token`. Query:
   ```sql
   SELECT provider, refresh_token IS NULL AS no_rt
   FROM integrations
   WHERE user_id = '<user-uuid>';
   ```
   If `no_rt = true` for any row, that integration is stuck — the
   only recovery is revoke-and-reconnect from the user's Google
   account settings.
2. **Frontend "Sync" button is wired to the authorize URL** instead
   of `POST /integrations/google/sync`. Verify with Chrome DevTools
   Network tab.
3. **Token cipher key rotation** invalidated old encrypted
   refresh_tokens. Check whether the Fernet key set has changed
   without a re-encryption migration.

### Action items
- Add a small ops query / admin endpoint that reports, for a given
  user, the state of each integration row (has refresh_token? last
  successful refresh timestamp?).
- Log a distinct metric when `ReconsentRequired` is raised so we can
  count how often this is actually hitting users vs. an
  frontend-only redirect.

### Related references
- `navis/signals/google/oauth.py:44-49` — `ReconsentRequired` exception
- `navis/signals/router.py:956-1069` — manual sync endpoint

---

## 4. Multi-email self-detection (viewer alter-egos)

**Status:** PENDING
**Priority:** Low
**Discovered:** 2026-07-26

### Problem
The viewer's OWN email (e.g. `me@abmrc.com`) is hidden from
`/api/v1/memory/people` (Fix 3), but any *alter-ego* address the same
human uses (e.g. `mulayamabmrc@gmail.com` for the founder logged in as
`abmrc.sw@abmrc.com`) still shows up as a distinct "person" in the
dashboard. Founder sees themselves as a top contact in their own
People panel. Cosmetic only — not a data-integrity issue.

### Root cause
`memory_router._viewer_email()` (`navis/api/memory_router.py`) resolves
exactly one email — `principal.user.email`. There is no identity
resolution layer wiring alternate emails to the same user. The
`identity_resolution` domain exists (`navis/identity_resolution/`) but
is used for cross-signal person merging, not for self-detection.

### Files to change
- `navis/identity/models.py` — add a `UserAlternateEmail` table (or
  reuse an existing membership/identity secondary email field if one
  exists — audit first)
- `navis/api/memory_router.py:_viewer_email()` — return a `set[str]`
  of the viewer's known emails instead of a single string; filter
  `email.lower() in viewer_emails` in `list_people`
- `tests/api/test_memory_router.py::test_people_hides_viewer_own_email`
  — extend to cover multiple emails

### Implementation steps
1. Decide the storage shape (dedicated table vs. JSON column on `users`).
2. Add a simple admin/self-service endpoint to add an alternate email.
3. Update `_viewer_email()` to load and return all of them.
4. Consider a separate roleHint `"self"` if the frontend wants to
   *show* the alter-egos with a special badge instead of hiding them.

### Verification
- Log in as email A, add email B as an alternate → GET
  `/memory/people` — neither A nor B appears in the list.
- Unit test with a `Principal` whose user has two known addresses.

### Related references
- `navis/api/memory_router.py:170-176` — current `_viewer_email()`
- `navis/identity_resolution/` — existing cross-signal resolver (may
  or may not be the right hook)

---

## 5. Parser bug — malformed emails leaking into memory

**Status:** PENDING
**Priority:** Low
**Discovered:** 2026-07-26

### Problem
The founder's `memory_records` contains 3 rows with obviously broken
email strings:
- `'ajiteshkr533@gmail.com'}`  (quotes + close brace)
- `'probe`                     (half an email)
- `'probe-sender@example.com'}`

These are now filtered from the dashboard by the shape guard in
`_is_automated_email()` (rows that fail `_VALID_EMAIL_RE.match()`
are dropped) — so no user-facing impact today — but the underlying
rows persist in `memory_records` and waste space + confuse audits.

### Root cause (suspected — not confirmed)
Two candidates, one or both may be true:

1. **Test/probe leakage.** A local probe/test script pushed
   `payload={"from": "'probe-sender@example.com'}` (Python repr of a
   set literal) through the ingestion path in dev, and those rows
   made it to the same DB the dashboard is reading. The trailing
   `}` and leading `'` are exactly what `repr()` of a set produces.
2. **Gmail parser edge case.** `navis/signals/gmail/parser.py`
   occasionally stores the raw envelope header verbatim (with
   surrounding quotes) instead of running it through
   `email.utils.parseaddr()`. Would require a specific malformed
   header from Google to reproduce.

### Files to investigate
- `navis/signals/gmail/parser.py` — verify EVERY code path passes
  through `parseaddr()` before storing
- `navis/memory/extractors/email.py:_addresses()` — this DOES call
  `parseaddr()`, but if the upstream signal payload already contains
  the corrupt string, `parseaddr("'probe-sender@example.com'}")`
  returns `("", "'probe-sender@example.com'}")` — garbage in,
  garbage out
- Look for any `probe` / `example.com` in `scripts/`, `tests/`, and
  `navis/observation/` — the leak source is probably a dev tool

### Implementation steps
1. Query the DB for the offending rows and read their
   `source_signal_event_id` — inspect the signal_event to confirm
   whether the payload arrived corrupt or was corrupted during
   extraction.
2. If it's ingestion, add an input-validation guard in the extractor
   that skips malformed addresses and logs a metric.
3. Once fixed, add a governance/scrub migration that tombstones the
   existing bad rows (mirror the pattern in
   `MemoryRepository.scrub_payload`).

### Verification
- Re-run the sync end-to-end with the guard in place → no new rows
  with quoted/braced local parts.
- Confirm the 3 known bad rows are tombstoned.

### Related references
- `navis/memory/repository.py:scrub_payload` — governance-only
  in-place rewrite path
- `navis/memory/extractors/email.py:_addresses` — the current
  address parser

---

## 6. Reply latency computation for People cards

**Status:** PENDING
**Priority:** Medium
**Discovered:** 2026-07-26

### Problem
`/api/v1/memory/people` returns `"replyLatencyHours": null` for every
person. The dashboard's "you reply to X in ~4h" hint is therefore
never populated — it silently falls back to a placeholder or an empty
badge.

### Root cause
Reply latency requires per-message timing across the same thread,
which lives in `signal_events` (not `memory_records`). The current
handler in `navis/api/memory_router.py` only touches memory records
to keep the endpoint cheap; the timing join was deliberately deferred.

### Files to change
- `navis/signals/repository.py` — add a bulk query that returns, for
  a set of `person_email`s, the average time-delta between an inbound
  message from that person and the next outbound message in the same
  thread
- `navis/api/memory_router.py:_derive_person()` — populate
  `replyLatencyHours` from that map

### Implementation notes
- Use `signal_events.payload->>'from'` and `->>'thread_id'` for the
  join; both are indexed today via JSONB path indexes (verify before
  querying).
- Cap the look-back window (30 days) — older thread stats are noise
  for a dashboard hint.
- Compute in Python (bounded set of people, bounded threads) rather
  than pushing a complex windowed aggregation to Postgres.

### Verification
- For a person with clear inbound → outbound chains in the test
  fixture, `replyLatencyHours` should be a finite float within an
  expected band.
- No performance regression on the endpoint (add a `timed()` block
  around the new query and log latency).

### Related references
- `navis/api/memory_router.py:283-286` — current TODO / deferred field
- `navis/signals/repository.py` — where the join query should live

---

## 7. Team domain detection (roleHint="team")

**Status:** PENDING
**Priority:** Medium
**Discovered:** 2026-07-26

### Problem
Colleagues who share the org's own email domain (e.g. two people at
`abmrc.com`) currently show as `roleHint: "external"` — same badge as
a customer or investor. The dashboard has no way to visually separate
"someone on my team" from "someone outside".

### Root cause
`memory_router._role_hint()` uses only a free-domain check (`gmail.com`
et al.) → returns "personal" for those, "external" for everything
else. The org record does not carry a canonical primary domain
(`Organization` / `OrganizationProfile` models don't have a `domain`
column — verified 2026-07-26).

### Files to change
- `navis/organization/models.py` — add `primary_domain: str | None`
  (or a `verified_domains: list[str]`) on `Organization` or a related
  table
- `navis/organization/service.py` — populate on org creation from
  the founder's email domain (unless free-domain) and expose a
  self-service endpoint to add more
- `navis/api/memory_router.py:_role_hint()` — take a `set[str]` of
  team domains as an argument; return `"team"` when the email domain
  matches

### Implementation steps
1. Add the column + migration.
2. Backfill from `users.email` for existing orgs where possible.
3. Load once per request in `list_people` (single query) and pass to
   `_derive_person`.
4. Update tests to cover the `"team"` classification.

### Verification
- Create org with primary_domain=`acme.com`.
- Fixture with `ceo@acme.com` and `investor@vc.com` → GET
  `/memory/people` returns `"team"` and `"external"` respectively.

### Related references
- `navis/organization/models.py` — where the schema addition lands
- `navis/api/memory_router.py:_role_hint()` — the classification site

---

## 8. LLM hybrid in reasoning engines

**Status:** PENDING
**Priority:** Roadmap
**Discovered:** 2026-07-26

### Problem
All reasoning engines in `navis/reasoning/v2/engines/` are pure
rule-based threshold functions:
`age >= 5 days → COMMITMENT_SLIPPING`, `age >= 7 days →
RELATIONSHIP_COOLING`, etc. Rules are fast, replayable, and cheap,
but they mis-classify context-dependent cases: a client on vacation
still flags "slipping"; a client explicitly postponed still flags
"stalled". This is what makes the dashboard occasionally feel dumb.

### Design goal (NOT to be done for VC demo)
Add an OPTIONAL LLM tier BEHIND the rules — rules classify first
(cheap, always-on), and only *ambiguous* signals get sent to the
LLM for a second opinion. Cost stays ~10% of naive per-signal LLM
inference.

### Files to change (sketch)
- `navis/reasoning/v2/contract.py` — extend `SignalDraft` with an
  `ambiguity: float` field (rules populate this today implicitly:
  a signal exactly at the threshold is more ambiguous than one 3x
  past it)
- `navis/reasoning/v2/engines/base.py` — introduce a `Engine`
  protocol so the LLM engine can slot in beside rule engines
- New `navis/reasoning/v2/engines/llm/` — LLM engine that takes an
  ambiguous signal + the relevant state slice and returns a
  confidence-adjusted verdict (or drops it entirely)
- `navis/ml/generation/prompts.py` — add a `reasoning_review`
  prompt template (constrained: input is facts + a proposed verdict,
  output is a JSON `{verdict, confidence, why}`)
- `navis/reasoning/v2/service.py` — orchestrator: run rules → select
  top-N ambiguous → LLM review → merge

### Preconditions to defer this until met
- 100+ labelled examples of "rule said X, correct answer was Y" to
  evaluate the LLM tier before turning it on
- Ability to A/B briefs (with/without LLM tier) via a feature flag
- Cost budget signed off — Opus is not cheap even at 10%

### Verification
- Determinism preserved: same state → same verdict (LLM tier uses
  temperature 0 + a fixed prompt/model version)
- Rules still run first; LLM tier is a no-op when disabled
- Eval harness reports precision/recall gain vs. rules-only

### Related references
- `navis/reasoning/v2/engines/risk.py` — current rule thresholds
- `navis/ml/providers/claude.py` — Claude call site (already live in
  `/chat`)
- `navis/reasoning/v2/contract.py` — signal contract to extend

---

## 9. RAG expansion beyond `/chat` (briefs + context)

**Status:** PENDING
**Priority:** Roadmap
**Discovered:** 2026-07-26

### Problem
The full RAG stack (embeddings, vector search, grounded context,
LLM narration) is wired up only in `/api/v1/chat`
(`navis/api/chat_router.py`). The morning brief
(`/api/v1/brief/today`) still renders a rule-based sentence like
"3 commitments slipping, 2 relationships cooling" — the LLM
narrative version we discussed for demo polish is not implemented.

### Root cause
`navis/orchestration/brief.py:build_brief()` composes
`ReasoningSignal[]` + `Decision[]` into a `DailyBrief` with no
narration step. The retrieval + prompting infrastructure exists
(`navis/ml/retrieval/retriever.py`, `navis/ml/rag/context_builder.py`,
`navis/ml/generation/narratives.py`) but is not called from the brief
path.

### Files to change (sketch)
- `navis/orchestration/brief.py` — after signals + decisions are
  composed, call `ContextBuilder(db).build(...)` + `explanation(...)`
  to produce a narrated summary
- `navis/decisions/models.py:DailyBrief` — add a `narrative: str | None`
  column (nullable so old briefs remain valid on replay)
- `navis/api/brief_router.py:_serialize` — include narrative in the
  response
- Feature flag on the LLM call so tests / cheap paths can skip it

### Preconditions
- Cost budget for daily brief LLM narration
- Grounding guard verified end-to-end for brief flow (must not
  fabricate; must cite the facts from the retriever)

### Verification
- With flag OFF: brief renders identically to today's output
  (rule-based sentence)
- With flag ON: brief carries a narrative that names real people /
  threads with citation markers `[n]`
- Determinism check: same state + same reasoning version → identical
  narrative (temperature 0)

### Related references
- `navis/api/chat_router.py:129-157` — pattern to mirror
- `navis/ml/generation/narratives.py:38-63` — explanation call site
- `navis/orchestration/brief.py` — where the narration slots in

---

## 10. Reasoning engine protocol refactor

**Status:** PENDING
**Priority:** Roadmap
**Discovered:** 2026-07-26

### Problem
Each engine in `navis/reasoning/v2/engines/` is a class with an
`analyze(state, ctx)` method, discovered via the `ENGINES` list.
Adding an LLM-backed variant (see item #8) or swapping a specific
engine's implementation requires touching the registry directly.
There is no explicit `Protocol` / ABC that new engines must satisfy,
so mismatches surface at runtime (via the "one bad engine never
blanks the pass" catch-all) rather than at import.

### Root cause
`navis/reasoning/v2/base.py` defines `ReasoningContext` only. No
formal engine contract exists — the `for engine in ENGINES` loop in
`navis/reasoning/v2/service.py:63` relies on duck typing.

### Files to change
- `navis/reasoning/v2/base.py` — add a `ReasoningEngine` `Protocol`
  or ABC with `name: str`, `version: str`, `analyze(state, ctx) ->
  Iterable[SignalDraft]`
- Every engine file in `engines/` — declare inheritance / satisfy
  the protocol (mypy will catch any drift)
- `service.py` — narrow the loop variable to the protocol type

### Why bother
- Enables item #8 (LLM engines) to slot in without special-casing.
- Enables per-engine A/B testing (swap in a new rule set behind a
  flag).
- Catches contract violations at import time, not on the daily brief
  pass in production.

### Verification
- `mypy` clean on the reasoning package
- Existing engines unchanged behaviourally (replay determinism check
  against a pinned state fixture)

### Related references
- `navis/reasoning/v2/engines/__init__.py` — the ENGINES list
- `navis/reasoning/v2/service.py:63` — the untyped loop today

---

## 11. User dismissal tracking → eval dataset

**Status:** PENDING
**Priority:** Roadmap
**Discovered:** 2026-07-26

### Problem
Every time the founder dismisses a decision from today's brief
(`POST /brief/decisions/{ref}/dismiss` — already implemented), we
persist a `DecisionDismissal` row and emit a `DECISION_DISMISSED`
event. But we don't:
- Correlate dismissals back to the specific reasoning signals that
  produced the decision (which engine + which thresholds fired?)
- Aggregate the correlation into a per-engine precision/recall
  metric that ops can watch

The dismissal event stream is the highest-signal labelled data we
have for tuning rule thresholds or (eventually) training an LLM
reasoning tier (item #8). Currently it just sits there.

### Files to change
- `navis/decisions/service.py` — when composing a brief, persist
  the *lineage* from each Decision to its source ReasoningSignal(s)
  (a JSONB column, or a `decision_signals` join table)
- `navis/observation/` (or a new sub-package) — a periodic
  aggregation job that computes `dismiss_rate_by_engine`,
  `dismiss_rate_by_signal_type`, exposed via
  `/admin/metrics/reasoning_quality`
- `navis/api/admin_router.py` — surface a read endpoint for the
  aggregation

### Verification
- Dismiss a decision from a specific engine (e.g. CommitmentEngine)
  → the aggregation reflects it within one cycle
- Threshold tuning experiments can be evaluated against the
  aggregation without touching production code

### Related references
- `navis/api/brief_router.py:113-156` — dismissal endpoint
- `navis/reasoning/v2/service.py:_stamp` — where signal provenance
  is captured
- `navis/decisions/models.py:DecisionDismissal` — dismissal table

---

## Template — copy for new entries

```markdown
## N. <Title>

**Status:** PENDING
**Priority:** <Low | Medium | High>
**Discovered:** YYYY-MM-DD

### Problem
<User-visible symptom or the concrete thing that is broken / missing.>

### Root cause
<Why it happens. Cite file paths + line numbers so future-you can
jump straight to the code.>

### Files to change
- `path/to/file.py` — what changes here

### Implementation steps
1. Step one.
2. Step two.

### Verification
- How you'll know the fix works end-to-end.

### Related references
- Links to code, docs, or external resources.
```
