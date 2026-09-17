# Microsoft 365 — Backend Implementation Plan

Wiring the Outlook / Outlook Calendar / OneDrive / Teams connectors and Entra
sign-in into the backend, against the contracts the frontend has already
shipped.

| | |
|---|---|
| **Frontend** | `D:\NavisAi\Navis-AI` — `feature/microsoft-integration` |
| **Backend** | `D:\NavisAi\Backend` — `feat/microsoft-Integration` (@ `93ceb12`) |
| **Commits reviewed** | `36d5584` (connections), `080cafd` (sign-in) |

The frontend shipped the whole Microsoft surface behind two flags and left the
seams documented. This is what the backend has to build to fill them — ordered
so that nothing Google does today changes behaviour at any step.

---

## 1. What the two commits already fixed in place

These are not suggestions. They are contracts the frontend already compiles
against, and every one is a thing the backend must match exactly or a card
renders wrong.

### 1.1 Routes the frontend expects

| Endpoint | Called from | Response shape |
|---|---|---|
| `GET /integrations/microsoft/connect` | Connect button on any Microsoft card | `{ authorization_url }` |
| `GET /integrations/microsoft/callback` | Entra redirect target | 303 to frontend (§1.2) |
| `DELETE /integrations/microsoft/disconnect` | `DisconnectDialog` | `{ disconnected }` |
| `POST /integrations/microsoft/sync` | Manual "Sync now" | mirror of the Google one |
| `POST /auth/microsoft` | Sign-in page — **separate track** | `TokenPair` |

Today `src/services/api.ts` throws a specific, loud error for the whole
Microsoft family rather than falling through to the synthesized-Connection
path — deliberately, so a Connect click can never paint a green "Connected"
pill over a mailbox nobody is reading.

### 1.2 The callback's return URL

`_frontend_return_url` in `navis/signals/router.py` currently emits
`?connected=<provider>` on success and `?error=<code>` on failure. The account
page reads a **third** param the Google flow never sends:

- Success: `?connected=outlook`
- Failure: `?error=<code>&provider=outlook`

Without `provider=` on the error path, `decodeOAuthError` falls back to its
default and an Outlook org is told to "Reconnect Google" — which is not a thing
they have.

The error codes are a closed set (`src/lib/oauthErrors.ts`). Anything outside
it renders a generic banner:

```
missing_scopes   no_email_claim   exchange_failed   invalid_state   oauth_failed
```

### 1.3 Provider strings on `GET /integrations`

The alias map in `services/api.ts` matches Integration rows to cards by exact
provider value. **One row per resource** — no aliasing `outlook_calendar` onto
the `outlook` row, or Calendar reports "synced 2 min ago" off the mailbox's
timestamp:

```
outlook
outlook_calendar
onedrive
teams
```

If the callback only writes one row at first, the other cards honestly read
Not connected until it writes theirs. That is the intended behaviour.

### 1.4 Entitlements

The sidebar filters connector cards on capability.
`GET /organization/entitlements` must emit `outlook`, `outlook_calendar`,
`onedrive` and `teams` as their own caps — deliberately **not** a shared `mail`
cap, because an org can be licensed for Gmail and not Outlook. With the flag on
and the caps missing, the cards vanish and nobody can tell why.

---

## 2. State of play in the backend

`D:\NavisAi\Backend` is already on `feat/microsoft-Integration` with a clean
tree and nothing Microsoft in it.

| Layer | Where | Today | Needs |
|---|---|---|---|
| Provider enum | `identity/models.py:33` | gmail, google_calendar, google_meet, slack, notion, drive | **Migration** — native PG enum |
| Capability enum | `organization/contract.py:26` | gmail, calendar, drive, slack | No migration — `String(32)` |
| Signal source | `signals/base.py:38` | gmail, calendar, drive, docs, meet, slack, notion, crm | No migration — `String(32)` |
| OAuth routes | `signals/router.py:238` | Google + Slack only | New module |
| Token lifecycle | `signals/google/oauth.py` | `TokenManager`, Google-shaped | Mirror, do **not** reuse (§4 Phase 3) |
| HTTP transport | `signals/google/transport.py` | `GoogleTransport` | Graph equivalent |
| Sync registry | `orchestration/sync.py:52` | 3 Google entries | Add rows |
| Connectors | `signals/{gmail,calendar,drive}/` | 3, all `WorkspaceConnector` | 3 new |
| Sign-in | `identity/router.py:97` | `POST /auth/google` | Parallel route |

### The one schema change

`integrations.provider` is a **native PostgreSQL enum**
(`SAEnum(IntegrationProvider, values_callable=...)`), not a string column.
Adding four members needs `ALTER TYPE ... ADD VALUE`, which cannot run inside a
transaction and cannot be reversed. There is no precedent for one in
`migrations/versions/` — this will be the first.

Everything else in this plan is additive Python.

---

## 3. Five decisions before any code

Each changes what gets built. Recommendation on each.

### 3.1 Does Teams ship in this round?

Reading Teams *channel* messages through Graph (`ChannelMessage.Read.All`) sits
behind Microsoft's Protected APIs programme — an application, a review, and a
metered billing model. That is weeks of calendar time you do not control. A
user's own chats (`Chat.Read`) are consentable today.

> **Recommend:** ship mail + calendar + files. Leave the Teams card behind its
> own sub-flag and start the Protected APIs application in parallel, so the wait
> runs alongside the build rather than after it.

### 3.2 Read-only, or send from day one?

Google already carries `gmail.send` and `calendar.events`, so execution works
there. The Graph equivalents are `Mail.Send` and `Calendars.ReadWrite`. Without
them an Outlook org gets a decision it cannot execute, and the Ask Navis loop
dead-ends at the last step.

> **Recommend:** request them in the same consent. Asking twice costs a second
> admin conversation, and the lane resolution in `actionPolicies` already
> assumes the mail lane can send.

### 3.3 Polling or push?

Graph delta queries are a straight poll and work behind any firewall. Change
notifications are faster but need a public HTTPS endpoint, a validation
handshake, and a renewal job — mail and calendar subscriptions expire in under
three days.

> **Recommend:** delta polling first, on the beat schedule that already drives
> Google. Push is Phase 7, after ingestion is proven.

### 3.4 Which plans get the Microsoft caps?

`PLAN_CAPABILITIES` seeds entitlements per plan. `ENTERPRISE` is
`frozenset(Capability)` so it picks up new caps automatically; FREE, PRO and
TEAM will deny by default. Existing orgs have no row for the new caps, so they
fall through to the plan default — which fails closed, correctly.

> **Recommend:** mirror Google exactly. `outlook` + `outlook_calendar` wherever
> `gmail` + `calendar` appear, `onedrive` wherever `drive` does. Same product,
> different vendor — pricing it differently is a decision nobody has made.

### 3.5 One Entra app or two?

The frontend split the flags precisely so sign-in and the connector can ship
apart and never share a consent screen. The backend config already has the same
shape for Google — `google_redirect_uri` and `google_gmail_redirect_uri` are
separate settings.

> **Recommend:** one multi-tenant app registration, two redirect URIs, two scope
> sets. Never `Mail.Read` at the login screen.

---

## 4. The build, in dependency order

Each phase compiles and deploys on its own, and each leaves Google untouched.
Phases 1-6 are the connector; Phase 8 is sign-in and can be built by someone
else at the same time.

### Phase 0 — Turn the flags back off · *do this first*

`.env.local` has both `NEXT_PUBLIC_ENABLE_MICROSOFT` and
`NEXT_PUBLIC_ENABLE_MICROSOFT_AUTH` set to `true` right now, with no backend
behind either. Every Microsoft card is live and its Connect button throws;
"Continue with Microsoft" throws. `.env.example` also ships `=true`,
contradicting its own comment two lines above.

Both go to `false` until the routes answer, then flip per environment — staging
leading production — never in code.

```
Frontend: .env.local, .env.example
```

### Phase 1 — Entra registration and config · *no code paths change*

Register a multi-tenant app. Two redirect URIs (sign-in, connector). Request the
`xms_edov` optional claim on the ID token — that claim is what makes the "link
only on a verified email" rule in `src/lib/auth.ts` enforceable rather than
aspirational.

New settings alongside the Google OAuth block. Every one defaults to `None`, so
an environment without them behaves exactly as it does today.

```
navis/platform/config/settings.py  (after line 182)
  microsoft_client_id / microsoft_client_secret
  microsoft_tenant = "common"
  microsoft_redirect_uri
  microsoft_signin_redirect_uri
```

### Phase 2 — Enums and the one migration · *irreversible*

Four members onto each of three enums. Only the first needs SQL.

- `IntegrationProvider` += `OUTLOOK`, `OUTLOOK_CALENDAR`, `ONEDRIVE`, `TEAMS`
  → migration `0046`, autocommit block, `ADD VALUE IF NOT EXISTS`, downgrade a
  documented no-op.
- `Capability` += the same four, plus their entries in `PLAN_CAPABILITIES`
  per §3.4.
- `SignalSource` += the same four. Already `String(32)` on `signal_events`.

`_SEED_SOURCE_TO_PROVIDER` gains its four rows so the dev seed endpoint can
inject Outlook signals before the connector exists — which is what makes
Phase 5 testable without a live mailbox.

```
navis/identity/models.py
navis/organization/contract.py
navis/signals/base.py
navis/signals/router.py            (seed map)
migrations/versions/0046_microsoft_providers.py
```

### Phase 3 — Graph transport and token manager · *one real divergence*

A `navis/signals/microsoft/` package mirroring `signals/google/` — `oauth.py`
(authorization URL, code exchange, refresh, scope constants) and `transport.py`
(retry, `429` with `Retry-After`, auth-failure callback).

> **Do not copy the Google refresh path.** Microsoft **rotates** refresh tokens:
> every refresh returns a new one and invalidates the old. The Google callback's
> `coalesce(enc_refresh, Integration.refresh_token)` is correct for Google, which
> omits the token on re-consent — carried over to Microsoft it silently keeps a
> dead token and every integration dies at the 90-day inactivity mark with no
> error until then.

```
navis/signals/microsoft/__init__.py
navis/signals/microsoft/oauth.py
navis/signals/microsoft/transport.py
```

### Phase 4 — Connect, callback, disconnect, sync

A sibling router rather than four more handlers in `signals/router.py`, which is
already past a thousand lines. The Redis OAuth state helpers —
`_create_oauth_state`, `_consume_oauth_state`, single-use, ten-minute TTL,
attribution resolved server-side only — get lifted into a shared module and used
verbatim. That CSRF property is not worth re-deriving.

The callback follows the Google sequence exactly:

1. Validate state first; errors bounce to the UI, never a JSON 400.
2. Exchange the code (bound to the validated state).
3. Reject incomplete grants — no row for a scope that was not given.
4. Identify the account from the ID token. **Not** `preferred_username`, which
   is a display value — `email`, and only when `xms_edov` asserts it.
5. `ON CONFLICT` upsert one row per granted resource against
   `uq_org_provider_account`. Race-safe, reconnect-safe.

`_frontend_return_url` gains a `provider=` param on the error path (§1.2).
Google keeps passing nothing and keeps defaulting to "Google" — same behaviour,
new capability.

```
navis/signals/microsoft_router.py       (new)
navis/signals/oauth_state.py            (extracted)
navis/signals/router.py                 (return-url param only)
```

### Phase 5 — Three connectors

Each implements `WorkspaceConnector` — `full_sync`, `incremental_sync`,
`create_watch` — and returns `SignalEvent`s. Pure capture, no DB writes;
`SignalSyncService` already owns persistence, dedupe and the dead-letter queue,
and does not care which vendor produced the event.

| Connector | Graph endpoint |
|---|---|
| `outlook` | `/me/mailFolders/inbox/messages/delta` |
| `outlook_calendar` | `/me/calendarView/delta` |
| `onedrive` | `/me/drive/root/delta` |

The delta link goes in `sync_state`, the same slot Gmail's `historyId` uses.
Content fields get the same `field_cipher` treatment Gmail applies, and for the
same reason — the append-only log and every backup of it hold ciphertext.
Routing fields (from/to/cc, thread id) stay plaintext or the people graph goes
blind.

```
navis/signals/outlook/{client,connector,parser}.py
navis/signals/outlook_calendar/...
navis/signals/onedrive/...
```

### Phase 6 — Register with the sync engine · *nine lines*

`_build_client` currently hardcodes `GoogleTransport` + `TokenManager`; it takes
a per-provider factory instead, so a Microsoft row builds a Graph transport.
Then rows in `_REGISTRY` and `_CAPABILITY`.

`_CAPABILITY` is the fail-closed gate: a provider absent from it **cannot be
synced**. That is why an unmapped Microsoft row is safe to have in the database
before this phase — it simply never runs.

```
navis/orchestration/sync.py
  _build_client -> per-provider factory
  _REGISTRY   += 3
  _CAPABILITY += 4
```

### Phase 7 — Execution and push · *after ingestion is proven*

An `outlook` sender beside `execution/providers/gmail.py`, registered in
`_senders.py`, so an approved reply on an Outlook thread goes out through
Outlook. The frontend already routes this correctly: `resolveLaneConnection`
takes the decision's evidence provider and picks the matching connector.

Then Graph change notifications, if §3.3 went that way — a subscription endpoint
with the validation handshake and a beat task that renews before the ~3-day
expiry.

### Phase 8 — Sign-in · *independent track, parallelisable*

`POST /auth/microsoft` mirroring `/auth/google`: same `TokenPair`, same routing
flags, identity scopes only (`openid profile email User.Read`). A
`microsoft_sub` column beside `google_sub` on `users`, plus `microsoft_tid` so a
personal Microsoft account is distinguishable from a work one.

> **The security decision in this phase.** Link a Microsoft sign-in to an
> existing account on the email claim **only** when the IdP asserts the domain is
> verified. Entra carries an unverified `email` for personal Microsoft accounts,
> and anyone can put any address on one — linking on it is an account-takeover
> primitive. Absent that assertion, create a separate account and let the user
> link deliberately from inside the product.

```
navis/identity/microsoft_auth.py
navis/identity/router.py
navis/identity/models.py                (User)
migrations/versions/0047_user_microsoft_sub.py
```

### Phase 9 — Close the frontend seams · *three files*

The last step, and deliberately tiny — the previous commits did this work
already. Two `throw`s become real calls, and the flags flip in the environment
that has the routes.

```
src/services/api.ts    microsoftFamily throw -> startMicrosoftConnect / disconnectMicrosoft
src/lib/auth.ts        loginWithMicrosoftCode
.env                   per environment
```

Nothing else on the frontend.

---

## 5. What must not break

Four things in this plan can touch a working Google org. Each has a specific
containment.

| Risk | Why it could bite | Containment |
|---|---|---|
| The enum migration | `ALTER TYPE` cannot be rolled back, and a new value cannot be used in the same transaction that adds it. | Its own migration, autocommit, nothing else in it. Deploy a release ahead of any code that writes the value. |
| Extracting the OAuth state helpers | Google's connect flow depends on them. A refactor here breaks a live path. | Move, do not rewrite. Google's existing OAuth tests must pass untouched before the Microsoft router imports them. |
| `_build_client` becoming generic | Every Google sync goes through it. A wrong default silently stops ingestion. | Google keeps the identical branch; Microsoft added as a new key. Unknown providers still return `(None, None, None)`. |
| Widening `PLAN_CAPABILITIES` | `seed_for_plan` rewrites every capability row on a plan change — an org could gain or lose access as a side effect. | Additive only. Verify against an org on each plan that no existing cap flips. |

Everything else is new files or new enum members. The Google connect path, the
Gmail watch, the Slack webhook and the sync beat are untouched by phases 1, 3,
5, 7 and 8 entirely.

### Proving it

- Run the existing backend suite green before and after Phase 2 — that migration
  is the only step that can move data.
- **Google-only org, flags off:** identical Connections page, identical
  entitlements payload, identical `/integrations` response.
- **Google-only org, flags on:** four extra cards reading Not connected. No
  green pill anywhere.
- **Both suites connected:** `primaryWorkspaceProvider` resolves, and a reply to
  an Outlook thread routes to Outlook rather than Gmail.
- The frontend's `workspaceProvider.test.ts` and `signalSource.test.ts` already
  cover the lane logic and **should need no edits at any point in this plan**.
  If one needs changing, the backend contract has drifted.

---

*Written against commits `36d5584` and `080cafd` on
`feature/microsoft-integration`, and the working tree of `D:\NavisAi\Backend`
at `93ceb12`. Line numbers are from those revisions.*
