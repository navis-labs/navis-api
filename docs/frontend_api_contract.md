# Navis API Contract — Layer 1 (Identity + Organization)

Stable HTTP surface for frontend integration. Everything is org-scoped after login.

**Base URL:**
- Local dev: `http://127.0.0.1:8000`
- Staging/Prod: provided per environment.

**API version prefix:** `/api/v1` (all endpoints in this doc are prefixed with this).

**Auth header:** `Authorization: Bearer <access_token>` on every protected route.

**Content-Type:** `application/json` on every POST/PUT body.

**Error shape (FastAPI default):**
```json
{ "detail": "human-readable error" }
```
Validation errors (422) return a list of issues:
```json
{ "detail": [{ "type": "...", "loc": ["body", "email"], "msg": "...", "input": "..." }] }
```

---

## Quick Start — Frontend integration flow

```
┌─────────────┐   1. Signup OR Google SSO   ┌──────────────────┐
│   Landing   │ ─────────────────────────▶  │  Backend issues   │
│    page     │                              │  access+refresh  │
└─────────────┘                              │  + onboarding    │
                                             │    flags         │
                                             └────────┬─────────┘
                                                      │
                  ┌───────────────────────────────────┘
                  │  is_new_user=true OR
                  │  onboarding_completed=false
                  ▼
            ┌──────────────┐   PUT /organization/profile   ┌────────────┐
            │  /onboarding │ ───────────────────────────▶  │  Dashboard │
            │     page     │   (sets PROFILE_COMPLETE)     │            │
            └──────────────┘                                └────────────┘

  Token lifecycle: access expires (~15 min) → POST /auth/refresh → rotate both.
  On 401 from any protected route → call refresh once → retry the original request.
  If refresh ALSO 401 → log the user out, redirect to login.
```

---

# 1. Authentication

All routes here are **public** (no auth header) unless noted.

## 1.1 `POST /api/v1/auth/signup` — Email + password signup

Creates a new user, a new organization (where the user is the founder), and issues tokens.

**Request body:**
```json
{
  "email": "user@example.com",
  "password": "Strong-password-123",
  "full_name": "Jane Founder",
  "organization_name": "Acme Co"
}
```

**Field rules:**
| Field | Type | Constraint |
|---|---|---|
| `email` | string | Valid email; lowercased server-side; reserved TLDs (`.local`, `.localhost`, `.test`, `.example`, `.invalid`) are rejected |
| `password` | string | 8–128 chars; not in weak-password blocklist (`password`, `12345678`, …); not all-same-character |
| `full_name` | string \| null | ≤ 255 chars |
| `organization_name` | string | 2–255 chars |

**Success — 201:**
```json
{
  "access_token": "eyJhbGciOi...",
  "refresh_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_in": 900,
  "is_new_user": false,
  "onboarding_completed": false
}
```

**Errors:**
- `400` — email already registered.
- `422` — schema validation (bad email, weak password, missing field).

---

## 1.2 `POST /api/v1/auth/login` — Email + password login

**Request body:**
```json
{ "email": "user@example.com", "password": "Strong-password-123" }
```

**Success — 200:** same `TokenPair` shape as signup.

**Errors:**
- `401 invalid email or password` — generic for ALL failures (wrong password, unknown email, disabled account, **locked from brute-force throttle**). Do not try to disambiguate client-side.

**Throttle:** 5 failed attempts per email OR per IP within 15 minutes locks the identifier. Lock clears automatically.

---

## 1.3 `POST /api/v1/auth/google` — Google SSO login

Sign in via Google OAuth2. Accepts EITHER `code` (preferred — authorization-code flow) OR `id_token` (one-tap fallback). Exactly one must be provided.

**Request body (preferred):**
```json
{ "code": "<authorization code from Google's initCodeClient>" }
```

**Or (legacy fallback):**
```json
{ "id_token": "<id_token from accounts.id one-tap>" }
```

**Success — 200:**
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 900,
  "is_new_user": true,
  "onboarding_completed": false
}
```

**Frontend routing logic** (use the flags from the response):
- `is_new_user=true` OR `onboarding_completed=false` → `/onboarding`
- `onboarding_completed=true` → `/dashboard`

**Errors:**
- `401` — invalid Google credential, expired token, Google unreachable, account disabled.
- `422` — neither `code` nor `id_token` provided (or both).

**Throttle:** Same IP-based gate as `/login`. 5 failures lock the IP for 15 minutes.

---

## 1.4 `POST /api/v1/auth/refresh` — Rotate tokens

Use this when an access token expires (any 401 from a protected route). Issues a NEW access + refresh pair; the **old refresh becomes invalid immediately**.

**Request body:**
```json
{ "refresh_token": "<current refresh token>" }
```

**Success — 200:** new `TokenPair`.

**Errors:**
- `401 refresh token revoked or unknown` — token expired, replayed, or logged-out session.
- `401 user not found or inactive` — account was deactivated.
- `401 token revoked` — explicit bulk-revoke (admin deactivation, password change, etc).

**Security:** If a refresh token is replayed (already-rotated token reused), the backend **revokes the entire session family** — assume token theft. Re-login required.

---

## 1.5 `POST /api/v1/auth/logout` — End session

**Headers:** `Authorization: Bearer <access_token>`

**Body:** none.

**Success — 200:**
```json
{ "status": "logged_out" }
```

Both the access JTI AND the bound refresh session are revoked. Subsequent calls with either token return 401.

---

## 1.6 `GET /api/v1/auth/me` — Current user, org, role, and effective permissions

**Headers:** `Authorization: Bearer <access_token>`

**Success — 200:**
```json
{
  "user": {
    "id": "b8b9c5ee-ff79-463a-af87-48d30a34b5cc",
    "email": "user@example.com",
    "full_name": "Jane Founder",
    "is_active": true,
    "is_verified": false,
    "created_at": "2026-06-30T09:00:00+00:00"
  },
  "organization": {
    "id": "82ddae08-d242-4d54-96aa-551e97f7bb46",
    "name": "Acme Co",
    "slug": "acme-co"
  },
  "role": "founder",
  "permissions": ["execute_actions", "manage_members", "view_decisions", "…"],
  "organizations": [
    { "id": "82ddae08-…", "name": "Acme Co", "slug": "acme-co", "is_current": true }
  ],
  "onboarding_completed": false
}
```

**Use this** to populate the user menu, role badge, gate routes that need to know
if onboarding is finished, and — via `permissions` — decide what to render.

### `permissions` — read it, do not recompute it

`permissions` is the caller's **effective** permission set for the current
organization: the role's defaults **plus** any per-org grants this organization
has configured. It is the exact set the backend's own authorization gate checks.

**Do not derive permissions from `role` on the client.** A client-side copy of the
§4.1 table is wrong in two ways it cannot detect:

- per-org grants live only in the database, so a role's real powers in *this*
  org may exceed the defaults;
- the table drifts the moment the backend changes a default.

Read `permissions` and check membership of the array. It is sorted, so it is safe
to diff or cache. Discovering permissions by provoking `403`s is likewise
unnecessary — and was never reliable, since a `403` can also mean a capability
(plan) gate, not a permission gate.

This is still **presentation only**. Every protected route re-checks server-side;
holding a string here has never been what grants access (§6.1).

### `role` can be `null`

`role` is `null` when the membership has no usable role — a row that predates the
`NOT NULL` constraint, or a role value the running build does not recognise. When
it is `null`, `permissions` is `[]`. Treat that as "no privileges" and fail
closed; the endpoint reports it rather than erroring.

### `organizations` — reporting only, no switching

`organizations` lists every organization the user holds a seat in, with exactly
one entry flagged `is_current: true`. That entry is always the same organization
as the top-level `organization` object: the tenant this access token
authenticated into.

Each entry carries **identity only** — `id`, `name`, `slug`, `is_current`. There
is deliberately no `role` and no `permissions` per entry, because authorization
is published for the active organization and nothing else.

**There is no organization-switch endpoint.** Multi-org switching — an active-org
token mechanism, an org picker, per-org onboarding — is **Phase 7** work. Today,
changing the active organization means authenticating into it. Use this list to
*show* that the account has more than one organization; do not build a switcher
against it.

**Errors:** `401` with no credentials, an invalid/expired/revoked token, or an API
key (this endpoint describes a human session; machine principals are rejected).
`403` when the token names an organization the user has no membership in.

---

# 2. Organization

All routes here require `Authorization: Bearer <access_token>`. The org is implicit from the token.

## 2.1 `GET /api/v1/organization/profile` — Read org profile

**Success — 200:**
```json
{
  "profile": {
    "company_name": "Acme Co",
    "website": "https://acme.example",
    "industry": "saas",
    "employee_count": 25,
    "hq_country": "IN",
    "team_size": "11-50",
    "org_stage": "seed",
    "primary_goal": "Centralize organizational memory"
  }
}
```

If no profile has been saved yet:
```json
{ "profile": null }
```

---

## 2.2 `PUT /api/v1/organization/profile` — Update org profile (onboarding form)

**Required permission:** `manage_members` (founder + admin + ceo + coo have it by default).

**Request body** (all fields optional; send only what changed):
```json
{
  "company_name": "Acme Co",
  "website": "https://acme.example",
  "industry": "saas",
  "employee_count": 25,
  "hq_country": "IN",
  "team_size": "11-50",
  "org_stage": "seed",
  "primary_goal": "Centralize organizational memory"
}
```

**Field rules:**
| Field | Constraint |
|---|---|
| `company_name` | ≤ 255 chars |
| `website` | ≤ 255 chars |
| `industry` | ≤ 128 chars |
| `employee_count` | integer ≥ 0 |
| `hq_country` | ISO 3166-1 alpha-2 (exactly 2 chars, e.g. `"IN"`, `"US"`) |
| `team_size` | ≤ 16 chars — suggested values: `"1"`, `"2-10"`, `"11-50"`, `"51-200"`, `"200+"` |
| `org_stage` | ≤ 32 chars — suggested values: `"idea"`, `"pre-seed"`, `"seed"`, `"series-a"`, `"series-b-plus"` |
| `primary_goal` | ≤ 500 chars (free-text) |

**Success — 200:**
```json
{ "status": "updated" }
```

**Side-effect:** when `company_name + industry + team_size + org_stage` are ALL set, the org's onboarding stage advances to `profile_complete` and `/me` will return `onboarding_completed: true` on next call.

**Errors:**
- `403` — caller lacks `manage_members`.
- `422` — schema validation.

---

## 2.3 `GET /api/v1/organization/onboarding/status` — Lightweight onboarding gate

The route AuthGuard should hit before showing the dashboard. Cheaper than `/auth/me`.

**Success — 200:**
```json
{ "stage": "profile_complete", "onboarding_completed": true }
```

**Stage values** (canonical forward order):
1. `created`
2. `profile_complete`
3. `gmail_connected`
4. `calendar_connected`
5. `first_signal`
6. `memory_ready`
7. `state_ready`
8. `first_brief`
9. `activated`

`onboarding_completed` is `true` once stage is at or beyond `profile_complete`.

---

## 2.4 `GET /api/v1/organization/onboarding` — Detailed onboarding state

**Success — 200:**
```json
{ "stage": "profile_complete", "signals_count": 0, "memory_count": 0 }
```

---

## 2.5 `GET /api/v1/organization/entitlements` — Which capabilities this org can use

Use to decide **which features/tiles to render** in the UI.

**Success — 200:**
```json
{
  "entitlements": {
    "gmail": true,
    "calendar": true,
    "drive": false,
    "slack": false,
    "memory": true,
    "state": true,
    "reasoning": false,
    "decision": false,
    "execution": false,
    "observation": false,
    "learning": false,
    "intelligence": false,
    "adaptive": false,
    "ml": false
  }
}
```

**Capabilities** (the keys above):
- **Signal sources:** `gmail`, `calendar`, `drive`, `slack`
- **Pipeline:** `memory`, `state`, `reasoning`, `decision`, `execution`, `observation`, `learning`, `intelligence`, `adaptive`, `ml`

A capability is enabled when:
1. Org lifecycle is `trial` or `active` (NOT `suspended`/`churned`), AND
2. No kill-switch flag is set (`kill:<capability>`), AND
3. Either an explicit grant exists OR the org's plan includes it.

**Default plan capabilities** (for reference):
| Plan | Granted |
|---|---|
| `free` | gmail, calendar, memory, state |
| `pro` | + drive, reasoning, decision |
| `team` | + slack, execution, intelligence |
| `enterprise` | all of them |

---

## 2.6 `GET /api/v1/organization/metrics` — Activation metrics

**Success — 200:**
```json
{
  "status": "trial",
  "plan": "free",
  "stage": "profile_complete",
  "signals_count": 0,
  "memory_count": 0,
  "ttfi_seconds": null,
  "activation_score": 0.1111
}
```

`ttfi_seconds` is time-to-first-insight; null until the org reaches the `first_brief` stage. `activation_score` ∈ [0.0, 1.0].

---

## 2.7 `PUT /api/v1/organization/plan` — Change plan

**Required permission:** `manage_billing` (founder + admin by default).

**Request body:**
```json
{ "plan": "pro" }
```

`plan` must be one of `"free"`, `"pro"`, `"team"`, `"enterprise"`.

**Success — 200:**
```json
{ "plan": "pro" }
```

**Side-effect:** entitlements are re-seeded for the new plan. `/entitlements` reflects the change immediately.

---

## 2.8 `PUT /api/v1/organization/flags/{flag}` — Toggle a feature flag

**Required permission:** `manage_billing`.

**Request body:**
```json
{ "enabled": true, "kind": "release" }
```

`kind` is one of `"release"`, `"permission"`, `"kill_switch"`. Default `"release"`.

A kill-switch flag named `kill:<capability>` (e.g. `kill:gmail`) forces that capability off regardless of plan/entitlement.

**Success — 200:**
```json
{ "flag": "kill:gmail", "enabled": true }
```

---

# 3. Identity Admin (delegated administration)

All routes here are prefixed `/api/v1/identity/...` and **require a specific permission**.

| Endpoint | Method | Required Permission |
|---|---|---|
| `/identity/members` | GET | `manage_members` |
| `/identity/members/invite` | POST | `manage_members` |
| `/identity/members/{user_id}` | DELETE | `manage_members` |
| `/identity/members/{user_id}/role` | POST | `manage_roles` |
| `/identity/api-keys` | GET | `manage_api_keys` |
| `/identity/api-keys` | POST | `manage_api_keys` |
| `/identity/api-keys/{key_id}` | DELETE | `manage_api_keys` |
| `/identity/service-accounts` | GET | `manage_service_accounts` |
| `/identity/service-accounts` | POST | `manage_service_accounts` |
| `/identity/service-accounts/{sa_id}/rotate` | POST | `manage_service_accounts` |
| `/identity/service-accounts/{sa_id}` | DELETE | `manage_service_accounts` |
| `/identity/audit` | GET | `view_audit` |
| `/identity/sso` | PUT | `manage_sso` |

If the caller lacks the permission → `403`. The error message is generic — don't show it to end-users.

## 3.1 List members
`GET /api/v1/identity/members?limit=50&offset=0`
```json
{
  "members": [{ "user_id": "uuid", "role": "founder" }],
  "limit": 50, "offset": 0, "count": 1
}
```

## 3.2 Invite member
`POST /api/v1/identity/members/invite`
```json
{ "email": "newhire@acme.example", "role": "employee" }
```
Success **201**:
```json
{ "user_id": "uuid", "role": "employee" }
```
Errors: `403` (caller can't grant that role), `409` (already a member).

## 3.3 Remove member
`DELETE /api/v1/identity/members/{user_id}`
Success **200**: `{ "status": "removed" }`. Errors: `404` (not a member), `409` (would leave the org without a founder).

## 3.4 Change role
`POST /api/v1/identity/members/{user_id}/role`
```json
{ "role": "admin" }
```
Success **200**: `{ "user_id": "uuid", "role": "admin" }`. A caller cannot grant a role with permissions they themselves lack.

## 3.5 API keys (machine credentials for org-scoped automation)
- `GET /identity/api-keys` — list (returns prefix, not secret).
- `POST /identity/api-keys` body `{ "name": "...", "scopes": ["execute_actions", ...] }` — returns the secret **once**. Store it client-side immediately.
- `DELETE /identity/api-keys/{id}` — revoke.

**Important:** API key scopes are limited to permissions the creator holds. Cannot grant scopes you don't have (403).

## 3.6 Service accounts
Same pattern as API keys but represent a bot identity (not a key for a user). Adds:
- `POST /identity/service-accounts/{sa_id}/rotate` — returns a new secret; old secret invalidated.

## 3.7 Audit log
`GET /api/v1/identity/audit?limit=100&offset=0&since=2026-06-01T00:00:00Z`
```json
{
  "audit": [{
    "action": "login",
    "actor_id": "uuid", "actor_kind": "user",
    "target_id": "uuid",
    "metadata": {"method": "password", "request_id": "..."},
    "occurred_at": "2026-06-30T09:00:00+00:00"
  }],
  "limit": 100, "offset": 0, "count": 1
}
```

## 3.8 Configure SSO
`PUT /api/v1/identity/sso`
```json
{ "provider": "google", "config": { ... provider-specific ... }, "enabled": true }
```

---

# 4. Roles & Permissions Reference

> **This section is documentation, not an implementation spec.** Read the
> caller's actual permissions from `GET /auth/me` (§1.6) — that response
> accounts for per-org grants, which this table cannot. Reproducing the table
> below in client code creates a second source of truth that will drift.

## 4.1 Built-in roles

| Role | Default permissions |
|---|---|
| `founder` | Everything (all permissions) |
| `admin` | View signals/memory/state/reasoning/decisions/intelligence + manage members/roles/integrations/api-keys/service-accounts/sso + view audit |
| `ceo` | Views + execute_actions + manage_members + manage_policies + manage_integrations + view_audit |
| `coo` | Views + execute_actions + manage_integrations + manage_members |
| `chief_of_staff` | Views + execute_actions + manage_members |
| `employee` | view_decisions + view_signals only |

## 4.2 Full permission vocabulary

**Reads:** `view_signals`, `view_memory`, `view_state`, `view_reasoning`, `view_decisions`, `view_intelligence`, `view_audit`

**Operations:** `execute_actions`, `manage_integrations`, `manage_policies`

**Administration:** `manage_members`, `manage_roles`, `manage_api_keys`, `manage_service_accounts`, `manage_sso`, `manage_billing`

UI tip: hide buttons/menus the user lacks permission for — gate on
`permissions.includes('…')` from §1.6, not on a locally derived set. The exact
permission required for each Layer-1 admin route is in the table in §3.

---

## 4.3 Routes that were never real (removed)

These five paths used to answer `200` with an empty payload **without checking
credentials**. They had no backing implementation, so an anonymous caller, an
expired session, and a genuinely empty organization were indistinguishable on the
wire — a dead session rendered as "nothing to do today" instead of a re-login.

They are **gone**. All five now `404`:

| Removed path | What to use instead |
|---|---|
| `GET /decide/stack` | `GET /brief/today` — the ranked decision list |
| `GET /skills` | nothing; no skills domain is shipped |
| `GET /signals` | `GET /signals/gmail/recent`, `/signals/calendar/upcoming`, `/signals/drive/recent` |
| `GET /execution/slack/blockers` | nothing; not shipped |
| `GET /workspace/meetings/{id}/brief` | nothing; not shipped |

The frontend already tolerates four of these (`fetchJsonOr` maps `404` to the
empty fallback). `getMeetingBrief` uses `fetchJson`, so the meeting-brief drawer
now renders its existing error state rather than a fabricated blank brief —
switch it to `fetchJsonOr` if an empty state reads better than an error.

**Every remaining route requires authentication.** The only endpoints that serve
an anonymous caller are `/auth/*` (which mint or exchange credentials), the
Google OAuth callback (signed `state`), and the `/webhooks/*` endpoints
(provider signature). A `200` from anything else means you were authenticated.

---

# 5. Token handling cheatsheet

### Storage
- **Access token:** in-memory (a JS variable or React state). Short-lived (~15 min).
- **Refresh token:** httpOnly cookie OR localStorage (whatever your security model dictates). Long-lived (~30 days).
- Never put the access token in localStorage if you can avoid it.

### Auto-refresh pattern
```ts
async function fetchWithAuth(url, opts = {}) {
  let resp = await fetch(url, { ...opts, headers: { ...opts.headers, Authorization: `Bearer ${accessToken}` } });
  if (resp.status !== 401) return resp;

  // Try to rotate once
  const r = await fetch(`${API}/api/v1/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });
  if (!r.ok) {
    // Refresh failed — full logout, redirect to login
    redirectToLogin();
    throw new Error("session expired");
  }
  const tokens = await r.json();
  accessToken = tokens.access_token;
  refreshToken = tokens.refresh_token;

  // Retry the original request once with the new token
  return fetch(url, { ...opts, headers: { ...opts.headers, Authorization: `Bearer ${accessToken}` } });
}
```

### Logout
1. Call `POST /api/v1/auth/logout` with the current access token.
2. Clear both tokens from client storage.
3. Redirect to landing.

Even if the logout request fails, **always clear local storage** — server-side state will eventually expire on its own.

---

# 6. Common pitfalls

1. **Don't trust `/auth/me` for permission gating** — it returns `role`, but actual gating is permission-based. Backend always re-checks. Show/hide UI on `role` for UX, never assume it's enforcement.
2. **Refresh token rotation:** every successful refresh returns a NEW refresh token. **Always store the new one** — the old one is dead the instant the new one is issued. Holding onto the old triggers reuse detection and kills the whole session.
3. **Don't disambiguate auth errors:** `401 invalid email or password` covers wrong password, unknown email, disabled account, AND locked-from-throttle. Show the same message in the UI for all of them.
4. **Google SSO needs the Google JS client on the frontend** — backend only accepts a `code` or `id_token` produced by Google's `initCodeClient` / `accounts.id`. Backend cannot mint these.
5. **Onboarding gate:** `is_new_user` AND `onboarding_completed` in the token response cover the first-load case. For subsequent navigations, hit `/organization/onboarding/status` — it's the cheapest check.
6. **Plan vs. entitlement:** plan is the billing tier; entitlements are the actual per-capability gate. Render UI off `/entitlements`, not off `plan` — kill switches and explicit grants can diverge from the plan default.

---

# 7. Status code summary

| Code | Meaning |
|---|---|
| 200 | OK (success with body) |
| 201 | Created (signup, invite, key/sa create) |
| 400 | Business-logic refusal (email taken, invalid plan transition) |
| 401 | Auth missing/invalid/expired; or generic-credential failure on login |
| 403 | Authenticated but lacks the required permission |
| 404 | Resource not found in this org's scope |
| 409 | Conflict (already a member, would orphan org of last founder) |
| 422 | Schema validation failure (invalid email, missing field) |
| 5xx | Server bug — report a ticket with the `X-Request-ID` response header |

Every response includes an `X-Request-ID` header. Include it in bug reports — server logs are indexed by it.
