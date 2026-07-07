# Navis — Frontend Architecture & Flow Guide

> **Hand-off note (paste this to the frontend Claude session):**
> Build a frontend for Navis using this document as the single source of truth.
> The backend is **already shipped and frozen** for the surfaces listed here — do
> not invent endpoints, payload shapes, or auth flows. If something you need is
> missing, ask before assuming. Cross-reference `docs/frontend_api_contract.md`
> for exact request/response shapes of the identity + organization surface; this
> document is the architecture, route map, state model, and UX flow.

---

## 0. Product in one paragraph

Navis is an **organizational intelligence** platform. Companies plug in their
signal sources (Gmail, Calendar, Drive, Slack); Navis builds a **Memory** of
people / projects / commitments / decisions, derives **State** of the org,
runs **Reasoning** engines, and produces a daily **Brief** of what matters and
what to act on. The founder/CEO approves **Actions** (emails, tasks, meetings);
**Observation** records outcomes; **Learning** improves future decisions; the
**Intelligence Score** tracks the org's overall organizational IQ over time.

The frontend is the **operator console** for that pipeline — auth + onboarding,
daily brief, action approval queue, memory/insights views, member + integration
admin, and adaptive policy management.

---

## 1. Tech stack (recommended)

| Concern | Choice | Why |
|---|---|---|
| Framework | **Next.js 15 (App Router) + React 19 + TypeScript** | SSR-friendly auth, file-based routing maps cleanly to our route plan, RSC for read-mostly pages |
| Styling | **Tailwind CSS v4 + shadcn/ui** | shadcn primitives match the dense data-heavy UI; Tailwind keeps the bundle lean |
| State (server) | **TanStack Query v5** | per-route cache + retry-on-401 hook integrates with refresh flow |
| State (client) | **Zustand** (single `useAuthStore` + `useOrgStore`) | no Redux ceremony; survives route changes |
| Forms | **react-hook-form + zod** | mirror backend Pydantic validation client-side |
| HTTP | Native `fetch` wrapped in `apiClient.ts` | full control over token rotation; no axios |
| Auth tokens | access in memory (Zustand) · refresh in `httpOnly` cookie via Next.js route handler | meets the contract in §5 of `frontend_api_contract.md` |
| Charts | **Recharts** | activation score, intelligence score, brief trends |
| Icons | **lucide-react** | shadcn default |
| Dates | **date-fns + date-fns-tz** | server returns ISO UTC; UI renders in user TZ |
| Tests | **Vitest + Playwright** | unit for hooks/components, E2E for the auth+onboarding+brief golden path |

> If you must use Vite + React Router instead, the route map in §4 still applies;
> only the file paths change. Do **not** introduce Redux/MobX/Recoil.

---

## 2. Folder layout

```
app/
  layout.tsx                       # root: <html>, AuthProvider, QueryClient
  page.tsx                         # marketing/landing → CTA to /login
  (auth)/
    login/page.tsx
    signup/page.tsx
    callback/google/page.tsx       # Google OAuth popup target
  (onboarded)/                     # group requires onboarding_completed=true
    layout.tsx                     # AppShell (sidebar + topbar)
    dashboard/page.tsx             # Today's Brief
    actions/
      page.tsx                     # approval queue
      [id]/page.tsx                # action detail
    memory/page.tsx
    insights/page.tsx
    integrations/
      page.tsx                     # list
      google/callback/page.tsx     # OAuth return
    settings/
      profile/page.tsx
      organization/page.tsx
      members/page.tsx
      api-keys/page.tsx
      service-accounts/page.tsx
      sso/page.tsx
      audit/page.tsx
      plan/page.tsx
    adaptive/
      policy/page.tsx
      proposals/page.tsx
  onboarding/
    page.tsx                       # profile form → PUT /organization/profile
components/
  ui/                              # shadcn primitives
  auth/AuthGuard.tsx               # protects (onboarded) group
  auth/OnboardingGuard.tsx
  shell/Sidebar.tsx, Topbar.tsx
  brief/BriefCard.tsx, PriorityList.tsx
  actions/ActionRow.tsx, ApprovalDialog.tsx
  permission/Can.tsx               # <Can perm="manage_members">{...}</Can>
  entitlement/Feature.tsx          # <Feature cap="execution">{...}</Feature>
lib/
  api/client.ts                    # fetchWithAuth + refresh logic (§6)
  api/endpoints.ts                 # typed wrappers, one per backend route
  api/types.ts                     # generated/hand-written DTOs
  auth/store.ts                    # Zustand: accessToken, user, org, role
  auth/google.ts                   # Google Identity Services loader
  permissions/roles.ts             # role → permissions map (mirrors §4 of contract)
  permissions/has.ts               # has(perm) helper
  entitlements/store.ts            # cached /entitlements result
  errors/toast.ts                  # uniform error → toast mapping
hooks/
  useMe.ts                         # GET /auth/me (TanStack)
  useEntitlements.ts
  useOnboardingStatus.ts
  useBriefToday.ts
  useActions.ts
```

---

## 3. Global state model

Three Zustand stores. Everything else is server state (TanStack Query).

### `useAuthStore`
```ts
{
  accessToken: string | null;          // in memory ONLY
  refreshToken: string | null;         // hydrated from httpOnly cookie via /api/auth/bootstrap
  user: { id, email, full_name, is_active, is_verified, created_at } | null;
  organization: { id, name, slug } | null;
  role: 'founder' | 'admin' | 'ceo' | 'coo' | 'chief_of_staff' | 'employee' | null;
  onboardingCompleted: boolean;
  setTokens(tp: TokenPair): void;
  clear(): void;                       // on logout / refresh failure
}
```

### `useEntitlementsStore`
```ts
{
  capabilities: Record<Capability, boolean>;   // from GET /organization/entitlements
  plan: 'free' | 'pro' | 'team' | 'enterprise' | null;
  lifecycle: 'trial' | 'active' | 'suspended' | 'churned' | null;
  refresh(): Promise<void>;
}
```

### `useUiStore`
```ts
{
  sidebarCollapsed: boolean;
  theme: 'system' | 'light' | 'dark';
  commandPaletteOpen: boolean;
}
```

**Hydration order on app boot:**
1. Restore refresh token (cookie → Zustand) via a Next.js route handler.
2. Call `POST /auth/refresh` to mint a fresh access token.
3. Call `GET /auth/me` → populate user/org/role/onboardingCompleted.
4. Call `GET /organization/entitlements` → populate capabilities.
5. Render. Until step 3 completes, show a single full-screen spinner — never
   flash the login page to an authenticated user.

If step 2 fails → clear store, route to `/login`.

---

## 4. Route map (sitemap)

Every route lists: **purpose · data calls · permission/capability gate · navigation rules**.

### Public

| Route | Purpose | Calls | Gate |
|---|---|---|---|
| `/` | Marketing landing | — | none |
| `/login` | Email/password + "Continue with Google" | `POST /auth/login`, `POST /auth/google` | redirect to `/dashboard` if already authed |
| `/signup` | New org + founder signup | `POST /auth/signup` | same |
| `/callback/google` | Receives `code` from Google popup, sends to backend | `POST /auth/google` | route is hit by Google's redirect |

### Onboarding (authed, but `onboarding_completed=false`)

| Route | Purpose | Calls | Gate |
|---|---|---|---|
| `/onboarding` | Org profile form (single page; see §7.2) | `GET /organization/profile`, `PUT /organization/profile` | requires auth; redirects to `/dashboard` if already complete |

### App shell (authed AND onboarded)

| Route | Purpose | Calls | Gate |
|---|---|---|---|
| `/dashboard` | **Today's Brief** | `GET /brief/today` (fall back to `POST /brief/generate`), `GET /organization/metrics` | capability: `decision` (show empty state otherwise) |
| `/actions` | Pending action approval queue | `GET /actions` | capability: `execution`, permission: `view_decisions` to view, `execute_actions` to act |
| `/actions/[id]` | Action detail + approve/reject/cancel | `POST /actions/{id}/approve\|reject\|cancel` | permission: `execute_actions` |
| `/memory` | Browse memory entities (people/projects/commitments/decisions) | (NB: read endpoints not in this slice — see §11) | capability: `memory`, permission: `view_memory` |
| `/insights` | Intelligence insights list + score | `GET /intelligence/insights`, `POST /intelligence/project` | capability: `intelligence`, permission: `view_intelligence` |
| `/integrations` | Connect/disconnect Gmail, Calendar, Drive, Slack | `GET /integrations/google/connect`, OAuth popup → callback | capability gates each tile (`gmail`, `calendar`, `drive`, `slack`); permission: `manage_integrations` |
| `/adaptive/policy` | View active policy | `GET /adaptive/policy` | capability: `adaptive` |
| `/adaptive/proposals` | Pending proposals + approve/reject/rollback/kill | `GET /adaptive/proposals`, `POST /adaptive/proposals/{id}/approve\|reject`, `POST /adaptive/rollback/{policy_id}`, `POST /adaptive/kill-switch` | capability: `adaptive`, permission: `manage_policies` |
| `/settings/profile` | Edit user name / password (TBD) | `GET /auth/me` | self |
| `/settings/organization` | Edit org profile post-onboarding | `GET/PUT /organization/profile` | permission: `manage_members` |
| `/settings/members` | List + invite + role-change + remove | `GET /identity/members`, `POST /identity/members/invite`, `POST /identity/members/{id}/role`, `DELETE /identity/members/{id}` | permission: `manage_members` (+ `manage_roles` for role changes) |
| `/settings/api-keys` | Create + revoke API keys | `GET/POST /identity/api-keys`, `DELETE /identity/api-keys/{id}` | permission: `manage_api_keys` |
| `/settings/service-accounts` | Create + rotate + revoke service accounts | `GET/POST /identity/service-accounts`, `POST /identity/service-accounts/{id}/rotate`, `DELETE …/{id}` | permission: `manage_service_accounts` |
| `/settings/sso` | Configure SSO | `PUT /identity/sso` | permission: `manage_sso` |
| `/settings/audit` | Audit log table | `GET /identity/audit` | permission: `view_audit` |
| `/settings/plan` | Plan selector + feature flags | `GET /organization/metrics`, `PUT /organization/plan`, `PUT /organization/flags/{flag}` | permission: `manage_billing` |

### Route guards

Implement **three guards**, composable as React components:

```tsx
<AuthGuard>                          // redirects unauthed → /login
  <OnboardingGuard>                  // redirects !onboardingCompleted → /onboarding
    <PermissionGuard perm="…">       // 403 page if user lacks perm
      <CapabilityGuard cap="…">      // upgrade-CTA page if entitlement off
        {page}
      </CapabilityGuard>
    </PermissionGuard>
  </OnboardingGuard>
</AuthGuard>
```

The backend always re-checks. Guards exist for **UX only** (don't fetch data
the user can't see, don't show buttons that will 403).

---

## 5. The five flows (end-to-end)

### 5.1 Signup → onboarding → dashboard

```
/signup
  POST /auth/signup → { access, refresh, is_new_user, onboarding_completed }
  store tokens; populate /auth/me; route to /onboarding (since is_new_user=true)

/onboarding
  user fills company_name + industry + team_size + org_stage (+ optional rest)
  PUT /organization/profile
  → side-effect: backend advances stage to profile_complete
  POST refetch /auth/me (now onboarding_completed=true)
  route to /dashboard
```

### 5.2 Google SSO

```
Login page renders Google's initCodeClient button (popup mode).
On success → frontend has `code`.
POST /auth/google { code } → TokenPair { is_new_user, onboarding_completed }
Same routing as 5.1: onboarding_completed=false → /onboarding, else /dashboard.
```

**Do not** use Google's redirect flow — backend expects `code` from `initCodeClient`
with `redirect_uri: 'postmessage'`. See `frontend_api_contract.md §1.3`.

### 5.3 Daily brief

```
/dashboard mounts:
  GET /brief/today
    200 with brief → render
    404 (no brief yet) → POST /brief/generate → render result
    503 (state not ready) → show "We're still building your org's state" empty state
                            with last-updated timestamp from /organization/metrics
```

The brief is the **hero** of the product. Layout: hero header (date, greeting),
"What matters today" priority list (3-5 cards), "Risks & opportunities" sidebar,
"Action proposals" footer with one-click "Propose actions" CTA →
`POST /brief/propose-actions` → routes to `/actions`.

### 5.4 Action approval queue

```
/actions:
  GET /actions → list grouped by status (pending / approved / executed / failed)
  Each row: action type icon, target (email recipient / task title / meeting),
            preview, "Approve" / "Reject" buttons (hidden if no execute_actions perm).
  Approve → POST /actions/{id}/approve → optimistic update; on success refetch.
  Failed actions show retry/cancel + error reason from response.
```

Surface a **risk badge** on each action: low / medium / high based on backend
metadata. Default the queue to "pending" tab; show count badges in the sidebar.

### 5.5 Connect integrations

```
/integrations:
  Tiles for Gmail / Calendar / Drive / Slack, each gated by capability.
  Connect tile → GET /integrations/google/connect (returns OAuth URL)
                → open in popup
                → backend callback writes tokens
                → popup posts message to opener → close
                → frontend invalidates entitlements + integration state queries.
  Connected state shows last-sync time, "Disconnect" (TBD endpoint).
```

Slack uses the Events API; "Connect Slack" links to the install URL Slack
provides (not in this backend slice — frontend should expose a configurable env
var `NEXT_PUBLIC_SLACK_INSTALL_URL`).

---

## 6. API client contract

### 6.1 `apiClient.ts` — the only file that touches `fetch`

Responsibilities:
- Inject `Authorization: Bearer ${accessToken}` on every protected call.
- On `401`, call `POST /auth/refresh` **once**, rotate both tokens in
  `useAuthStore`, retry the original request **once**.
- If refresh itself returns `401` → clear store, route to `/login`, throw.
- Surface `X-Request-ID` from every response — attach to errors so bug reports include it.
- Map status codes to typed errors: `AuthError(401) | PermissionError(403) | NotFoundError(404) | ConflictError(409) | ValidationError(422) | ServerError(5xx)`.
- Never retry mutating verbs (POST/PUT/DELETE) on network failure — surface to the user.

Skeleton in `frontend_api_contract.md §5` is the reference implementation.

### 6.2 `endpoints.ts` — one typed function per backend route

Pattern:
```ts
export const api = {
  auth: {
    signup: (body: SignupBody) => client.post<TokenPair>('/auth/signup', body),
    login: (body: LoginBody) => client.post<TokenPair>('/auth/login', body),
    google: (body: { code?: string; id_token?: string }) => client.post<TokenPair>('/auth/google', body),
    refresh: (body: { refresh_token: string }) => client.post<TokenPair>('/auth/refresh', body),
    logout: () => client.post<{ status: string }>('/auth/logout'),
    me: () => client.get<MeResponse>('/auth/me'),
  },
  org: {
    getProfile: () => client.get<{ profile: OrgProfile | null }>('/organization/profile'),
    putProfile: (body: Partial<OrgProfile>) => client.put('/organization/profile', body),
    onboardingStatus: () => client.get<{ stage: Stage; onboarding_completed: boolean }>('/organization/onboarding/status'),
    onboarding: () => client.get<{ stage: Stage; signals_count: number; memory_count: number }>('/organization/onboarding'),
    entitlements: () => client.get<{ entitlements: Record<Capability, boolean> }>('/organization/entitlements'),
    metrics: () => client.get<OrgMetrics>('/organization/metrics'),
    setPlan: (plan: Plan) => client.put('/organization/plan', { plan }),
    setFlag: (flag: string, body: { enabled: boolean; kind: 'release'|'permission'|'kill_switch' }) =>
      client.put(`/organization/flags/${flag}`, body),
  },
  identity: { /* members, api-keys, service-accounts, audit, sso */ },
  brief: {
    today: () => client.get<DailyBrief>('/brief/today'),
    generate: () => client.post<DailyBrief>('/brief/generate'),
    proposeActions: () => client.post<{ created: string[] }>('/brief/propose-actions'),
  },
  actions: {
    list: () => client.get<ActionRequest[]>('/actions'),
    create: (body: CreateActionBody) => client.post<ActionRequest>('/actions', body),
    approve: (id: string) => client.post(`/actions/${id}/approve`),
    reject: (id: string, reason?: string) => client.post(`/actions/${id}/reject`, { reason }),
    cancel: (id: string) => client.post(`/actions/${id}/cancel`),
  },
  intelligence: {
    insights: () => client.get<Insight[]>('/intelligence/insights'),
    project: () => client.post('/intelligence/project'),
  },
  adaptive: {
    policy: () => client.get<Policy>('/adaptive/policy'),
    proposals: () => client.get<Proposal[]>('/adaptive/proposals'),
    propose: () => client.post('/adaptive/propose'),
    approveProposal: (id: string) => client.post(`/adaptive/proposals/${id}/approve`),
    rejectProposal: (id: string) => client.post(`/adaptive/proposals/${id}/reject`),
    rollback: (policyId: string) => client.post(`/adaptive/rollback/${policyId}`),
    killSwitch: () => client.post('/adaptive/kill-switch'),
  },
};
```

### 6.3 TanStack Query keys

Stable, hierarchical keys keep invalidation clean:
```
['me']
['org', 'profile']
['org', 'onboarding']
['org', 'entitlements']
['org', 'metrics']
['identity', 'members']
['identity', 'api-keys']
['identity', 'service-accounts']
['identity', 'audit', { limit, offset, since }]
['brief', 'today']
['actions', { status }]
['intelligence', 'insights']
['adaptive', 'policy']
['adaptive', 'proposals']
```

After `PUT /organization/profile` → invalidate `['me']`, `['org', 'profile']`, `['org', 'onboarding']`, `['org', 'entitlements']`.

After `POST /actions/{id}/approve` → invalidate `['actions']`.

After `PUT /organization/plan` or `PUT /organization/flags/*` → invalidate `['org', 'entitlements']`.

---

## 7. Page-by-page UX spec

### 7.1 Login (`/login`)

- Centered card. Email + password + submit. Below: "Continue with Google".
- Wire Google button via `lib/auth/google.ts` (loads `https://accounts.google.com/gsi/client`).
- One generic error toast for all 401s: *"Invalid email or password."* Never disambiguate.
- After 5 consecutive failures show a 15-min lockout hint (do NOT count client-side; the backend enforces — just soften the message after the 5th 401).
- Link to `/signup`.

### 7.2 Signup (`/signup`)

- Fields: email, password, full_name, organization_name.
- Client validation mirrors `frontend_api_contract.md §1.1` field rules; password strength meter shown live.
- On success, route per `is_new_user` / `onboarding_completed`.

### 7.3 Onboarding (`/onboarding`)

- Single-page form, two columns:
  - Left: progress steps (visual only — backend stage advances on first PUT).
  - Right: form (company_name, website, industry, employee_count, hq_country [country picker], team_size [select], org_stage [select], primary_goal [textarea]).
- "Save and continue" → `PUT /organization/profile`.
- After success: refetch `/auth/me`; if `onboarding_completed=true` → route to `/dashboard`.
- A subtle "Skip for now" link is allowed but warn the user that some features need the profile to unlock.

### 7.4 Dashboard / Today's Brief (`/dashboard`)

Hero layout:
```
┌──────────────────────────────────────────────────────────────┐
│  Good morning, Jane.   Tuesday, June 30 · Acme Co            │
│  Activation 0.42  ·  Intelligence Score 64                   │
├──────────────────────────────────────────────────────────────┤
│  What matters today                                          │
│  ┌──────────────────────┐ ┌──────────────────────┐            │
│  │ priority card        │ │ priority card        │            │
│  └──────────────────────┘ └──────────────────────┘            │
├──────────────────────────────────────────────────────────────┤
│  Risks & opportunities      │   Suggested actions             │
│  • …                        │   [Propose actions] →           │
└──────────────────────────────────────────────────────────────┘
```

States:
- **Loading:** skeleton hero + 3 skeleton cards.
- **Empty (no brief, capability on):** "We don't have enough signal yet — connect Gmail / Calendar from Integrations."
- **Capability off (`decision=false`):** Upgrade CTA → `/settings/plan`.
- **Error:** retry button + request ID.

### 7.5 Actions (`/actions`)

Table view with status tabs: Pending · Approved · Executed · Failed.
Row columns: type (icon), summary, target, risk badge, created_at, actions (approve/reject buttons).
Click row → drawer or `/actions/[id]` detail page (mobile: full page; desktop: side drawer).

### 7.6 Members (`/settings/members`)

- Table: avatar · name · email · role · joined · actions menu.
- "Invite" opens dialog (email + role select, scoped to roles the caller can grant).
- Role cell is editable inline if caller has `manage_roles`.
- Cannot remove the last founder — backend returns 409, show toast.

### 7.7 API keys (`/settings/api-keys`)

- Create dialog: name + scopes (multi-select from permissions the caller holds).
- After create, show the secret **once** in a copyable dialog with a giant "I've stored it safely" confirm — never display it again.
- Table: name · prefix · scopes · created · last-used · revoke.

### 7.8 Audit (`/settings/audit`)

- Filterable table (since-date picker, action filter, actor filter).
- Pagination via `limit`/`offset`.
- Each row expandable to show `metadata` JSON pretty-printed.

### 7.9 Adaptive (`/adaptive/proposals`)

- Two-column: active policy summary (left) · proposals (right).
- Each proposal card: diff vs. active policy + Approve / Reject.
- Sticky footer: "Rollback to previous policy" + "Kill switch (revert to default)" — both behind a confirm dialog.

---

## 8. Permission & entitlement gating patterns

### 8.1 `<Can perm="...">` (permission gate)

Computes `hasPerm(role, perm)` from the role→permissions map in
`frontend_api_contract.md §4`. Hides the child if false. **Never** wrap a whole
route in `<Can>` — use `<PermissionGuard>` for that (which shows a proper 403
page). `<Can>` is for buttons, menu items, table action columns.

### 8.2 `<Feature cap="...">` (entitlement gate)

Reads `useEntitlementsStore`. If `capabilities[cap] !== true`:
- For inline UI: render disabled state with a tooltip "Upgrade to unlock."
- For full pages: render a centered upgrade card → `/settings/plan`.

### 8.3 Order of guards

`Auth → Onboarding → Permission → Capability → page`. A user without the
required *permission* sees a 403 page; a user with permission but without the
*capability* sees an upgrade page.

### 8.4 Role/permission source of truth

Hard-code the role→permission table in `lib/permissions/roles.ts`. It mirrors
`frontend_api_contract.md §4`. If the backend roles table ever changes, update
this file in the same PR. The backend always re-enforces, so a stale UI map
just means wrong-looking buttons, not a security hole.

---

## 9. Error handling & request tracing

- **Every** mutation goes through a single `useMutationWithErrorToast` wrapper that:
  - Reads `X-Request-ID` from the response.
  - On 4xx: shows `error.detail` in a toast + the request ID in monospace at the bottom.
  - On 5xx: shows "Something went wrong. Reference: `<X-Request-ID>`" with a "Copy" button.
  - On 422: focuses the first invalid field by `loc` path.
- **Never** leak raw error JSON to users.
- Log every request ID + status to `console.debug` in dev; ship to Sentry/Datadog in prod.

---

## 10. Auth/token lifecycle (must-have details)

1. Access token lives **only** in `useAuthStore` (memory). Page reload = lost.
2. Refresh token lives in `httpOnly; Secure; SameSite=Lax` cookie, set by a
   Next.js route handler `app/api/auth/set-cookie/route.ts` that the
   `apiClient` calls after every successful `TokenPair` response (signup / login
   / google / refresh).
3. On boot, a `app/api/auth/bootstrap/route.ts` reads the cookie server-side
   and returns the refresh token to the client store **only** in the bootstrap
   step (so the client can call `/auth/refresh`). Alternative: the route
   handler itself calls `/auth/refresh` server-side and returns the new
   `TokenPair`; this is cleaner and avoids ever shipping the refresh token to
   the browser JS. Prefer this.
4. After a successful refresh: **replace both** tokens. Holding the old refresh
   token triggers backend reuse detection and revokes the entire session.
5. On logout: `POST /auth/logout`, clear cookie, clear Zustand, route to `/`.
   Do this **even if the logout call fails** — never leave the user "stuck logged in".

---

## 11. What the backend does NOT yet expose (and how the frontend should behave)

| Need | Status | Frontend handling |
|---|---|---|
| Memory entity read endpoints (people, projects, commitments) | not in this slice | `/memory` page renders "Coming soon — your memory is being built." Read `signals_count` + `memory_count` from `/organization/onboarding` to show progress. |
| Slack OAuth install URL from backend | not exposed | Use `NEXT_PUBLIC_SLACK_INSTALL_URL` env var. |
| Disconnect integration | not exposed | Hide "Disconnect" or surface "Contact support" until added. |
| WebSocket / SSE for live brief updates | none | Use TanStack Query `refetchInterval: 60_000` on `/brief/today` and `/actions`. Do **not** poll faster than 60s. |
| Password reset / email verification | not exposed | Hide "Forgot password" until added. |
| User profile self-edit | not exposed | `/settings/profile` is read-only until backend ships an endpoint. |

Do not invent these endpoints. Render disabled UI with a "coming soon" tooltip
and surface the gap in the PR description.

---

## 12. Design language

- **Visual:** Linear-esque density meets Stripe Dashboard clarity. Generous
  whitespace on hero/brief; compact tables on settings. Mono font for IDs,
  request IDs, and code-like values (slugs, plan keys).
- **Color:** neutral grayscale base; one accent (indigo); semantic red/amber/green for risk badges and action status. Support `prefers-color-scheme: dark`.
- **Empty states:** every list view has a deliberate empty state with a one-line "what to do next" CTA. No blank rectangles.
- **Loading:** prefer **skeletons** over spinners for content-heavy pages; spinners only for button-level actions.
- **Toasts:** top-right, 4s auto-dismiss for success, sticky for errors, action-button for retryable failures.
- **Mobile:** sidebar collapses to a bottom nav under 768 px. Brief, actions queue, and settings must work on mobile. Admin tables (audit, members) can horizontal-scroll on mobile.

---

## 13. Accessibility floor (non-negotiable)

- All interactive elements keyboard-reachable; visible focus rings.
- shadcn dialogs/menus already trap focus — don't override.
- Forms: every input has a `<label>`; error messages reference the input via `aria-describedby`.
- Color contrast ≥ WCAG AA on the chosen palette.
- Tables use `<th scope="col">` and `<caption>`.

---

## 14. Build order (recommended)

Ship in this sequence — each milestone is usable end-to-end:

1. **Skeleton + auth:** layout, Zustand stores, `apiClient`, `/login`, `/signup`, `/callback/google`, `/auth/me` bootstrap.
2. **Onboarding gate:** `/onboarding` form, guards.
3. **Dashboard with Brief:** `/dashboard`, empty/loading/error states, capability gate.
4. **Actions queue:** `/actions` + approve/reject, propose-actions wiring.
5. **Settings — Members + Audit:** admin loop.
6. **Settings — Plan + Flags + Entitlements wiring.**
7. **Integrations:** Google OAuth popup flow.
8. **Insights + Intelligence Score.**
9. **Adaptive policy + proposals.**
10. **API keys, service accounts, SSO config.**
11. **Memory views** (once backend endpoints land).

Each milestone gets a Playwright E2E test before the next starts.

---

## 15. Hard rules (do not break)

1. **Never call a backend endpoint not listed in `frontend_api_contract.md` or in §4 of this doc.** If you need one, ask first.
2. **Never store the access token in `localStorage` or a non-httpOnly cookie.**
3. **Never persist the old refresh token after a refresh succeeds** — replace immediately.
4. **Never disambiguate 401 errors on the login form.** One message: "Invalid email or password."
5. **Never gate UI on `role` alone for security** — backend re-checks. Role is for UX hints; permission map is for showing/hiding controls.
6. **Never poll faster than 60 s.** No live endpoints exist; respect the backend.
7. **Never hardcode org IDs or user IDs** — the access token scopes everything; the org is implicit.
8. **Every error toast must include the `X-Request-ID`.** No exceptions.
9. **Don't introduce a new state-management library.** Zustand + TanStack Query covers everything in §3.
10. **Don't add backend-shaped logic to the frontend** (no client-side capability calculation, no client-side rate limiting, no client-side audit logging). Read from the API.

---

## 16. Reference appendix

- **Exact request/response shapes:** `docs/frontend_api_contract.md` (canonical).
- **Backend architecture (for context, not for building against):** `docs/ARCHITECTURE.md`.
- **Identity deep-dive (auth, sessions, refresh rotation, SSO):** `docs/IDENTITY_FILE_MAP.md`, `docs/identity_audit_report.md`.
- **OpenAPI:** `GET /docs` on the running backend (`http://127.0.0.1:8000/docs`) — use it to verify field types and enum values before coding any form.

---

**That's the whole frontend.** Build it page-by-page in the order in §14,
honor the rules in §15, and the integration will work the first time.
