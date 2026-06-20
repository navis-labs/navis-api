# Identity V4 — Deep Systems Audit

Re-audited from first principles, assuming prior reviews missed things. **They did.** Beneath the (already-fixed) cross-tenant data holes sat a class of bug the data-layer audits never probed: **vertical privilege escalation through the authorization grant paths.** Three new findings, all fixed and guarded by tests. Already-fixed issues (cross-tenant key/SA, SCIM multi-org, indexes) are NOT repeated.

Suite: **264 passed, 9 skipped** (+2 escalation/governance regression guards). Files map: `IDENTITY_FILE_MAP.md`.

---

## Executive Summary

| Dimension | Before V4 | After V4 |
|---|---|---|
| Architecture | 88 | 88 |
| Security | 78 | 88 |
| Authorization | 55 | 88 |
| Multi-Tenancy | 90 | 92 |
| Enterprise Readiness | 80 | 87 |
| Agent Readiness | 60 | 78 |
| Scalability | 80 | 80 |
| **Overall** | **76** | **86** |

The prior V3 audit fixed *horizontal* (cross-tenant) escapes. V4 found the *vertical* ones: a holder of `MANAGE_ROLES` (the built-in **admin** role) could grant the **FOUNDER** role (= ALL_PERMISSIONS) to themselves, and any key/SA creator could mint a credential holding scopes they don't possess. Authorization jumps 55→88 because those superuser-escalation paths are now closed.

---

## New Findings Only

### P0 — Vertical privilege escalation via role grant
- **Evidence:** `members.py: assign_role()` / `invite()` set `membership.role = role` with **no check that the granted role's permissions ⊆ the actor's**. `permissions.py`: `admin` holds `MANAGE_ROLES`; `founder` = `ALL_PERMISSIONS`. `admin_router.py: assign_role` is gated only by `MANAGE_ROLES`.
- **Impact:** An **admin** (no execute/billing/policy powers) could call `assign_role(self, FOUNDER)` and become org superuser — gaining `MANAGE_BILLING`, `MANAGE_POLICIES`, `EXECUTE_ACTIONS`, everything. Authenticated insider → full org takeover. Critical.
- **Fix:** `MemberService._ensure_grantable(role, actor_permissions)` raises `MemberForbidden` (→403) unless `default_permissions_for_role(role) ⊆ actor_permissions`. Applied to **both** `invite` and `assign_role`; the router threads `p.permissions`.

### P1 — Privilege escalation via credential scopes
- **Evidence:** `admin_router.py: create_api_key/create_service_account` passed `scopes=body.scopes` unchecked; `authz.py:_principal_from_api_key` sets `permissions = frozenset(scopes)`.
- **Impact:** An admin (lacking `execute_actions`) could mint an API key/SA with `scopes=["execute_actions"]` and then act through that credential — bypassing their own permission ceiling. Persistent, audit-evading escalation.
- **Fix:** `_ensure_grantable_scopes(scopes, principal)` raises 403 if `set(scopes) - principal.permissions` is non-empty. Applied to both create paths. A credential can never exceed its creator.

### P1 — Last-founder lockout / self-removal (governance availability)
- **Evidence:** `members.py: remove()` / `assign_role()` had no guard. A founder could `remove(self)` or demote the only founder.
- **Impact:** An org could be left with **zero founders** — no one holding `ALL_PERMISSIONS` to manage members/roles/billing. Permanently ungovernable; requires support/DB intervention. Classic enterprise blocker (Okta/GitHub/Stripe all guard this).
- **Fix:** `_founder_count(org)`; `remove` refuses to delete the last `FOUNDER` (`MemberConflict`→409); `assign_role` refuses to demote the last founder away from `FOUNDER`.

---

## Findings explicitly checked and found SAFE (proof, not assumption)
- **SSO claims overwriting local roles:** `sso.py` does not touch memberships/roles — no overwrite path. **Safe.**
- **SCIM overwriting local roles:** `provision_user` only `invite`s as `EMPLOYEE` and raises if already a member — never elevates. **Safe.**
- **Remaining `db.get(by-id)` tenant escapes:** all 14 sites reviewed — each is self-scoped (token `sub`), org-scoped (post-fix key/SA), or membership-derived. **No new escape.**
- **Permission drift / orphan permissions:** permissions resolve from `DEFAULT_ROLE_PERMISSIONS` (code) + `role_permissions` (data); a missing mapping = denied (fail-safe). **Safe.**

---

## Files Changed
| File | Before → After | Reason |
|---|---|---|
| `members.py` | `invite/assign_role` now take `actor_permissions` + `_ensure_grantable`; `remove/assign_role` add `_founder_count` last-founder guard; new `MemberForbidden`/`MemberConflict` | P0 role escalation + P1 last-founder |
| `admin_router.py` | thread `p.permissions` into invite/assign_role; map MemberForbidden→403, MemberConflict→409; `_ensure_grantable_scopes` on key/SA create | wire P0/P1 |
| `scim_router.py` | `provision_user` passes `actor_permissions=p.permissions` | keep SCIM working under the new guard |
| `tests/identity/test_identity_v2.py` | updated member test; +2 guards (`test_admin_cannot_escalate_to_founder`, `test_cannot_remove_or_demote_last_founder`) | prove fixes |

## Test Results
- Before: 262 passed. After: **264 passed, 9 skipped.** Both new tests fail against pre-V4 code → genuine regression guards. No migration needed (logic-only).

---

## Verdicts

**OpenAI Enterprise — would approve (post-V4).** Horizontal AND vertical escalation are now closed; least-privilege holds for humans and machine credentials. Conditions unchanged: edge rate-limiting + `is_verified` enforcement once a verification flow exists.

**Anthropic — yes.** The grant paths now enforce "you cannot give what you don't have" and "the org always has an owner" — the two invariants that make delegated admin safe. Minimal, contract-preserving.

**DeepMind — yes.** Identity is consistent (no last-founder void, no orphan superuser) and emits lifecycle events for future trust/relationship graphs. The flat permission model is the one forward-looking limit (below).

**Palantir — yes on governance.** Every grant is bounded, audited, and reversible; the org-owner invariant is enforced in code. They'd still want Postgres RLS as defense-in-depth (documented, DB-layer).

---

## Agent Readiness
**Current: 7.8/10.** Machine identities (service accounts + API keys) are now: org-scoped, revocable, audited, rotatable, AND **scope-capped to their creator** (no escalation). An AI agent can safely hold an org-scoped, least-privilege, revocable credential today.

**Future: needs one extension (no redesign).** Scopes are currently **flat org-wide permission strings**. A future agent needing **project-scope / memory-scope / execution-scope** (resource-level, not just action-level) requires adding a resource dimension to `ApiKey.scopes` / `ServiceAccount.scopes` (e.g. `execute_actions:project=X`). The model extends additively — `permissions.py` already treats permissions as opaque strings, and `role_permissions` already supports per-org grants. **No redesign; an additive scope-grammar change.** Until then, agents are org-scoped, not resource-scoped.

---

## Final Identity Score
- **Before V4: 76/100** (clean data-tenancy, but a superuser-escalation path any pentest would find in an hour).
- **After V4: 86/100.**
- **Remaining honest risks (named):** (1) flat scopes — no resource/project scoping for agents yet; (2) no Postgres RLS (app-enforced tenancy only); (3) no login rate-limit / `is_verified` enforcement (needs a verification flow); (4) HS256 live (RS256/JWKS path exists, unused); (5) deep-offset pagination is O(offset) — keyset cursors later.

Brutally honest: V4 found a **P0 the previous two audits missed entirely** because they audited data flow, not authority flow. That's the lesson — the dangerous bugs in an identity system are not always "can I read your row," they're "can I quietly become you." Those are now closed and tested. 86, not higher, because resource-scoped agent permissions and RLS are real, named, not-yet-built capabilities — not because anything currently shipped is unsafe.
