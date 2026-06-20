# Identity File Map (Phase 1)

| File | Purpose | Depends on | Called by | Risk |
|---|---|---|---|---|
| `models.py` | Core ORM: User, Organization, Team, Membership, Session, Integration, Role enum | platform.database | everything | **High** (tenancy anchor) |
| `enterprise_models.py` | RolePermission, ApiKey, ServiceAccount, RevokedToken, AuditLog, SsoConnection, ScimGroup | platform.database | authz, apikeys, sa, audit, sso, scim | **High** |
| `permissions.py` | Permission vocabulary + DEFAULT_ROLE_PERMISSIONS (data) | — (pure) | authz, members, admin_router | **High** (authorization source of truth) |
| `authz.py` | `Principal` + `get_principal` (JWT or API key) + `require_permission` | platform.security, models, enterprise_models, tokens, permissions | every gated route | **Critical** (the gate) |
| `dependencies.py` | `get_current_context` (legacy auth dep, revocation-enforced) | platform.security, models, tokens | identity/signals/brief routes | **High** |
| `tokens.py` | jti denylist + `tokens_valid_from` bulk-revoke | enterprise_models, models | authz, dependencies, members, scim | **High** |
| `service.py` | signup / authenticate / token issue+rotate / logout | org_service, audit, tokens, security | router | **High** |
| `audit.py` | append-only audit log + paginated reads (`since`) | enterprise_models | members, admin_router, scim | Medium |
| `members.py` | invite / remove / assign_role / list (+ **V4 escalation & last-founder guards**) | audit, models, permissions, tokens | admin_router, scim | **Critical** (privilege grants) |
| `apikeys.py` | hashed, scoped, **org-scoped-revoke** API keys | enterprise_models | authz, admin_router, service_accounts | **Critical** (credentials) |
| `service_accounts.py` | machine identities; **org-scoped** rotate/revoke | apikeys, enterprise_models | admin_router | **Critical** (machine creds) |
| `sso.py` | provider-agnostic SSO connection model + normalize (no role mapping yet) | enterprise_models | admin_router | Medium |
| `org_service.py` | Organization-lifecycle seam (extraction prep) | models | service | Low |
| `router.py` | /auth signup/login/refresh/logout/me | service, dependencies, security, events | edge | High |
| `admin_router.py` | delegated admin: members/keys/SAs/audit/sso (+ **V4 scope ceiling**) | authz, members, apikeys, sa, audit, sso, permissions, events | edge | **Critical** |
| `scim_router.py` | minimal SCIM Users/Groups (+ **multi-org-safe deprovision**) | authz, members, tokens, enterprise_models | edge | High |
| `migrations 0000/0013/0016` | base tables, enterprise tables, scale indexes | — | alembic | High |
| `tests/identity/*` | unit + isolation + escalation guards | fakes | CI | — |

**Dependency direction:** every file imports only `platform.*` + `navis.identity.*`. No business-domain import (Phase-1 clean, verified).
