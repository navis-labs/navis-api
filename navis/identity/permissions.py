"""The permission vocabulary + the default role→permission mapping (as DATA).

Authorization is permission-based, never role-string-based: routes gate on a
Permission, and a principal's effective permissions are resolved from this default
map plus any per-org `role_permissions` grants. Adding/altering a role's powers is
a data change (a row), not a code/enum migration.

Pure module — no DB, no imports from other domains.
"""
from __future__ import annotations


class Permission:
    # Reads
    VIEW_SIGNALS = "view_signals"
    VIEW_MEMORY = "view_memory"
    VIEW_STATE = "view_state"
    VIEW_REASONING = "view_reasoning"
    VIEW_DECISIONS = "view_decisions"
    VIEW_INTELLIGENCE = "view_intelligence"
    VIEW_AUDIT = "view_audit"
    # Operations
    EXECUTE_ACTIONS = "execute_actions"
    MANAGE_INTEGRATIONS = "manage_integrations"
    MANAGE_POLICIES = "manage_policies"
    # Administration (delegated)
    MANAGE_MEMBERS = "manage_members"
    MANAGE_ROLES = "manage_roles"
    MANAGE_API_KEYS = "manage_api_keys"
    MANAGE_SERVICE_ACCOUNTS = "manage_service_accounts"
    MANAGE_SSO = "manage_sso"  # SSO/IdP config — distinct from role management
    MANAGE_BILLING = "manage_billing"  # owned by Organization later; vocab lives here


ALL_PERMISSIONS: frozenset[str] = frozenset(
    v for k, v in vars(Permission).items() if not k.startswith("_") and isinstance(v, str)
)
# Canonical vocabulary for validation (a persisted grant/scope must be one of these).
VALID_PERMISSIONS: frozenset[str] = ALL_PERMISSIONS

# Founder's powers are ENUMERATED explicitly, NOT `ALL_PERMISSIONS`. A wildcard would
# auto-grant any future permission (e.g. a later DELETE_ALL_MEMORY) to founders with
# no review — the classic RBAC wildcard anti-pattern. Every new permission must be a
# DELIBERATE choice: added to FOUNDER_PERMISSIONS or FOUNDER_EXCLUDED. A CI test
# (test_every_permission_has_a_founder_decision) fails until that choice is made.
FOUNDER_PERMISSIONS: frozenset[str] = frozenset({
    Permission.VIEW_SIGNALS, Permission.VIEW_MEMORY, Permission.VIEW_STATE,
    Permission.VIEW_REASONING, Permission.VIEW_DECISIONS, Permission.VIEW_INTELLIGENCE,
    Permission.VIEW_AUDIT,
    Permission.EXECUTE_ACTIONS, Permission.MANAGE_INTEGRATIONS, Permission.MANAGE_POLICIES,
    Permission.MANAGE_MEMBERS, Permission.MANAGE_ROLES, Permission.MANAGE_API_KEYS,
    Permission.MANAGE_SERVICE_ACCOUNTS, Permission.MANAGE_SSO, Permission.MANAGE_BILLING,
})
# Permissions deliberately WITHHELD from founder (empty today — founder is the owner).
FOUNDER_EXCLUDED: frozenset[str] = frozenset()

_VIEW = frozenset({
    Permission.VIEW_SIGNALS, Permission.VIEW_MEMORY, Permission.VIEW_STATE,
    Permission.VIEW_REASONING, Permission.VIEW_DECISIONS, Permission.VIEW_INTELLIGENCE,
})

# Default mapping. `role_permissions` rows (per org) augment these at runtime.
DEFAULT_ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "founder": FOUNDER_PERMISSIONS,
    "ceo": _VIEW | {
        Permission.EXECUTE_ACTIONS, Permission.MANAGE_MEMBERS,
        Permission.MANAGE_POLICIES, Permission.MANAGE_INTEGRATIONS, Permission.VIEW_AUDIT,
    },
    "coo": _VIEW | {
        Permission.EXECUTE_ACTIONS, Permission.MANAGE_INTEGRATIONS, Permission.MANAGE_MEMBERS,
    },
    "chief_of_staff": _VIEW | {Permission.EXECUTE_ACTIONS, Permission.MANAGE_MEMBERS},
    # IT/admin: delegated administration, but not business operations by default.
    "admin": _VIEW | {
        Permission.MANAGE_MEMBERS, Permission.MANAGE_ROLES, Permission.MANAGE_INTEGRATIONS,
        Permission.MANAGE_API_KEYS, Permission.MANAGE_SERVICE_ACCOUNTS, Permission.MANAGE_SSO,
        Permission.VIEW_AUDIT,
    },
    "employee": frozenset({Permission.VIEW_DECISIONS, Permission.VIEW_SIGNALS}),
}


def validate_permission(permission: str) -> None:
    """Reject typo/invalid permission strings before they're persisted (grants/scopes).
    Keeps `role_permissions` rows and API-key/service-account scopes free of garbage."""
    if permission not in VALID_PERMISSIONS:
        raise ValueError(f"unknown permission: {permission!r}")


def default_permissions_for_role(role: str | None) -> frozenset[str]:
    return DEFAULT_ROLE_PERMISSIONS.get(role or "", frozenset())


def effective_permissions(role: str | None, grants: list[str] | None = None) -> frozenset[str]:
    """Defaults for the role, augmented by per-org grants (data overrides)."""
    return default_permissions_for_role(role) | frozenset(grants or [])
