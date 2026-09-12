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
    # Observation and Learning are pipeline stages exactly as the six above are,
    # and each already has its own `Capability`. They had no VIEW_* permission
    # only because they had no read surface when this vocabulary was written —
    # `learning.router` records that nothing exposed Learning over HTTP at all.
    # When those readers shipped they could gate on the capability alone, and the
    # Developer boundary then had to borrow `VIEW_DECISIONS` for both.
    #
    # Borrowing was the wrong long-term contract for a reason that is about
    # blast radius, not tidiness: `VIEW_DECISIONS` is one of the two permissions
    # `employee` holds, so making it the key to the outcome log (provider
    # references, failure reasons) and to Learning's org-wide success rates would
    # widen what the least-privileged role means — a decision taken by an edge
    # rather than by the permission model. These two make the vocabulary say what
    # the pipeline already says.
    VIEW_OBSERVATION = "view_observation"
    VIEW_LEARNING = "view_learning"
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
    # Writing to the agent execution ledger — tasks, runs, steps, outcomes.
    #
    # A WRITE permission of its own, and the reasoning is the same shape as the
    # note above on VIEW_OBSERVATION. Reads of the ledger are gated on
    # VIEW_OBSERVATION, which is a read permission and must not authorize a
    # write; and MANAGE_SERVICE_ACCOUNTS is the wrong gate in the other
    # direction, because it would mean only an agent that can MINT other agents
    # may report what it did. An agent recording its own history needs neither
    # of those powers, so it gets a permission that grants exactly this.
    RECORD_AGENT_RUNS = "record_agent_runs"
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
    Permission.VIEW_OBSERVATION, Permission.VIEW_LEARNING,
    Permission.VIEW_AUDIT,
    Permission.EXECUTE_ACTIONS, Permission.MANAGE_INTEGRATIONS, Permission.MANAGE_POLICIES,
    Permission.MANAGE_MEMBERS, Permission.MANAGE_ROLES, Permission.MANAGE_API_KEYS,
    Permission.MANAGE_SERVICE_ACCOUNTS, Permission.MANAGE_SSO, Permission.MANAGE_BILLING,
    Permission.RECORD_AGENT_RUNS,
})
# Permissions deliberately WITHHELD from founder (empty today — founder is the owner).
FOUNDER_EXCLUDED: frozenset[str] = frozenset()

# The read bundle the leadership + admin roles share. VIEW_OBSERVATION and
# VIEW_LEARNING join it so those roles KEEP the reach they have today: both
# `/outcomes` and `/learning/summary` are capability-gated only, so every role
# already reads them, and a new permission that did not follow into this bundle
# would be a silent removal dressed up as a vocabulary fix. `employee` is
# deliberately NOT given them — see the note on the permissions themselves.
_VIEW = frozenset({
    Permission.VIEW_SIGNALS, Permission.VIEW_MEMORY, Permission.VIEW_STATE,
    Permission.VIEW_REASONING, Permission.VIEW_DECISIONS, Permission.VIEW_INTELLIGENCE,
    Permission.VIEW_OBSERVATION, Permission.VIEW_LEARNING,
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
        Permission.VIEW_AUDIT, Permission.RECORD_AGENT_RUNS,
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


# ── The catalog: what each permission MEANS ─────────────────────────────────
#
# DESCRIPTIVE DATA ABOUT THE VOCABULARY ABOVE, AND NOTHING MORE. Nothing here
# is consulted by a gate. `require_permission` reads `principal.permissions`
# and `ensure_grantable_scopes` reads the caller's own set; both are unchanged
# by anything in this section, and a permission absent from the catalog is
# still perfectly grantable through the API. Making the catalog authoritative
# would put a second authorization axis next to `Permission` — the exact
# failure the Developer boundary exists to prevent.
#
# IT EXISTS BECAUSE THE CONSOLE WAS MAINTAINING ITS OWN COPY. The Developer
# Console shipped a hand-written list of four permissions in TypeScript
# (`DEVELOPER_AGENT_PERMISSIONS`) while the Developer boundary had grown to
# gate on nine. The drift was invisible — a permission the API accepted could
# not be selected in the UI that exists to grant it, and `record_agent_runs`
# was ungrantable through the console for exactly that reason. One catalog,
# served from the module that defines the vocabulary, is what stops a
# management surface from describing a system that no longer exists.


class PermissionKind:
    """What holding a permission lets a principal DO.

    `read` and `write` are about the DATA a permission opens. `admin` is
    about the ORGANIZATION itself — members, roles, credentials, billing —
    and is called out separately because "can read your outcomes" and "can
    mint credentials" are not the same kind of grant, however similar two
    checkboxes next to each other look.
    """

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class PermissionInfo:
    """One catalog entry. Plain object, matching this module's no-dependency rule."""

    __slots__ = ("permission", "kind", "summary", "sensitive")

    def __init__(self, permission: str, kind: str, summary: str, *, sensitive: bool = False):
        self.permission = permission
        self.kind = kind
        self.summary = summary
        #: Worth a second look before granting — it writes, or it can reach
        #: the organization's own identity and money. Advisory: the console
        #: marks these, and no gate consults the flag.
        self.sensitive = sensitive


PERMISSION_CATALOG: dict[str, PermissionInfo] = {
    info.permission: info
    for info in (
        PermissionInfo(Permission.VIEW_SIGNALS, PermissionKind.READ,
                       "Read raw signals ingested from connected sources."),
        PermissionInfo(Permission.VIEW_MEMORY, PermissionKind.READ,
                       "Read organizational memory — what Navis has retained."),
        PermissionInfo(Permission.VIEW_STATE, PermissionKind.READ,
                       "Read context, State and the entity graph."),
        PermissionInfo(Permission.VIEW_REASONING, PermissionKind.READ,
                       "Read the reasoning log — how conclusions were reached."),
        PermissionInfo(Permission.VIEW_DECISIONS, PermissionKind.READ,
                       "Read recorded decisions."),
        PermissionInfo(Permission.VIEW_OBSERVATION, PermissionKind.READ,
                       "Read outcomes and the agent execution ledger."),
        PermissionInfo(Permission.VIEW_LEARNING, PermissionKind.READ,
                       "Read what Navis has learned across the organization."),
        PermissionInfo(Permission.VIEW_INTELLIGENCE, PermissionKind.READ,
                       "Read derived intelligence and adaptive projections."),
        PermissionInfo(Permission.VIEW_AUDIT, PermissionKind.READ,
                       "Read the audit log.", sensitive=True),
        PermissionInfo(Permission.RECORD_AGENT_RUNS, PermissionKind.WRITE,
                       "Record agent tasks, runs, steps and outcomes to the "
                       "execution ledger.", sensitive=True),
        PermissionInfo(Permission.EXECUTE_ACTIONS, PermissionKind.WRITE,
                       "Take actions in connected systems on the organization's "
                       "behalf.", sensitive=True),
        PermissionInfo(Permission.MANAGE_INTEGRATIONS, PermissionKind.ADMIN,
                       "Connect and disconnect source systems.", sensitive=True),
        PermissionInfo(Permission.MANAGE_POLICIES, PermissionKind.ADMIN,
                       "Change organizational policy.", sensitive=True),
        PermissionInfo(Permission.MANAGE_MEMBERS, PermissionKind.ADMIN,
                       "Invite, edit and remove members.", sensitive=True),
        PermissionInfo(Permission.MANAGE_ROLES, PermissionKind.ADMIN,
                       "Change what each role may do.", sensitive=True),
        PermissionInfo(Permission.MANAGE_API_KEYS, PermissionKind.ADMIN,
                       "Mint and revoke API keys.", sensitive=True),
        PermissionInfo(Permission.MANAGE_SERVICE_ACCOUNTS, PermissionKind.ADMIN,
                       "Register, re-scope and revoke agents.", sensitive=True),
        PermissionInfo(Permission.MANAGE_SSO, PermissionKind.ADMIN,
                       "Configure SSO and identity providers.", sensitive=True),
        PermissionInfo(Permission.MANAGE_BILLING, PermissionKind.ADMIN,
                       "Change the plan and payment details.", sensitive=True),
    )
}

#: Permissions the DEVELOPER BOUNDARY actually consumes — the set that has any
#: effect on an agent, derived from the routes under `navis/developer/`.
#:
#: A DEFAULT FOR A MANAGEMENT SURFACE, NOT A SERVER RULE. The API still accepts
#: any valid permission the caller themselves holds; this is the list the
#: console offers, and offering a grant that changes nothing is how a
#: permission screen stops describing reality. `test_agent_assignable_matches_
#: developer_routes` fails when a Developer route starts gating on a permission
#: that is not here, so the drift that produced this file cannot recur silently.
AGENT_ASSIGNABLE_PERMISSIONS: frozenset[str] = frozenset({
    Permission.VIEW_STATE,
    Permission.VIEW_MEMORY,
    Permission.VIEW_OBSERVATION,
    Permission.VIEW_LEARNING,
    Permission.VIEW_REASONING,
    Permission.VIEW_DECISIONS,
    Permission.VIEW_INTELLIGENCE,
    Permission.RECORD_AGENT_RUNS,
    # Opens the agent ROSTER to a machine caller (list/read only — every agent
    # lifecycle WRITE is refused to a machine principal by `_operator`). An
    # operator tool that manages a fleet needs it; most agents do not.
    Permission.MANAGE_SERVICE_ACCOUNTS,
})


def permission_info(permission: str) -> PermissionInfo:
    """Catalog entry for `permission`, or a neutral one for a permission the
    catalog has not described yet.

    NEVER RAISES. A vocabulary entry with no catalog row is a documentation
    gap, and a management screen that 500s because somebody added a permission
    without a summary is worse than one that renders the bare string.
    """
    described = PERMISSION_CATALOG.get(permission)
    if described is not None:
        return described
    return PermissionInfo(permission, PermissionKind.READ, "", sensitive=False)
