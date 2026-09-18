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
    # Registering agents YOU OWN, and managing their credentials (P1-7).
    #
    # THE TWENTIETH PERMISSION, AND THE ONLY ONE P1-6 AND P1-7 ADDED BETWEEN
    # THEM. It exists because the vocabulary could not express the thing the
    # product needs, and the audit that proved that is worth restating: an
    # engineer at a customer needs to create an agent and a development
    # credential for their own work, and the ONLY permission that opened that
    # was MANAGE_SERVICE_ACCOUNTS — which also opens the organization's entire
    # agent roster, everybody else's credentials, and the four legacy
    # `/identity/service-accounts` handlers. So the only way to unblock a
    # developer was to make them an administrator of every machine identity in
    # the company, which is not a permission model, it is a shrug.
    #
    # WHAT MAKES IT DIFFERENT FROM MANAGE_SERVICE_ACCOUNTS IS A ROW PREDICATE,
    # NOT A DIFFERENT KIND OF POWER. A holder may create agents, and may read
    # and administer exactly those agents whose `service_accounts.created_by` is
    # their own subject id. `MANAGE_SERVICE_ACCOUNTS` is the same powers over
    # every agent in the organization, and it continues to mean exactly that —
    # nothing about it changed, so no existing credential's reach moved.
    #
    # THIS IS THE PATTERN EVERY COMPARABLE PLATFORM CONVERGED ON, checked
    # against current documentation rather than assumed. OpenAI separates
    # "Project API Keys — permission for a user to manage their own API keys"
    # (available to a project MEMBER) from "Service Accounts — view and manage
    # project service accounts" (project OWNER only). Anthropic's `Developer`
    # role "can manage API keys" while only `Admin` manages users and the
    # organization. Glean ships "can create and publish agents" ON by default
    # for an ordinary member, and requires Agent Moderator to touch anybody
    # else's. Three independent products, one line in the same place.
    MANAGE_OWN_AGENTS = "manage_own_agents"
    # Administering WHO MAY REACH THE DEVELOPER SURFACE — the Developer team
    # (P1-8).
    #
    # THE TWENTY-FIRST PERMISSION, AND THE AUDIT THAT PROVED IT NECESSARY.
    # Every route that administers Developer team membership was gated on an
    # APPLICATION permission: `GET /identity/members` and
    # `POST /identity/members/invite` on MANAGE_MEMBERS,
    # `POST /identity/members/{id}/access` on MANAGE_ROLES. No Developer role
    # holds either. So a Developer owner or admin who is an Application
    # `employee` could not list their own team, invite an engineer, or grant
    # Developer access — they had to ask an Application administrator. That is
    # precisely the coupling the surface split exists to remove, and it was
    # invisible because the person who SIGNS UP is `role=FOUNDER` and therefore
    # holds both Application permissions by accident of being first.
    #
    # WHY NOT REUSE A PERMISSION THAT ALREADY EXISTS — all three candidates were
    # checked and each is wrong for a different reason:
    #
    #   MANAGE_SERVICE_ACCOUNTS is AGENT-ASSIGNABLE, and a live agent key
    #   carries it. Gating human role administration on it would let a machine
    #   credential grant Developer access to people. That is not a widening,
    #   it is an escalation path.
    #
    #   MANAGE_ROLES and MANAGE_MEMBERS are Application-classified. Granting
    #   either to a Developer role would hand Developer users Application
    #   member administration and the ability to assign Application roles —
    #   the exact inheritance P1-6 removed.
    #
    #   `organizations.kind` is not an authorization input and must never
    #   become one (`test_workspace_kind_is_not_an_authorization_input`).
    #
    # DELIBERATELY NOT AGENT-ASSIGNABLE. It is the one Developer permission a
    # machine must never hold: agents administer nothing about people. No route
    # under `navis/developer/` gates on it, so
    # `test_agent_assignable_matches_developer_routes` is satisfied without an
    # exception — the invariant stays a plain subset check.
    #
    # SCOPED TO THE DEVELOPER SURFACE BY THE ROUTES, NOT BY A CLAIM. It reaches
    # three identity-administration routes as an ALTERNATIVE to the Application
    # permission already there, and `_ensure_grantable` still requires its
    # holder to already hold anything they hand out — so a Developer admin can
    # confer Developer access and cannot confer Application authority they do
    # not have.
    #
    # THE PATTERN EVERY COMPARABLE PLATFORM CONVERGED ON, checked against
    # current documentation. Portkey's workspace `Manager` may "invite
    # organization members to the workspace" and "assign workspace roles"
    # without being an organization admin. Zep scopes RBAC at both account and
    # project level and states plainly that RBAC governs humans while key
    # policies govern agents. OpenAI separates organization owner/reader from
    # project owner/member. Three products, one boundary: the workspace
    # administrator administers the workspace's people.
    MANAGE_DEVELOPER_ACCESS = "manage_developer_access"
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


# ── THE SURFACE PARTITION (P1-6) ────────────────────────────────────────────
#
# WHY THIS IS A PARTITION OF THE EXISTING VOCABULARY AND NOT A NEW ONE.
# Application and Developer are two product surfaces over one organization, one
# authorization evaluator and one shared Core. The question "may this member use
# the Developer surface" therefore has to be answered with the permissions that
# already exist, because inventing a parallel vocabulary is inventing a second
# authorization system — which is the one thing the surface split may not do.
#
# It turned out not to need inventing. Every permission below is consumed
# EXCLUSIVELY by routes under `navis/developer/`; no Application route gates on
# any of them (`test_developer_permissions_gate_no_application_route` asserts
# this against the live route table, so it cannot rot). That is what makes the
# split safe: removing these from an Application role's defaults cannot change
# what an Application route does, because no Application route ever asked.
#
# MANAGE_SERVICE_ACCOUNTS IS THE ONE PERMISSION THAT ALSO GATES APPLICATION
# ROUTES — the four legacy `/identity/service-accounts` handlers. It is still
# classified here, because registering a machine identity IS Developer authority
# whichever door it is done through, and `0051_surface_rbac_columns` named
# exactly this as the bug the split exists to fix ("an Application administrator
# silently holds `manage_service_accounts`"). The consequence is deliberate and
# recorded in the P1-6 report: those four routes now require a Developer role.
DEVELOPER_PERMISSIONS: frozenset[str] = frozenset({
    Permission.VIEW_STATE,
    Permission.VIEW_MEMORY,
    Permission.VIEW_REASONING,
    Permission.VIEW_DECISIONS,
    Permission.VIEW_OBSERVATION,
    Permission.VIEW_LEARNING,
    Permission.VIEW_INTELLIGENCE,
    Permission.RECORD_AGENT_RUNS,
    Permission.MANAGE_SERVICE_ACCOUNTS,
    Permission.MANAGE_OWN_AGENTS,
    # Developer-classified so `_application_only` strips it from every
    # Application role. An Application founder still administers Developer
    # access — through MANAGE_ROLES, which the routes accept alongside this —
    # so nothing they could do before is lost; what they do NOT get is a
    # Developer-surface permission by virtue of an Application role, which is
    # the whole P1-6 rule.
    Permission.MANAGE_DEVELOPER_ACCESS,
})

#: Everything else. Derived, never hand-listed, so a new permission lands on
#: exactly one side of the line and a reviewer can see which.
APPLICATION_PERMISSIONS: frozenset[str] = ALL_PERMISSIONS - DEVELOPER_PERMISSIONS


def _application_only(permissions: frozenset[str]) -> frozenset[str]:
    """Strip Developer authority from an Application role's default set.

    APPLIED TO THE ORIGINAL LITERALS RATHER THAN REPLACING THEM, on purpose.
    The role sets below are the ones this file has always declared; this
    function is the whole of the P1-6 change to them, so the diff shows that
    nothing but Developer permissions was removed and no Application permission
    was touched. Writing six new literals instead would make that a claim a
    reviewer has to verify by hand.
    """
    return permissions - DEVELOPER_PERMISSIONS

# Founder's powers are ENUMERATED explicitly, NOT `ALL_PERMISSIONS`. A wildcard would
# auto-grant any future permission (e.g. a later DELETE_ALL_MEMORY) to founders with
# no review — the classic RBAC wildcard anti-pattern. Every new permission must be a
# DELIBERATE choice: added to FOUNDER_PERMISSIONS or FOUNDER_EXCLUDED. A CI test
# (test_every_permission_has_a_founder_decision) fails until that choice is made.
#
# SINCE P1-6 THIS IS THE FOUNDER'S TOTAL ACROSS BOTH SURFACES, not the contents of
# one role map. A founder reaches it as `DEFAULT_ROLE_PERMISSIONS['founder']`
# (the Application half) plus `DEVELOPER_ROLE_PERMISSIONS['owner']` (the Developer
# half, from `memberships.developer_role`), and the two halves are disjoint by
# construction. The set is kept whole here because the review rule it encodes —
# "every permission is a deliberate founder decision" — is about the vocabulary,
# not about which column happens to carry it.
FOUNDER_PERMISSIONS: frozenset[str] = frozenset({
    Permission.VIEW_SIGNALS, Permission.VIEW_MEMORY, Permission.VIEW_STATE,
    Permission.VIEW_REASONING, Permission.VIEW_DECISIONS, Permission.VIEW_INTELLIGENCE,
    Permission.VIEW_OBSERVATION, Permission.VIEW_LEARNING,
    Permission.VIEW_AUDIT,
    Permission.EXECUTE_ACTIONS, Permission.MANAGE_INTEGRATIONS, Permission.MANAGE_POLICIES,
    Permission.MANAGE_MEMBERS, Permission.MANAGE_ROLES, Permission.MANAGE_API_KEYS,
    Permission.MANAGE_SERVICE_ACCOUNTS, Permission.MANAGE_SSO, Permission.MANAGE_BILLING,
    Permission.RECORD_AGENT_RUNS, Permission.MANAGE_OWN_AGENTS,
    # Included, and `_application_only` strips it straight back out — which is
    # the point. This literal is the "founder is the owner of everything"
    # statement the review rule below checks; the Application/Developer split
    # is applied to it afterwards. A founder reaches Developer team
    # administration through `developer_role='owner'`, which every workspace
    # creator is given, so founder+owner still resolves to the whole
    # vocabulary — see `test_a_founder_keeps_exactly_the_permissions_they_had_
    # before_the_split`.
    Permission.MANAGE_DEVELOPER_ACCESS,
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

# THE APPLICATION SURFACE'S role -> permission map. `role_permissions` rows (per
# org) augment these at runtime.
#
# EVERY ENTRY IS WRAPPED IN `_application_only`, and that is the entire P1-6
# change here. The literals are untouched; what changes is that an Application
# role can no longer carry Developer authority, which is what made `admin` able
# to register agents and `ceo`/`coo`/`chief_of_staff` able to read the whole
# Developer data surface without anyone granting them Developer access.
# `test_no_application_role_grants_developer_authority` makes the coupling
# unrepresentable rather than merely absent.
DEFAULT_ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "founder": _application_only(FOUNDER_PERMISSIONS),
    "ceo": _application_only(_VIEW | {
        Permission.EXECUTE_ACTIONS, Permission.MANAGE_MEMBERS,
        Permission.MANAGE_POLICIES, Permission.MANAGE_INTEGRATIONS, Permission.VIEW_AUDIT,
    }),
    "coo": _application_only(_VIEW | {
        Permission.EXECUTE_ACTIONS, Permission.MANAGE_INTEGRATIONS, Permission.MANAGE_MEMBERS,
    }),
    "chief_of_staff": _application_only(
        _VIEW | {Permission.EXECUTE_ACTIONS, Permission.MANAGE_MEMBERS}
    ),
    # IT/admin: delegated administration, but not business operations by default.
    "admin": _application_only(_VIEW | {
        Permission.MANAGE_MEMBERS, Permission.MANAGE_ROLES, Permission.MANAGE_INTEGRATIONS,
        Permission.MANAGE_API_KEYS, Permission.MANAGE_SERVICE_ACCOUNTS, Permission.MANAGE_SSO,
        Permission.VIEW_AUDIT, Permission.RECORD_AGENT_RUNS,
    }),
    "employee": _application_only(
        frozenset({Permission.VIEW_DECISIONS, Permission.VIEW_SIGNALS})
    ),
}


# ── THE DEVELOPER SURFACE'S role -> permission map (P1-6) ───────────────────
#
# FOUR ROLES, BECAUSE THE VOCABULARY IS ALREADY FIXED IN THE DATABASE.
# `0051_surface_rbac_columns` shipped `ck_memberships_developer_role`, which
# permits exactly 'admin', 'developer', 'owner', 'viewer'. This map is what
# makes those four values mean something; nothing here widens or narrows the
# constraint, and `test_developer_role_map_matches_the_database_vocabulary`
# fails if the two ever disagree.
#
# STRICTLY NESTED — viewer < developer < admin = owner — so a more senior
# Developer role can never hold LESS than a junior one. `_ensure_grantable`
# compares permission sets, so a non-nested ladder would make "may I grant this
# role" depend on which incomparable pair you happened to pick.
#
# OWNER AND ADMIN ARE PERMISSION-IDENTICAL, DELIBERATELY. The difference between
# them is GOVERNANCE, not privilege: an organization that has a Developer owner
# must keep at least one (`MemberService._count_developer_owners_locked`),
# exactly as it must keep at least one founder. Giving `owner` an extra
# permission to justify the name would mean inventing a permission for a rule
# that is already expressible, and the P1-6 brief forbids duplicate permissions.
_DEVELOPER_READ: frozenset[str] = frozenset({
    Permission.VIEW_STATE, Permission.VIEW_MEMORY, Permission.VIEW_REASONING,
    Permission.VIEW_DECISIONS, Permission.VIEW_OBSERVATION, Permission.VIEW_LEARNING,
    Permission.VIEW_INTELLIGENCE,
})

DEVELOPER_ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    # Read the organization's context, state, memory, reasoning, decisions,
    # outcomes, learning and intelligence over the Developer API. Cannot record
    # a run, and cannot see or manage an agent or a credential.
    #
    # DELIBERATELY NOT SELF-SERVE. A viewer is somebody who reads what the
    # organization knows — a data scientist, an analyst, a reviewer. They are
    # not building on Navis, so an agent and a credential are not what they
    # came for, and handing them the ability to mint machine identities would
    # make "viewer" the wrong word for the role.
    "viewer": _DEVELOPER_READ,
    # THE WORKING ROLE, AND THE WHOLE POINT OF P1-7. Everything a viewer reads,
    # plus writing the execution ledger, plus the ability to register agents of
    # their OWN and manage their credentials.
    #
    # WHY SELF-SERVE IS THE DEFAULT HERE AND NOT AN OPT-IN. Before P1-7 this
    # role could not obtain a credential of any kind: `manage_service_accounts`
    # belonged to Developer admin and `manage_api_keys` to the Application
    # administrator, so an engineer at a customer had to ask an organization
    # admin to mint them a key — every time, including for a rotation. The
    # product on the other side of that request is an SDK, an MCP server and a
    # local coding agent, all of which a developer is expected to set up in
    # minutes. A permission model that turns "I want to try this" into a ticket
    # is a permission model that gets bypassed with a shared credential.
    #
    # It is still least privilege, and the bound is a row predicate rather than
    # a promise: `MANAGE_OWN_AGENTS` reaches only agents whose `created_by` is
    # this person, and `ensure_grantable_scopes` caps whatever they mint at the
    # permissions they themselves hold. They cannot see, edit or revoke a
    # colleague's agent, cannot reach the legacy `/identity/service-accounts`
    # administration, and hold no Application authority at all.
    "developer": _DEVELOPER_READ | {
        Permission.RECORD_AGENT_RUNS, Permission.MANAGE_OWN_AGENTS,
    },
    # Adds the ORGANIZATION-WIDE agent roster and the agent/credential lifecycle
    # for every agent, not just their own. `MANAGE_OWN_AGENTS` is kept alongside
    # it so the ladder stays strictly nested — an admin must never hold less
    # than a developer — and so a route may gate on the narrower permission
    # without an admin falling through it.
    "admin": _DEVELOPER_READ | {
        Permission.RECORD_AGENT_RUNS,
        Permission.MANAGE_OWN_AGENTS,
        Permission.MANAGE_SERVICE_ACCOUNTS,
        # P1-8: administering the Developer TEAM, not just its machines. An
        # admin who can manage every agent in the organization but cannot add
        # the engineer who needs one is an administrator of half a surface.
        Permission.MANAGE_DEVELOPER_ACCESS,
    },
    # Same PERMISSIONS as admin; the difference is governance, and since P1-8
    # there is a second governance rule as well as the last-owner one:
    # `MemberService.assign_developer_role` lets only an OWNER confer or remove
    # the owner role. That rule cannot be expressed as a permission here —
    # these two sets must stay equal or `_ensure_grantable`'s subset
    # comparison would make "may I grant this role" depend on an ordering the
    # vocabulary does not define — so succession lives with the other
    # governance guards, next to `_count_developer_owners_locked`.
    "owner": _DEVELOPER_READ | {
        Permission.RECORD_AGENT_RUNS,
        Permission.MANAGE_OWN_AGENTS,
        Permission.MANAGE_SERVICE_ACCOUNTS,
        Permission.MANAGE_DEVELOPER_ACCESS,
    },
}

#: Namespace prefix for per-org Developer-role grants in `role_permissions`.
#:
#: REQUIRED, NOT COSMETIC. `role_permissions.role` is one `String(40)` column and
#: the three role vocabularies in this system share three strings — 'admin' is a
#: legacy `Role`, an `ApplicationRole` AND a `DeveloperRole`, and 'owner'/'viewer'
#: are shared by the latter two. An un-namespaced grant to "admin" would apply to
#: all of them indistinguishably, which is a silent cross-surface privilege leak
#: through a data row. Prefixing keeps one column serving both surfaces without a
#: migration and without ambiguity.
DEVELOPER_GRANT_PREFIX = "developer:"


def developer_grant_key(developer_role: str | None) -> str:
    """The `role_permissions.role` value that carries grants for `developer_role`.

    Returns the empty string for no Developer role, which matches no row — the
    same "resolves to nothing" behaviour `grants_for` already has for an unknown
    Application role.
    """
    return f"{DEVELOPER_GRANT_PREFIX}{developer_role}" if developer_role else ""


def validate_permission(permission: str) -> None:
    """Reject typo/invalid permission strings before they're persisted (grants/scopes).
    Keeps `role_permissions` rows and API-key/service-account scopes free of garbage."""
    if permission not in VALID_PERMISSIONS:
        raise ValueError(f"unknown permission: {permission!r}")


def default_permissions_for_role(role: str | None) -> frozenset[str]:
    return DEFAULT_ROLE_PERMISSIONS.get(role or "", frozenset())


def default_developer_permissions_for_role(developer_role: str | None) -> frozenset[str]:
    """The Developer surface's defaults for `developer_role`.

    The empty set for `None`, and that is the load-bearing value: NULL in
    `memberships.developer_role` means "no access to the Developer surface",
    which is what an Application-only member looks like. An unrecognised value
    also resolves to the empty set — fail closed, never guess.
    """
    return DEVELOPER_ROLE_PERMISSIONS.get(developer_role or "", frozenset())


def effective_permissions(
    role: str | None,
    grants: list[str] | None = None,
    *,
    developer_role: str | None = None,
    developer_grants: list[str] | None = None,
) -> frozenset[str]:
    """THE ONE authorization evaluator, for both surfaces.

    Application defaults for `role`, plus the Developer defaults for
    `developer_role`, each augmented by its own per-org grants. The union is a
    single flat permission set, which is why `require_permission` and every
    route gate are untouched by the surface split: they keep asking one question
    ("does this principal hold X") of one set.

    THE TWO SURFACES ARE ADDITIVE AND NEVER SUBTRACTIVE. Holding a Developer
    role cannot take away an Application permission and vice versa — the same
    property Anthropic's org-baseline/workspace-additive model and Zep's
    combinable account+project assignments have, and the reason a member can be
    an Application `ceo` and a Developer `viewer` without either grant
    surprising the other.

    THE NEW PARAMETERS ARE KEYWORD-ONLY so that every existing positional caller
    — `_principal_from_jwt`, `MemberService`, the characterization tests — keeps
    its exact previous meaning, and an Application-only resolution stays
    literally the expression it always was.
    """
    return (
        default_permissions_for_role(role)
        | frozenset(grants or [])
        | default_developer_permissions_for_role(developer_role)
        | frozenset(developer_grants or [])
    )


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
                       "Register, re-scope and revoke ANY agent in the "
                       "organization.", sensitive=True),
        PermissionInfo(Permission.MANAGE_OWN_AGENTS, PermissionKind.ADMIN,
                       "Register agents of your own and manage their "
                       "credentials. Does not reach anybody else's.",
                       sensitive=True),
        PermissionInfo(Permission.MANAGE_DEVELOPER_ACCESS, PermissionKind.ADMIN,
                       "See the organization's members, invite people, and "
                       "grant or revoke their Developer access. Does not "
                       "confer any Application authority.", sensitive=True),
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
    # INERT ON A MACHINE CREDENTIAL, AND LISTED ANYWAY. Every route it opens is
    # either a write (refused to a machine principal by `_operator`) or a read
    # scoped to `created_by == subject_id` — and `created_by` holds the USER who
    # registered the agent, so an agent presenting its own id matches no row and
    # reads an empty list. Granting it to an agent therefore changes nothing.
    #
    # It is here because `test_agent_assignable_matches_developer_routes` checks
    # that every permission a Developer route names is one the console can
    # grant, and that invariant is worth more as a plain subset check than as
    # one carrying an exception for human-only routes. The honest cost is a
    # checkbox that does nothing for a machine; the alternative is a CI rule
    # nobody can read.
    Permission.MANAGE_OWN_AGENTS,
})


#: Permissions NO machine credential may ever hold, enforced as a SERVER rule
#: by `authz.ensure_grantable_scopes` (P1-8).
#:
#: DISTINCT FROM `AGENT_ASSIGNABLE_PERMISSIONS` ABOVE, AND THE DIFFERENCE IS
#: THE WHOLE POINT. That set is a console DEFAULT — the checkboxes a
#: management screen offers — and its own docstring says the API still accepts
#: any valid permission the caller themselves holds. So omission from it is a
#: UI decision, not a boundary. This set is the boundary: a scope named here
#: is refused at credential-minting time no matter who is asking, including a
#: Developer owner who legitimately holds it.
#:
#: WHY THIS ONE PERMISSION. `manage_developer_access` decides who is on the
#: Developer team and what access they hold — it administers PEOPLE. Every
#: other Developer permission acts on data or on machine identities, which is
#: what an agent is for. A credential that could grant humans Developer access
#: would let anyone holding that key escalate a person (including themselves)
#: into the organization, which is an escape from the agent sandbox rather
#: than a capability within it.
#:
#: Verified live: a Developer owner attempting to mint an agent carrying this
#: scope is refused 403 rather than issued a credential.
HUMAN_ONLY_PERMISSIONS: frozenset[str] = frozenset({
    Permission.MANAGE_DEVELOPER_ACCESS,
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
