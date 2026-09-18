"""CHARACTERIZATION of effective permissions on both surfaces.

WHAT A CHARACTERIZATION TEST IS, AND WHY THIS ONE LOOKS ODD.

These tests do not argue that the current behaviour is desirable. They exist to
make it IMPOSSIBLE TO CHANGE BY ACCIDENT, because a permission table is exactly
the kind of thing that gets "tidied" in passing.

So a failure here means one of two things, and telling them apart is the whole
point:

  * an intended change to the access model reached this behaviour, in which case
    this file is updated IN THE SAME COMMIT, with the new expectation written
    out; or
  * something drifted that nobody meant to touch — which is the case this file
    is really defending against.

── THIS FILE CHANGED AT P1-6, AS ITS PREVIOUS VERSION PREDICTED ─────────────
The version before this one pinned the coupling deliberately: it asserted that
an Application `admin` held `manage_service_accounts` and `record_agent_runs`,
and that `ceo`/`coo`/`chief_of_staff` each held seven Developer read
permissions. Those assertions were tripwires for exactly this change, and they
have been rewritten to assert the OPPOSITE — that no Application role grants any
Developer authority at all. Section 2 below is where that flip is recorded.

The load-bearing new assertion is `test_a_founder_keeps_exactly_the_permissions
_they_had_before_the_split`: the split moved nine permissions out of the
Application role map and into the Developer one, and a founder — the only role
any existing production membership holds — must come out with the identical set
of 19. That is the regression this whole file protects.

Deliberately PURE — no database, no fixtures, no event loop. Both role maps are
module-level dicts and `effective_permissions` is a pure function, so this suite
is milliseconds and can be run after every edit.
"""
from __future__ import annotations

import hashlib
import inspect
import json

from navis.identity.permissions import (
    AGENT_ASSIGNABLE_PERMISSIONS,
    APPLICATION_PERMISSIONS,
    DEFAULT_ROLE_PERMISSIONS,
    DEVELOPER_PERMISSIONS,
    DEVELOPER_ROLE_PERMISSIONS,
    VALID_PERMISSIONS,
    Permission,
    effective_permissions,
)

# ── The frozen matrices ─────────────────────────────────────────────────────
#
# Written out in full rather than derived, so a diff shows exactly which
# permission moved for which role. A derived expectation would pass while the
# thing it derives from changed underneath it.

#: THE APPLICATION SURFACE. Every entry is the pre-P1-6 set minus the nine
#: permissions that only Developer routes gate on — nothing else moved, which is
#: why no Application route's behaviour changed.
APPLICATION_BASELINE: dict[str, set[str]] = {
    "founder": {
        "view_signals", "view_audit", "execute_actions", "manage_integrations",
        "manage_policies", "manage_members", "manage_roles", "manage_api_keys",
        "manage_sso", "manage_billing",
    },
    "ceo": {
        "view_signals", "view_audit", "execute_actions", "manage_members",
        "manage_policies", "manage_integrations",
    },
    "coo": {
        "view_signals", "execute_actions", "manage_integrations", "manage_members",
    },
    "chief_of_staff": {"view_signals", "execute_actions", "manage_members"},
    "admin": {
        "view_signals", "view_audit", "manage_members", "manage_roles",
        "manage_integrations", "manage_api_keys", "manage_sso",
    },
    "employee": {"view_signals"},
}

#: THE DEVELOPER SURFACE. Strictly nested: viewer < developer < admin = owner.
DEVELOPER_BASELINE: dict[str, set[str]] = {
    "viewer": {
        "view_state", "view_memory", "view_reasoning", "view_decisions",
        "view_observation", "view_learning", "view_intelligence",
    },
    # `manage_own_agents` joined at P1-7: the self-serve grant that lets an
    # engineer register their own development agent without an organization
    # administrator. It is bounded to `created_by`, not to the whole fleet.
    "developer": {
        "view_state", "view_memory", "view_reasoning", "view_decisions",
        "view_observation", "view_learning", "view_intelligence",
        "record_agent_runs", "manage_own_agents",
    },
    # `manage_developer_access` joined BOTH of these at P1-8: administering
    # the Developer TEAM — listing members, inviting, granting and revoking
    # Developer access — which until then was reachable only through the
    # Application permissions `manage_members`/`manage_roles`. A Developer
    # administrator who is an Application `employee` could not run their own
    # team.
    #
    # The two sets stay IDENTICAL, deliberately: `_ensure_grantable` compares
    # permission sets, so owner and admin must remain comparable. What
    # separates them is governance — only an owner may confer or remove the
    # owner role (`MemberService._actor_is_developer_owner`).
    "admin": {
        "view_state", "view_memory", "view_reasoning", "view_decisions",
        "view_observation", "view_learning", "view_intelligence",
        "record_agent_runs", "manage_own_agents", "manage_service_accounts",
        "manage_developer_access",
    },
    "owner": {
        "view_state", "view_memory", "view_reasoning", "view_decisions",
        "view_observation", "view_learning", "view_intelligence",
        "record_agent_runs", "manage_own_agents", "manage_service_accounts",
        "manage_developer_access",
    },
}

#: THE PRE-SPLIT TOTAL for every role that had one, kept so the union tests can
#: assert against what a user ACTUALLY experienced before P1-6 rather than
#: against a set reconstructed from the new maps.
PRE_SPLIT_TOTALS: dict[str, int] = {
    "founder": 19, "ceo": 13, "coo": 11, "chief_of_staff": 10,
    "admin": 16, "employee": 2,
}

#: The vocabulary grew by one at P1-7 (`manage_own_agents`) and by one at P1-8
#: (`manage_developer_access`), so a founder's total is two more than the
#: pre-split figure above. Stated as an explicit delta rather than by editing
#: `PRE_SPLIT_TOTALS`, because that table records what users ACTUALLY had
#: before the split and must not drift.
PERMISSIONS_ADDED_SINCE_THE_SPLIT = 2

#: Fingerprints of the matrices above. One assertion each that fails on ANY
#: change to ANY role, including one the per-role tests were not updated to
#: cover.
APPLICATION_SHA256 = "253842bde3562ad786ea66e31c6544d7b6d705b8266aad419d6d6c7bd5c97abc"
# Recomputed at P1-8, when `manage_developer_access` joined Developer admin
# and owner. The Application fingerprint above is UNCHANGED, which is the
# claim worth checking in review: the new permission is Developer-classified,
# so `_application_only` strips it and no Application role moved.
DEVELOPER_SHA256 = "5c8d3b97e7e5683d82aab101c688bead0bcae8cf72c7932be998f660b0a97b28"


def _fingerprint(matrix: dict) -> str:
    return hashlib.sha256(
        json.dumps({r: sorted(p) for r, p in sorted(matrix.items())}, sort_keys=True).encode()
    ).hexdigest()


# ── 1. The matrices themselves ──────────────────────────────────────────────


def test_every_application_role_resolves_exactly_its_baseline_permissions():
    for role, expected in APPLICATION_BASELINE.items():
        actual = set(DEFAULT_ROLE_PERMISSIONS[role])
        assert actual == expected, (
            f"Application role {role!r} changed.\n"
            f"  gained: {sorted(actual - expected)}\n"
            f"  lost:   {sorted(expected - actual)}"
        )


def test_every_developer_role_resolves_exactly_its_baseline_permissions():
    for role, expected in DEVELOPER_BASELINE.items():
        actual = set(DEVELOPER_ROLE_PERMISSIONS[role])
        assert actual == expected, (
            f"Developer role {role!r} changed.\n"
            f"  gained: {sorted(actual - expected)}\n"
            f"  lost:   {sorted(expected - actual)}"
        )


def test_the_application_matrix_fingerprint_is_unchanged():
    """Catches a role this file forgot to enumerate."""
    assert _fingerprint(DEFAULT_ROLE_PERMISSIONS) == APPLICATION_SHA256


def test_the_developer_matrix_fingerprint_is_unchanged():
    assert _fingerprint(DEVELOPER_ROLE_PERMISSIONS) == DEVELOPER_SHA256


def test_the_role_vocabularies_are_unchanged():
    assert set(DEFAULT_ROLE_PERMISSIONS) == set(APPLICATION_BASELINE)
    assert set(DEVELOPER_ROLE_PERMISSIONS) == set(DEVELOPER_BASELINE)


def test_the_permission_vocabulary_has_grown_by_exactly_two_since_the_split():
    """P1-6 ADDED NO PERMISSION; P1-7 AND P1-8 ADDED EXACTLY ONE EACH.

    The surface split partitioned the nineteen that existed.
    `manage_own_agents` (P1-7) is the twentieth — the vocabulary could not
    otherwise express "register an agent of your own without administering
    everybody else's". `manage_developer_access` (P1-8) is the twenty-first,
    and the audit that justified it ruled out all three reuse candidates:
    `manage_service_accounts` is agent-assignable (a machine could then grant
    humans access), and `manage_roles`/`manage_members` are Application
    permissions that would re-couple the surfaces.

    A twenty-SECOND appearing here means somebody solved a problem by
    inventing a permission instead of reusing one, which is what this number
    exists to make visible.
    """
    assert len(VALID_PERMISSIONS) == 19 + PERMISSIONS_ADDED_SINCE_THE_SPLIT


def test_no_role_holds_a_permission_outside_the_vocabulary():
    for label, matrix in (
        ("application", DEFAULT_ROLE_PERMISSIONS),
        ("developer", DEVELOPER_ROLE_PERMISSIONS),
    ):
        for role, perms in matrix.items():
            unknown = set(perms) - VALID_PERMISSIONS
            assert unknown == set(), f"{label}/{role} holds unknown: {sorted(unknown)}"


# ── 2. The coupling P1-6 removed ────────────────────────────────────────────
#
# The four tests in this section previously asserted the BUG. They now assert
# its absence. Flipping them is what makes the change deliberate rather than
# incidental.


def test_the_surface_partition_is_a_partition():
    assert DEVELOPER_PERMISSIONS.isdisjoint(APPLICATION_PERMISSIONS)
    assert DEVELOPER_PERMISSIONS | APPLICATION_PERMISSIONS == VALID_PERMISSIONS


def test_no_application_role_grants_developer_authority():
    """THE BUG, INVERTED. Was: `admin` silently holds `manage_service_accounts`
    and `record_agent_runs`. Now: no Application role holds any of the nine.

    This is the single assertion that makes "an existing Application user should
    not automatically receive Developer authority" a property of the code rather
    than a claim about it.
    """
    for role, perms in DEFAULT_ROLE_PERMISSIONS.items():
        leaked = set(perms) & DEVELOPER_PERMISSIONS
        assert leaked == set(), (
            f"Application role {role!r} grants Developer authority: {sorted(leaked)}"
        )


def test_no_developer_role_grants_application_authority():
    """The converse, and equally required. A Developer must not acquire billing,
    membership, SSO or action-execution powers by being a Developer — Step 8's
    least-privilege requirement, stated as an invariant."""
    for role, perms in DEVELOPER_ROLE_PERMISSIONS.items():
        leaked = set(perms) & APPLICATION_PERMISSIONS
        assert leaked == set(), (
            f"Developer role {role!r} grants Application authority: {sorted(leaked)}"
        )


def test_an_application_admin_can_no_longer_manage_agents_or_record_runs():
    """The specific escalation `0051_surface_rbac_columns` was written to end."""
    admin = set(DEFAULT_ROLE_PERMISSIONS["admin"])
    assert Permission.MANAGE_SERVICE_ACCOUNTS not in admin
    assert Permission.RECORD_AGENT_RUNS not in admin
    # And is still a fully-powered Application administrator.
    assert Permission.MANAGE_MEMBERS in admin
    assert Permission.MANAGE_ROLES in admin
    assert Permission.MANAGE_API_KEYS in admin
    assert Permission.MANAGE_SSO in admin


def test_executive_roles_no_longer_hold_developer_reads():
    """Was: each of these held seven Developer read permissions without anybody
    granting them Developer access. This was the under-reported half of the
    coupling — the previous file pinned only `admin`."""
    for role in ("ceo", "coo", "chief_of_staff"):
        held = set(DEFAULT_ROLE_PERMISSIONS[role]) & DEVELOPER_PERMISSIONS
        assert held == set(), f"{role} still holds {sorted(held)}"


def test_developer_roles_are_strictly_nested():
    """`_ensure_grantable` compares permission SETS, so an incomparable pair of
    Developer roles would make "may I grant this" depend on which of two
    equally-senior roles you happened to name."""
    viewer = DEVELOPER_ROLE_PERMISSIONS["viewer"]
    developer = DEVELOPER_ROLE_PERMISSIONS["developer"]
    admin = DEVELOPER_ROLE_PERMISSIONS["admin"]
    owner = DEVELOPER_ROLE_PERMISSIONS["owner"]
    assert viewer < developer < admin
    assert owner == admin, "owner and admin differ by GOVERNANCE, not privilege"


def test_a_developer_viewer_cannot_manage_credentials():
    """Step 8, verbatim: "A Developer Viewer should not automatically manage
    credentials"."""
    assert Permission.MANAGE_SERVICE_ACCOUNTS not in DEVELOPER_ROLE_PERMISSIONS["viewer"]
    assert Permission.RECORD_AGENT_RUNS not in DEVELOPER_ROLE_PERMISSIONS["viewer"]


def test_a_plain_developer_cannot_manage_agents():
    """The working role writes the ledger but cannot mint identities."""
    developer = DEVELOPER_ROLE_PERMISSIONS["developer"]
    assert Permission.RECORD_AGENT_RUNS in developer
    assert Permission.MANAGE_SERVICE_ACCOUNTS not in developer


# ── 3. The regression the split must not cause ──────────────────────────────


def test_a_founder_keeps_exactly_the_permissions_they_had_before_the_split():
    """THE MOST IMPORTANT ASSERTION IN THIS FILE.

    Every membership in production is `role='founder'`, and migration
    `0053_developer_access` gives each of them `developer_role='owner'`. The
    union of the two halves must therefore be the identical 19 permissions a
    founder resolved to before P1-6 — otherwise the change is a live access
    regression for every existing user, whichever direction it moves.
    """
    after = effective_permissions("founder", developer_role="owner")
    assert after == VALID_PERMISSIONS
    # The pre-split total, plus the permissions the vocabulary has gained since.
    # A founder holds everything, so this must track the vocabulary exactly.
    assert len(after) == (
        PRE_SPLIT_TOTALS["founder"] + PERMISSIONS_ADDED_SINCE_THE_SPLIT
    )


def test_an_application_only_member_keeps_every_application_permission():
    """The other half of the safety claim: the roles that are NOT backfilled lose
    only Developer permissions, never an Application one. No Application route
    gates on a Developer permission, so nothing any of them could do changes."""
    for role, expected in APPLICATION_BASELINE.items():
        resolved = effective_permissions(role)
        assert resolved == frozenset(expected)
        assert resolved <= APPLICATION_PERMISSIONS


def test_the_two_surfaces_are_additive():
    """Holding a Developer role never removes an Application permission, and
    vice versa. `effective_permissions` is a union, and this pins it."""
    app_only = effective_permissions("ceo")
    both = effective_permissions("ceo", developer_role="viewer")
    assert app_only < both
    assert both - app_only == DEVELOPER_ROLE_PERMISSIONS["viewer"]


def test_a_developer_only_member_reaches_no_application_authority():
    """The Developer-team flow: `employee` + a Developer role. The result must
    carry no Application administrative power at all."""
    resolved = effective_permissions("employee", developer_role="developer")
    assert Permission.MANAGE_MEMBERS not in resolved
    assert Permission.MANAGE_BILLING not in resolved
    assert Permission.MANAGE_SSO not in resolved
    assert Permission.EXECUTE_ACTIONS not in resolved
    assert DEVELOPER_ROLE_PERMISSIONS["developer"] <= resolved


# ── 4. The invariant that protects every existing agent ─────────────────────


def test_machine_credentials_do_not_resolve_permissions_from_membership():
    """THE MOST IMPORTANT ASSERTION ABOUT AGENTS.

    An API key's permissions come from `api_keys.scopes` and from nothing else.
    That is what makes the entire human access model unable to change what any
    existing agent credential can reach — including the live Navis Engineer
    agent. If this ever stops being true, a change to a human's role starts
    silently re-scoping running agents.

    It is also why P1-6 needed no agent migration: a service-account principal
    never reads a `memberships` row, so it has no `developer_role` to resolve
    and could not have been affected by the split.
    """
    from navis.identity import authz

    source = inspect.getsource(authz._principal_from_api_key)
    assert "scopes = list(key.scopes or [])" in source
    assert "permissions=frozenset(scopes)" in source
    # The human path's resolver must not appear on the machine path.
    assert "effective_permissions(" not in source
    assert "DEFAULT_ROLE_PERMISSIONS" not in source
    assert "DEVELOPER_ROLE_PERMISSIONS" not in source
    assert "Membership" not in source


def test_effective_permissions_is_defaults_union_grants():
    """The human resolver is additive over the role defaults, on both surfaces."""
    assert effective_permissions("employee", []) == frozenset(
        APPLICATION_BASELINE["employee"]
    )
    assert effective_permissions("employee", ["view_audit"]) == frozenset(
        APPLICATION_BASELINE["employee"] | {"view_audit"}
    )
    # Developer grants layer onto the Developer defaults the same way.
    assert effective_permissions(
        "employee", developer_role="viewer", developer_grants=["record_agent_runs"]
    ) == frozenset(
        APPLICATION_BASELINE["employee"]
        | DEVELOPER_BASELINE["viewer"]
        | {"record_agent_runs"}
    )


def test_an_unknown_role_resolves_to_no_permissions():
    """Fail-closed on BOTH surfaces. A membership carrying a role this build does
    not know must reach nothing rather than defaulting to a populated set."""
    assert effective_permissions("not_a_real_role", []) == frozenset()
    assert effective_permissions(None, []) == frozenset()
    assert effective_permissions(None, developer_role="not_a_real_role") == frozenset()
    # NULL is not "unknown" — it is the defined value for no Developer access.
    assert effective_permissions(None, developer_role=None) == frozenset()


def test_agent_assignable_is_the_developer_surface_minus_team_administration():
    """The console offers a machine credential the Developer surface's
    permissions and no more — an agent has no business holding
    `manage_billing`.

    NO LONGER AN EQUALITY, as of P1-8, and the one exclusion is named rather
    than tolerated by a loose `<=`. `manage_developer_access` administers
    PEOPLE — who is on the Developer team and what access they hold — and a
    machine credential able to grant humans Developer access would be a way
    out of the agent sandbox. It is the first Developer permission that is
    human-only; every other one remains assignable, and a second exclusion
    appearing here has to be argued for in the same way.

    The "no Developer route is unreachable by an agent" half is unaffected and
    is tested directly by `test_agent_assignable_matches_developer_routes`: no
    route under `navis/developer/` gates on this permission.
    """
    assert AGENT_ASSIGNABLE_PERMISSIONS <= DEVELOPER_PERMISSIONS
    assert DEVELOPER_PERMISSIONS - AGENT_ASSIGNABLE_PERMISSIONS == {
        "manage_developer_access"
    }
