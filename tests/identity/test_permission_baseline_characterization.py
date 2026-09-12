"""CHARACTERIZATION of today's effective permissions — the migration's baseline.

WHAT A CHARACTERIZATION TEST IS, AND WHY THIS ONE LOOKS ODD.

These tests do not assert that the current behaviour is CORRECT. Several of them
assert the opposite — that an Application administrator today silently holds
Developer authority, which is precisely the coupling the Application/Developer
RBAC split exists to remove. They are here to make the current behaviour
IMPOSSIBLE TO CHANGE BY ACCIDENT.

So a failure here means one of two things, and telling them apart is the whole
point:

  * an intended step of the RBAC migration reached this behaviour, in which case
    the test is updated IN THE SAME COMMIT as the change, with the new
    expectation written out; or
  * something drifted that nobody meant to touch — which is the case this file
    is really defending against, because a permission table is exactly the kind
    of thing that gets "tidied" in passing.

THE FILE IS EXPECTED TO CHANGE AT PHASE 5 (legacy flip). Until then every
assertion below must hold unchanged, including the ones describing the bug.

Deliberately PURE — no database, no fixtures, no event loop. `DEFAULT_ROLE_
PERMISSIONS` is a module-level dict and `effective_permissions` is a pure
function, so this suite is milliseconds and can be run after every edit.
"""
from __future__ import annotations

import hashlib
import inspect
import json

from navis.identity.permissions import (
    AGENT_ASSIGNABLE_PERMISSIONS,
    DEFAULT_ROLE_PERMISSIONS,
    VALID_PERMISSIONS,
    Permission,
    effective_permissions,
)

# ── The frozen matrix ───────────────────────────────────────────────────────
#
# Written out in full rather than derived, so a diff shows exactly which
# permission moved for which role. A derived expectation would pass while the
# thing it derives from changed underneath it.

BASELINE: dict[str, set[str]] = {
    "founder": {
        "view_signals", "view_memory", "view_state", "view_reasoning",
        "view_decisions", "view_intelligence", "view_observation", "view_learning",
        "view_audit", "execute_actions", "manage_integrations", "manage_policies",
        "manage_members", "manage_roles", "manage_api_keys",
        "manage_service_accounts", "manage_sso", "manage_billing",
        "record_agent_runs",
    },
    "ceo": {
        "view_signals", "view_memory", "view_state", "view_reasoning",
        "view_decisions", "view_intelligence", "view_observation", "view_learning",
        "execute_actions", "manage_members", "manage_policies",
        "manage_integrations", "view_audit",
    },
    "coo": {
        "view_signals", "view_memory", "view_state", "view_reasoning",
        "view_decisions", "view_intelligence", "view_observation", "view_learning",
        "execute_actions", "manage_integrations", "manage_members",
    },
    "chief_of_staff": {
        "view_signals", "view_memory", "view_state", "view_reasoning",
        "view_decisions", "view_intelligence", "view_observation", "view_learning",
        "execute_actions", "manage_members",
    },
    "admin": {
        "view_signals", "view_memory", "view_state", "view_reasoning",
        "view_decisions", "view_intelligence", "view_observation", "view_learning",
        "manage_members", "manage_roles", "manage_integrations", "manage_api_keys",
        "manage_service_accounts", "manage_sso", "view_audit", "record_agent_runs",
    },
    "employee": {"view_decisions", "view_signals"},
}

#: Fingerprint of the matrix above. One assertion that fails on ANY change to
#: ANY role, including one this file's per-role tests were not updated to cover.
BASELINE_SHA256 = "6ac626070d64892716d5b0a2104bb61b29eddbf57070577aabad4185457dc6b1"


def _fingerprint(matrix: dict) -> str:
    return hashlib.sha256(
        json.dumps({r: sorted(p) for r, p in sorted(matrix.items())}, sort_keys=True).encode()
    ).hexdigest()


# ── 1. The matrix itself ────────────────────────────────────────────────────


def test_every_role_resolves_exactly_its_baseline_permissions():
    for role, expected in BASELINE.items():
        actual = set(DEFAULT_ROLE_PERMISSIONS[role])
        assert actual == expected, (
            f"role {role!r} changed.\n"
            f"  gained: {sorted(actual - expected)}\n"
            f"  lost:   {sorted(expected - actual)}"
        )


def test_the_matrix_fingerprint_is_unchanged():
    """Catches a role this file forgot to enumerate."""
    assert _fingerprint(DEFAULT_ROLE_PERMISSIONS) == BASELINE_SHA256


def test_the_role_vocabulary_is_unchanged():
    assert set(DEFAULT_ROLE_PERMISSIONS) == set(BASELINE)


def test_the_permission_vocabulary_is_nineteen():
    assert len(VALID_PERMISSIONS) == 19


def test_no_role_holds_a_permission_outside_the_vocabulary():
    for role, perms in DEFAULT_ROLE_PERMISSIONS.items():
        unknown = set(perms) - VALID_PERMISSIONS
        assert unknown == set(), f"{role} holds unknown permissions: {sorted(unknown)}"


# ── 2. The coupling this migration exists to remove ─────────────────────────
#
# These assert the CURRENT, INCORRECT behaviour on purpose. They are the
# tripwire for Phase 5: when the legacy flip lands, these four must be rewritten
# in the same commit, which is what forces the change to be deliberate.


def test_application_admin_today_also_holds_developer_authority():
    """THE BUG, PINNED. `admin` is an Application administrative role and today
    it can create agents and mint their credentials."""
    admin = DEFAULT_ROLE_PERMISSIONS["admin"]
    assert Permission.MANAGE_SERVICE_ACCOUNTS in admin
    assert Permission.RECORD_AGENT_RUNS in admin


def test_founder_today_also_holds_developer_authority():
    founder = DEFAULT_ROLE_PERMISSIONS["founder"]
    assert Permission.MANAGE_SERVICE_ACCOUNTS in founder
    assert Permission.RECORD_AGENT_RUNS in founder


def test_executive_roles_today_hold_developer_read_permissions():
    """`ceo`/`coo`/`chief_of_staff` are business roles that today carry the
    Developer boundary's read permissions."""
    for role in ("ceo", "coo", "chief_of_staff"):
        held = set(DEFAULT_ROLE_PERMISSIONS[role]) & AGENT_ASSIGNABLE_PERMISSIONS
        assert len(held) == 7, f"{role} holds {sorted(held)}"


def test_employee_is_the_only_role_without_developer_reads():
    employee = set(DEFAULT_ROLE_PERMISSIONS["employee"]) & AGENT_ASSIGNABLE_PERMISSIONS
    assert employee == {Permission.VIEW_DECISIONS}


# ── 3. The invariant that protects every existing agent ─────────────────────


def test_machine_credentials_do_not_resolve_permissions_from_membership():
    """THE MOST IMPORTANT ASSERTION IN THIS FILE.

    An API key's permissions come from `api_keys.scopes` and from nothing else.
    That is what makes the entire human-RBAC migration unable to change what any
    existing agent credential can reach — including the live Navis Engineer
    agent. If this ever stops being true, a change to a human's role starts
    silently re-scoping running agents.
    """
    from navis.identity import authz

    source = inspect.getsource(authz._principal_from_api_key)
    assert "scopes = list(key.scopes or [])" in source
    assert "permissions=frozenset(scopes)" in source
    # The human path's resolver must not appear on the machine path.
    assert "effective_permissions(" not in source
    assert "DEFAULT_ROLE_PERMISSIONS" not in source
    assert "Membership" not in source


def test_effective_permissions_is_defaults_union_grants():
    """The human resolver is additive over the role defaults. Phase 3's dual-read
    union relies on this shape being unchanged."""
    assert effective_permissions("employee", []) == frozenset(BASELINE["employee"])
    assert effective_permissions("employee", ["view_memory"]) == frozenset(
        BASELINE["employee"] | {"view_memory"}
    )


def test_an_unknown_role_resolves_to_no_permissions():
    """Fail-closed. A membership carrying a role this build does not know must
    reach nothing rather than defaulting to a populated set."""
    assert effective_permissions("not_a_real_role", []) == frozenset()
    assert effective_permissions(None, []) == frozenset()
