"""Which ROLES may register an agent.

WHY THIS FILE EXISTS. The Navis console showed an organization's FOUNDER the
sentence "Your role cannot register agents — it needs the
manage_service_accounts permission." The claim was false: `founder` resolves to
18 permissions and `manage_service_accounts` is one of them, so the Core would
have allowed the write all along. The defect was in the console, which turned a
failed `/auth/me` into an assertion about the operator's role.

`tests/identity/test_agents.py` already covers the agent LIFECYCLE — escalation,
tenant scoping, revocation, credential inheritance. What no test covered was the
join between the ROLE→PERMISSION table and the gate this router actually
declares: nothing failed if `MANAGE_SERVICE_ACCOUNTS` were dropped from
`FOUNDER_PERMISSIONS`, or if the router were re-gated on a different permission.
Either change would silently make the console's sentence TRUE, which is the
version of this bug that would be a real authorization regression rather than a
display one.

So these tests pin the contract from both ends: the role->permission table, and
the gate the router declares. They deliberately stop at the authorization
decision — no database, no entitlement row, no token. The full HTTP path, with a
real founder JWT against real Postgres, is
`tests/integration/test_developer_agents_postgres.py`, which already proves a
founder registers an agent in their own organization and that a client-supplied
organization cannot move the write.

NOTE ON THE READ GATE. `_reader` and `_operator` in `navis/developer/agents_router.py`
require the SAME permission. That equality is what made the console's claim
self-refuting — it printed "you lack manage_service_accounts" directly beneath a
200 from the list route, which only a caller holding that permission can obtain —
and it is asserted below so it cannot quietly stop being true.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from navis.identity.authz import Principal, PrincipalKind, require_permission
from navis.identity.models import User
from navis.identity.permissions import (
    DEFAULT_ROLE_PERMISSIONS,
    DEVELOPER_ROLE_PERMISSIONS,
    Permission,
    default_permissions_for_role,
    effective_permissions,
)

ORG_A = uuid.uuid4()

#: The permission `agents_router` gates every route on. Imported from the
#: vocabulary rather than spelled as a literal, so a rename moves both.
AGENT_PERMISSION = Permission.MANAGE_SERVICE_ACCOUNTS


def _principal(
    role: str,
    *,
    developer_role: str | None = None,
    kind: PrincipalKind = PrincipalKind.USER,
) -> Principal:
    """A principal built exactly as `_principal_from_jwt` builds one.

    BOTH SURFACES, SINCE P1-6. Resolving from the Application role alone would
    make this helper model a session that cannot occur: production always reads
    `memberships.developer_role` and unions it in, so a fixture that omits it
    would test a principal no real credential produces.
    """
    uid = uuid.uuid4()
    return Principal(
        kind=kind,
        org_id=ORG_A,
        subject_id=uid,
        permissions=effective_permissions(role, developer_role=developer_role),
        role=role,
        developer_role=developer_role,
        user=User(id=uid, email=f"{uid.hex[:8]}@x.com"),
    )


# ── The role → permission table ─────────────────────────────────────────────


def test_founder_holds_the_permission_the_agents_router_requires():
    """THE REGRESSION. The console told a founder they lacked this.

    WHERE THE PERMISSION COMES FROM MOVED AT P1-6. A founder holds it through
    their Developer role (`owner`, written by signup and by migration
    `0053_developer_access`) rather than through their Application role. The
    property this test defends is unchanged: an organization's founder can
    register an agent, and the console's sentence stays false.
    """
    assert AGENT_PERMISSION in effective_permissions("founder", developer_role="owner")
    assert AGENT_PERMISSION in DEVELOPER_ROLE_PERMISSIONS["owner"]


def test_the_founder_permission_set_matches_what_the_console_reports():
    """`/auth/me` publishes this set, and the console renders its size.

    It was 18 when the founder/agents bug was diagnosed — the console showed
    "18 granted" on the very screen that denied the founder agent access — and
    became 19 when `record_agent_runs` joined the vocabulary for the agent
    execution ledger. It is STILL 19 after the P1-6 surface split, which is the
    whole safety claim of that change: the nine Developer permissions moved from
    the Application role map to the Developer one, and a founder — who is a
    Developer owner — comes out holding exactly what they held before.

    The number is asserted rather than the exact set so that adding a permission
    is a DELIBERATE edit here, which is the same guarantee
    `test_every_permission_has_a_deliberate_founder_decision` gives upstream.
    It became 20 at P1-7, when `manage_own_agents` joined the vocabulary, and
    21 at P1-8, when `manage_developer_access` did — the permission that lets a
    Developer administrator run their own team without holding the Application
    `manage_members`/`manage_roles`. A founder is a Developer owner, so they
    hold it and the union is still the whole vocabulary.
    """
    from navis.identity.permissions import VALID_PERMISSIONS

    resolved = effective_permissions("founder", developer_role="owner")
    assert len(resolved) == 21
    # And a founder holds ALL of them, so this number cannot drift from the
    # vocabulary without one of the two assertions failing.
    assert resolved == VALID_PERMISSIONS


def test_an_application_admin_may_no_longer_manage_agents():
    """THE P1-6 NARROWING, PINNED.

    Was: "Delegated administration is the other role that runs machine
    identity." Registering an agent creates an identity that outlives the
    creator's own revocation, and `0051_surface_rbac_columns` named an
    Application administrator silently holding this as the bug the surface split
    exists to fix. An `admin` who should run machine identity is now given a
    Developer role explicitly, and the assertion below is what keeps that
    explicit.
    """
    assert AGENT_PERMISSION not in default_permissions_for_role("admin")
    # And an admin who IS granted Developer admin holds it again — the grant is
    # the mechanism, not a special case.
    assert AGENT_PERMISSION in effective_permissions("admin", developer_role="admin")


@pytest.mark.parametrize("role", ["founder", "ceo", "coo", "chief_of_staff", "admin", "employee"])
def test_no_application_role_alone_holds_it(role: str):
    """The negative half, now true of EVERY Application role. If this ever starts
    failing, an Application role has quietly regained the ability to mint machine
    credentials without anybody granting Developer access."""
    assert AGENT_PERMISSION not in default_permissions_for_role(role)


@pytest.mark.parametrize("developer_role", ["viewer", "developer"])
def test_junior_developer_roles_do_not_hold_it(developer_role: str):
    """Least privilege inside the Developer surface: reading the organization and
    recording runs do not imply managing identities."""
    assert AGENT_PERMISSION not in DEVELOPER_ROLE_PERMISSIONS[developer_role]


def test_every_declared_role_has_an_explicit_answer():
    """No role may be absent from the table — an unmapped role resolves to the
    empty set, which is a silent denial rather than a decided one."""
    for role in ["founder", "ceo", "coo", "chief_of_staff", "admin", "employee"]:
        assert role in DEFAULT_ROLE_PERMISSIONS, f"{role} has no permission mapping"


# ── The gate itself ─────────────────────────────────────────────────────────
#
# `require_developer(perm, cap)` checks the PERMISSION first and the capability
# second — 403 before 402 — so exercising `require_permission` is exercising the
# decision this bug was about, without needing an entitlement row or a database.
# The full HTTP path, with entitlement, rate limiting and a real founder token,
# is `tests/integration/test_developer_agents_postgres.py`.


@pytest.mark.asyncio
async def test_the_gate_admits_a_founder():
    """No exception raised is the gate's way of saying yes.

    A founder is a Developer owner — that is what signup writes and what
    `0053_developer_access` backfilled — so this is the session a real founder
    presents, not a constructed best case.
    """
    gate = require_permission(AGENT_PERMISSION)
    await gate(_principal("founder", developer_role="owner"))


@pytest.mark.asyncio
async def test_the_gate_admits_a_member_with_developer_admin():
    """Whoever is granted Developer admin may manage agents, whatever their
    Application role is — including the least-privileged one. Developer access
    does not depend on Application standing."""
    gate = require_permission(AGENT_PERMISSION)
    await gate(_principal("employee", developer_role="admin"))


@pytest.mark.asyncio
async def test_the_gate_refuses_an_application_admin_without_developer_access():
    """THE NARROWING, AT THE GATE. An Application administrator with no Developer
    role is refused the agent routes — the escalation `0051` named, closed."""
    gate = require_permission(AGENT_PERMISSION)

    with pytest.raises(HTTPException) as raised:
        await gate(_principal("admin"))

    assert raised.value.status_code == 403


@pytest.mark.parametrize("role", ["founder", "ceo", "coo", "chief_of_staff", "employee"])
@pytest.mark.asyncio
async def test_the_gate_refuses_a_role_without_the_permission(role: str):
    """Case 3 from the brief: a member without the permission is refused, and
    the refusal is a 403 from the Core rather than a decision made in the UI.

    `founder` is in this list now: a founder WITHOUT a Developer role holds no
    Developer authority either. The role does not carry it; the Developer
    assignment does.
    """
    gate = require_permission(AGENT_PERMISSION)

    with pytest.raises(HTTPException) as raised:
        await gate(_principal(role))

    assert raised.value.status_code == 403


@pytest.mark.parametrize("developer_role", ["viewer", "developer"])
@pytest.mark.asyncio
async def test_the_gate_refuses_a_junior_developer_role(developer_role: str):
    """Least privilege at the gate: a Developer viewer or a plain Developer is
    refused agent management even though both hold Developer access."""
    gate = require_permission(AGENT_PERMISSION)

    with pytest.raises(HTTPException) as raised:
        await gate(_principal("employee", developer_role=developer_role))

    assert raised.value.status_code == 403


@pytest.mark.asyncio
async def test_a_scopeless_machine_credential_is_refused():
    """An API key minted with no scopes holds no permission and reaches none of
    the agent routes, regardless of who created it."""
    gate = require_permission(AGENT_PERMISSION)
    uid = uuid.uuid4()
    scopeless = Principal(
        kind=PrincipalKind.API_KEY,
        org_id=ORG_A,
        subject_id=uid,
        permissions=frozenset(),
        role=None,
        user=None,
    )

    with pytest.raises(HTTPException) as raised:
        await gate(scopeless)

    assert raised.value.status_code == 403


def test_the_read_and_the_write_share_one_permission_set():
    """The equality that made the console's claim self-refuting.

    `_reader` and `_operator` in `navis/developer/agents_router.py` are built
    from the SAME permission set — since P1-7 that is `_AGENT_PERMISSIONS`,
    `(MANAGE_SERVICE_ACCOUNTS, MANAGE_OWN_AGENTS)`, rather than one string. The
    property the console depends on is unchanged and is what this pins: a 200
    from the list route is proof the caller may also register, so "No agents
    yet" and "your role cannot register agents" can never both be true.

    Asserted against the module source: the gates are built inside closures, so
    there is no object to compare, and the thing worth pinning is that the file
    hands BOTH the same set.
    """
    import inspect

    from navis.developer import agents_router as module

    source = inspect.getsource(module)
    # Real CALL SITES only — a line that both invokes the gate and passes the
    # capability. Matching `require_developer_any(` alone also matches the
    # module docstring, which names the gate in prose.
    gates = [
        line
        for line in source.splitlines()
        if "require_developer_any(" in line and "_AGENT_CAPABILITY" in line
    ]

    # Two call sites: `_operator`'s dependency and `_reader`.
    assert len(gates) == 2, f"expected exactly two gate call sites, found {gates}"
    for line in gates:
        assert "_AGENT_PERMISSIONS" in line, (
            "an agent route is gated on something other than _AGENT_PERMISSIONS; "
            "the console relies on the read and the write sharing one permission set"
        )


def test_the_agent_permission_set_is_exactly_the_two_grants():
    """WHICH permissions that set holds, pinned separately from the fact that
    both gates share it.

    Organization-wide administration stays in the set, so an existing credential
    carrying `manage_service_accounts` reaches these routes exactly as it did
    before P1-7 — that is the no-regression half. The self-serve grant joins it,
    which is the new half.
    """
    from navis.developer.agents_router import _AGENT_PERMISSIONS

    assert set(_AGENT_PERMISSIONS) == {
        Permission.MANAGE_SERVICE_ACCOUNTS,
        Permission.MANAGE_OWN_AGENTS,
    }
