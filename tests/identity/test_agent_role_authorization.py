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
    Permission,
    default_permissions_for_role,
)

ORG_A = uuid.uuid4()

#: The permission `agents_router` gates every route on. Imported from the
#: vocabulary rather than spelled as a literal, so a rename moves both.
AGENT_PERMISSION = Permission.MANAGE_SERVICE_ACCOUNTS


def _principal(role: str, *, kind: PrincipalKind = PrincipalKind.USER) -> Principal:
    """A principal whose permissions come from the role, exactly as
    `_principal_from_jwt` builds them for a real session."""
    uid = uuid.uuid4()
    return Principal(
        kind=kind,
        org_id=ORG_A,
        subject_id=uid,
        permissions=default_permissions_for_role(role),
        role=role,
        user=User(id=uid, email=f"{uid.hex[:8]}@x.com"),
    )


# ── The role → permission table ─────────────────────────────────────────────


def test_founder_holds_the_permission_the_agents_router_requires():
    """THE REGRESSION. The console told a founder they lacked this."""
    assert AGENT_PERMISSION in default_permissions_for_role("founder")


def test_the_founder_permission_set_matches_what_the_console_reports():
    """`/auth/me` publishes this set, and the console renders its size.

    It was 18 when the founder/agents bug was diagnosed — the console showed
    "18 granted" on the very screen that denied the founder agent access — and
    is 19 since `record_agent_runs` joined the vocabulary for the agent
    execution ledger. The number is asserted rather than the exact set so that
    adding a permission is a DELIBERATE edit here, which is the same guarantee
    `test_every_permission_has_a_deliberate_founder_decision` gives upstream.
    """
    assert len(default_permissions_for_role("founder")) == 19


def test_admin_may_also_manage_agents():
    """Delegated administration is the other role that runs machine identity."""
    assert AGENT_PERMISSION in default_permissions_for_role("admin")


@pytest.mark.parametrize("role", ["ceo", "coo", "chief_of_staff", "employee"])
def test_roles_without_delegated_administration_do_not_hold_it(role: str):
    """The negative half. If this ever starts failing for `employee`, the
    least-privileged role has quietly gained the ability to mint credentials."""
    assert AGENT_PERMISSION not in default_permissions_for_role(role)


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
    """No exception raised is the gate's way of saying yes."""
    gate = require_permission(AGENT_PERMISSION)
    await gate(_principal("founder"))


@pytest.mark.asyncio
async def test_the_gate_admits_an_admin():
    gate = require_permission(AGENT_PERMISSION)
    await gate(_principal("admin"))


@pytest.mark.parametrize("role", ["ceo", "coo", "chief_of_staff", "employee"])
@pytest.mark.asyncio
async def test_the_gate_refuses_a_role_without_the_permission(role: str):
    """Case 3 from the brief: a member without the permission is refused, and
    the refusal is a 403 from the Core rather than a decision made in the UI."""
    gate = require_permission(AGENT_PERMISSION)

    with pytest.raises(HTTPException) as raised:
        await gate(_principal(role))

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


def test_the_read_and_the_write_share_one_permission():
    """The equality that made the console's claim self-refuting.

    `_reader` and `_operator` in `navis/developer/agents_router.py` are both
    built from `require_developer(MANAGE_SERVICE_ACCOUNTS, STATE)`. Because they
    are the same permission, a 200 from the list route is proof the caller may
    also register — which is why "No agents yet" and "your role cannot register
    agents" could never both be true, and why the console was the thing at
    fault.

    Asserted against the module source: the gate is built inside a closure, so
    there is no object to compare, and the thing worth pinning is that the file
    names ONE permission for both.
    """
    import inspect

    from navis.developer import agents_router as module

    source = inspect.getsource(module)
    gates = [line for line in source.splitlines() if "require_developer(" in line]

    # Two call sites: `_operator`'s dependency and `_reader`.
    assert len(gates) >= 2, "expected the reader and the writer to declare a gate each"
    for line in gates:
        assert "MANAGE_SERVICE_ACCOUNTS" in line, (
            "an agent route is gated on something other than MANAGE_SERVICE_ACCOUNTS; "
            "the console relies on the read and the write sharing one permission"
        )
