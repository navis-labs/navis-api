"""TASK 6.4.3 — privileged membership authorization: ACTOR vs TARGET.

The vulnerability: authorization considered the ACTOR's permissions and, for role
changes, the NEW role — never the TARGET's current standing. `admin` deliberately
holds delegated administration without business powers (no EXECUTE_ACTIONS, no
MANAGE_POLICIES), yet could remove a `ceo`, or demote one to `employee`, because
`employee`'s permissions are a subset of `admin`'s own.

There is NO role hierarchy to appeal to — `Role` is an unordered enum and the
permission sets form a partial order (`admin` and `ceo` are INCOMPARABLE; each
holds permissions the other lacks). The rule enforced here is the one the codebase
already used for granting: permission-set containment.
"""
from __future__ import annotations

import uuid

import pytest

from navis.identity.members import (
    MemberConflict,
    MemberError,
    MemberForbidden,
    MemberService,
)
from navis.identity.models import Membership, Role, User
from navis.identity.permissions import DEFAULT_ROLE_PERMISSIONS, default_permissions_for_role
from tests.identity.fakes import FakeSession

ORG_A = uuid.uuid4()
ORG_B = uuid.uuid4()
ACTOR = uuid.uuid4()

FOUNDER = default_permissions_for_role("founder")
ADMIN = default_permissions_for_role("admin")
CEO = default_permissions_for_role("ceo")
EMPLOYEE = default_permissions_for_role("employee")


def _seed(db, org_id, role: Role) -> uuid.UUID:
    """Add a member with `role` to `org_id`; returns their user id."""
    uid = uuid.uuid4()
    db.add(User(id=uid, email=f"{uid.hex[:8]}@x.com"))
    db.add(Membership(user_id=uid, organization_id=org_id, role=role))
    return uid


# ── The model this rests on ───────────────────────────────────────────────────

def test_roles_are_not_a_hierarchy():
    """Pins the premise. If someone later makes these comparable, the containment
    rule silently changes meaning and this test says so."""
    assert not (CEO <= ADMIN), "ceo must not be a subset of admin"
    assert not (ADMIN <= CEO), "admin must not be a subset of ceo"
    # Founder is the top of the lattice: everything is manageable by a founder.
    for role in DEFAULT_ROLE_PERMISSIONS:
        assert default_permissions_for_role(role) <= FOUNDER


# ── D. HIGHER-PRIVILEGED TARGET — the reported vulnerability ─────────────────

async def test_admin_cannot_remove_a_ceo():
    db = FakeSession()
    victim = _seed(db, ORG_A, Role.CEO)
    with pytest.raises(MemberForbidden):
        await MemberService(db).remove(
            org_id=ORG_A, user_id=victim, actor_id=ACTOR, actor_permissions=ADMIN
        )
    assert await db.get(Membership, None) is None or True  # membership survives below
    rows = await MemberService(db).list_members(ORG_A)
    assert [r.role for r in rows] == [Role.CEO], "target must be untouched"


async def test_admin_cannot_demote_a_ceo():
    """The subtler half: `employee` IS grantable by an admin, so checking only the
    NEW role let a demotion through."""
    db = FakeSession()
    victim = _seed(db, ORG_A, Role.CEO)
    with pytest.raises(MemberForbidden):
        await MemberService(db).assign_role(
            org_id=ORG_A, user_id=victim, role=Role.EMPLOYEE,
            actor_id=ACTOR, actor_permissions=ADMIN,
        )
    rows = await MemberService(db).list_members(ORG_A)
    assert [r.role for r in rows] == [Role.CEO]


async def test_admin_cannot_remove_a_founder():
    db = FakeSession()
    _seed(db, ORG_A, Role.FOUNDER)  # a second founder, so the last-founder guard
    victim = _seed(db, ORG_A, Role.FOUNDER)  # is NOT what does the denying here
    with pytest.raises(MemberForbidden):
        await MemberService(db).remove(
            org_id=ORG_A, user_id=victim, actor_id=ACTOR, actor_permissions=ADMIN
        )


async def test_ceo_cannot_remove_an_admin():
    """Incomparable, so it fails in BOTH directions — not a one-way hierarchy."""
    db = FakeSession()
    victim = _seed(db, ORG_A, Role.ADMIN)
    with pytest.raises(MemberForbidden):
        await MemberService(db).remove(
            org_id=ORG_A, user_id=victim, actor_id=ACTOR, actor_permissions=CEO
        )


async def test_employee_cannot_remove_a_more_privileged_member():
    """TWO LAYERS, and this test covers the second one only.

    An `employee` never reaches `MemberService` at all: the route requires
    MANAGE_MEMBERS, which `employee` does not hold (asserted over HTTP in
    `test_membership_authorization_http.py`). That route gate answers "may you
    administer members?"; this service check answers "may you administer THIS
    member?". The service must hold on its own regardless, because it is also the
    SCIM path's guard.
    """
    assert "manage_members" not in EMPLOYEE, "premise: employee cannot reach the route"
    db = FakeSession()
    for role in (Role.CEO, Role.ADMIN, Role.FOUNDER):
        victim = _seed(db, ORG_A, role)
        with pytest.raises(MemberForbidden):
            await MemberService(db).remove(
                org_id=ORG_A, user_id=victim, actor_id=ACTOR, actor_permissions=EMPLOYEE
            )


# ── B. SAME-ORG ALLOWED CASES (no over-blocking) ─────────────────────────────

async def test_founder_may_manage_every_role():
    for role in (Role.CEO, Role.COO, Role.CHIEF_OF_STAFF, Role.ADMIN, Role.EMPLOYEE):
        db = FakeSession()
        target = _seed(db, ORG_A, role)
        await MemberService(db).remove(
            org_id=ORG_A, user_id=target, actor_id=ACTOR, actor_permissions=FOUNDER
        )
        assert await MemberService(db).list_members(ORG_A) == []


async def test_admin_may_manage_an_employee():
    db = FakeSession()
    target = _seed(db, ORG_A, Role.EMPLOYEE)
    await MemberService(db).remove(
        org_id=ORG_A, user_id=target, actor_id=ACTOR, actor_permissions=ADMIN
    )
    assert await MemberService(db).list_members(ORG_A) == []


async def test_peer_administration_is_preserved():
    """Containment is reflexive, so peers may administer each other. Unchanged from
    the pre-existing grant rule and not an escalation — neither party gains a
    permission they did not already hold."""
    db = FakeSession()
    target = _seed(db, ORG_A, Role.ADMIN)
    await MemberService(db).remove(
        org_id=ORG_A, user_id=target, actor_id=ACTOR, actor_permissions=ADMIN
    )
    assert await MemberService(db).list_members(ORG_A) == []


# ── C. PRIVILEGE ESCALATION VIA ROLE ASSIGNMENT ──────────────────────────────

async def test_admin_cannot_promote_anyone_to_founder():
    db = FakeSession()
    target = _seed(db, ORG_A, Role.EMPLOYEE)
    with pytest.raises(MemberForbidden):
        await MemberService(db).assign_role(
            org_id=ORG_A, user_id=target, role=Role.FOUNDER,
            actor_id=ACTOR, actor_permissions=ADMIN,
        )


async def test_admin_cannot_promote_an_employee_to_ceo():
    """`ceo` holds EXECUTE_ACTIONS / MANAGE_POLICIES that admin lacks."""
    db = FakeSession()
    target = _seed(db, ORG_A, Role.EMPLOYEE)
    with pytest.raises(MemberForbidden):
        await MemberService(db).assign_role(
            org_id=ORG_A, user_id=target, role=Role.CEO,
            actor_id=ACTOR, actor_permissions=ADMIN,
        )


async def test_employee_cannot_self_promote():
    """F/C: self-targeting obeys the SAME rule — no special case, so there is no
    'assign self FOUNDER' path."""
    db = FakeSession()
    me = _seed(db, ORG_A, Role.EMPLOYEE)
    for role in (Role.ADMIN, Role.FOUNDER, Role.CEO):
        with pytest.raises(MemberForbidden):
            await MemberService(db).assign_role(
                org_id=ORG_A, user_id=me, role=role,
                actor_id=me, actor_permissions=EMPLOYEE,
            )


async def test_admin_cannot_self_promote_to_founder():
    db = FakeSession()
    me = _seed(db, ORG_A, Role.ADMIN)
    with pytest.raises(MemberForbidden):
        await MemberService(db).assign_role(
            org_id=ORG_A, user_id=me, role=Role.FOUNDER,
            actor_id=me, actor_permissions=ADMIN,
        )


async def test_invite_cannot_exceed_actor_privilege():
    db = FakeSession()
    with pytest.raises(MemberForbidden):
        await MemberService(db).invite(
            org_id=ORG_A, email="new@x.com", role=Role.FOUNDER,
            actor_id=ACTOR, actor_permissions=ADMIN,
        )


# ── A. CROSS-ORG ─────────────────────────────────────────────────────────────

async def test_actor_cannot_reach_a_member_of_another_organization():
    """Org scope comes from the actor's trusted Principal.org_id. A target that
    exists — but in another tenant — must be indistinguishable from absent."""
    db = FakeSession()
    victim_b = _seed(db, ORG_B, Role.EMPLOYEE)
    svc = MemberService(db)
    with pytest.raises(MemberError):
        await svc.remove(
            org_id=ORG_A, user_id=victim_b, actor_id=ACTOR, actor_permissions=FOUNDER
        )
    with pytest.raises(MemberError):
        await svc.assign_role(
            org_id=ORG_A, user_id=victim_b, role=Role.ADMIN,
            actor_id=ACTOR, actor_permissions=FOUNDER,
        )
    # Org B is untouched, and a FOUNDER of A could not do it either.
    assert [r.role for r in await svc.list_members(ORG_B)] == [Role.EMPLOYEE]


async def test_cross_org_denial_is_not_leaked_as_a_privilege_error():
    """The refusal must not distinguish 'exists elsewhere' from 'does not exist' —
    that difference is a cross-tenant membership oracle."""
    db = FakeSession()
    elsewhere = _seed(db, ORG_B, Role.FOUNDER)
    svc = MemberService(db)
    with pytest.raises(MemberError) as real:
        await svc.remove(org_id=ORG_A, user_id=elsewhere, actor_id=ACTOR, actor_permissions=FOUNDER)
    with pytest.raises(MemberError) as ghost:
        await svc.remove(org_id=ORG_A, user_id=uuid.uuid4(), actor_id=ACTOR, actor_permissions=FOUNDER)
    assert str(real.value) == str(ghost.value)
    assert not isinstance(real.value, MemberForbidden)


# ── E. LAST-FOUNDER PROTECTION (must survive the new check) ──────────────────

async def test_sole_founder_cannot_be_removed_or_demoted():
    db = FakeSession()
    sole = _seed(db, ORG_A, Role.FOUNDER)
    svc = MemberService(db)
    with pytest.raises(MemberConflict):
        await svc.remove(org_id=ORG_A, user_id=sole, actor_id=sole, actor_permissions=FOUNDER)
    with pytest.raises(MemberConflict):
        await svc.assign_role(
            org_id=ORG_A, user_id=sole, role=Role.EMPLOYEE,
            actor_id=sole, actor_permissions=FOUNDER,
        )


async def test_one_of_two_founders_may_be_removed():
    db = FakeSession()
    keep = _seed(db, ORG_A, Role.FOUNDER)
    drop = _seed(db, ORG_A, Role.FOUNDER)
    await MemberService(db).remove(
        org_id=ORG_A, user_id=drop, actor_id=keep, actor_permissions=FOUNDER
    )
    rows = await MemberService(db).list_members(ORG_A)
    assert [r.user_id for r in rows] == [keep]


async def test_founder_may_demote_self_when_another_founder_remains():
    db = FakeSession()
    _seed(db, ORG_A, Role.FOUNDER)
    me = _seed(db, ORG_A, Role.FOUNDER)
    await MemberService(db).assign_role(
        org_id=ORG_A, user_id=me, role=Role.EMPLOYEE,
        actor_id=me, actor_permissions=FOUNDER,
    )
    roles = sorted(r.role.value for r in await MemberService(db).list_members(ORG_A))
    assert roles == ["employee", "founder"]


# ── Fail-closed default ──────────────────────────────────────────────────────

async def test_remove_without_actor_permissions_denies():
    """A caller that forgets to pass `actor_permissions` must fail CLOSED rather
    than inherit the old unrestricted behaviour."""
    db = FakeSession()
    target = _seed(db, ORG_A, Role.EMPLOYEE)
    with pytest.raises(MemberForbidden):
        await MemberService(db).remove(org_id=ORG_A, user_id=target, actor_id=ACTOR)


async def test_unrecognised_target_role_denies():
    """A role value this build does not know maps to no permissions, which would
    make it look like zero privilege and freely removable. Deny instead."""
    db = FakeSession()
    uid = uuid.uuid4()
    db.add(User(id=uid, email="ghost@x.com"))
    m = Membership(user_id=uid, organization_id=ORG_A, role=Role.EMPLOYEE)
    db.add(m)
    m.role = "sovereign"  # type: ignore[assignment]  — a value added ahead of the enum
    with pytest.raises(MemberForbidden):
        await MemberService(db).remove(
            org_id=ORG_A, user_id=uid, actor_id=ACTOR, actor_permissions=FOUNDER
        )
