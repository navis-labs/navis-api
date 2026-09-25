"""P1-6 — the Developer access model, flow by flow.

WHAT THIS FILE PROVES. Application and Developer are two surfaces over ONE
organization, ONE authorization evaluator and ONE shared Core. A member's
Developer access comes from `memberships.developer_role` and from nothing else;
it is never inherited from an Application role, never implied by organization
membership, and never acquired by omission.

The sections map onto the required flows:

    1  Developer solo            personal org, creator is Developer owner
    2  Developer-first company   company org, creator is Developer owner
    3  Developer team            explicit roles per teammate
    4  Existing Application org  Developer access must be granted, not inherited
    5  Application + Developer    one org, one core, additive permissions

WHY MOST OF IT IS PURE. `effective_permissions` is a pure function and both role
maps are module-level dicts, so the authorization model can be exercised without
a database, a token or an event loop. The flows that write rows use `FakeSession`
— the same in-memory session the rest of `tests/identity` uses. The full HTTP
path with real Postgres is `tests/integration/`.

NOTHING HERE TOUCHES THE CORE. No test in this file imports `navis.orchestration`,
`navis.state`, `navis.memory` or `navis.api.chat_router`, because the access
model has no reach into them — which is itself asserted, once, in section 7.
"""
from __future__ import annotations

import inspect
import uuid

import pytest

from navis.identity.members import (
    MemberConflict,
    MemberForbidden,
    MemberService,
)
from navis.identity.models import DeveloperRole, Membership, OrganizationKind, Role, User
from navis.identity.permissions import (
    APPLICATION_PERMISSIONS,
    DEVELOPER_PERMISSIONS,
    DEVELOPER_ROLE_PERMISSIONS,
    Permission,
    default_permissions_for_role,
    developer_grant_key,
    effective_permissions,
)
from navis.identity.schemas import SignupRequest
from navis.identity.service import IdentityService
from tests.identity.fakes import FakeSession

ORG_A = uuid.uuid4()
ORG_B = uuid.uuid4()

#: A founder's full resolution — what signup writes and what `0053` backfilled.
FOUNDER = effective_permissions("founder", developer_role="owner")
#: An Application administrator with no Developer access.
APP_ADMIN = default_permissions_for_role("admin")


def _seed(
    db,
    org_id,
    role: Role,
    developer_role: DeveloperRole | None = None,
) -> uuid.UUID:
    """Add a member to `org_id`; returns their user id."""
    uid = uuid.uuid4()
    db.add(User(id=uid, email=f"{uid.hex[:8]}@x.com"))
    db.add(
        Membership(
            user_id=uid,
            organization_id=org_id,
            role=role,
            developer_role=developer_role.value if developer_role else None,
        )
    )
    return uid


async def _membership(db, org_id, user_id) -> Membership:
    rows = await MemberService(db).list_members(org_id)
    return next(m for m in rows if m.user_id == user_id)


# ═══════════════════════════════════════════════════════════════════════════
# 1. FLOW: DEVELOPER SOLO  —  signs up only for Developer
# ═══════════════════════════════════════════════════════════════════════════


async def test_solo_signup_creates_a_personal_org_whose_creator_is_developer_owner():
    """Flow 1, end to end at the service layer.

    Omitting `organization_name` is the signal for a personal workspace (P0-4),
    and the creator comes out as a Developer owner — so a solo developer can
    register an agent and call the Developer API immediately, with no
    Application onboarding anywhere in the path.
    """
    db = FakeSession()
    user, org, membership = await IdentityService(db).signup(
        SignupRequest(email="solo@p16probe.example.com", password="Str0ngPass!x")
    )

    assert org.kind == OrganizationKind.PERSONAL.value
    assert membership.role == Role.FOUNDER
    assert membership.developer_role == DeveloperRole.OWNER.value

    resolved = effective_permissions(
        membership.role.value, developer_role=membership.developer_role
    )
    assert Permission.MANAGE_SERVICE_ACCOUNTS in resolved
    assert Permission.VIEW_STATE in resolved
    assert Permission.RECORD_AGENT_RUNS in resolved


async def test_a_solo_developer_needs_no_application_onboarding():
    """Nothing in the Developer authorization path consults onboarding stage.

    Pinned structurally rather than by simulating a stage, because the claim is
    about what the resolver READS: if `OnboardingStage` never reaches it, no
    onboarding state can gate Developer access.
    """
    from navis.identity import authz, permissions

    for module in (authz, permissions):
        source = inspect.getsource(module)
        assert "OnboardingStage" not in source
        assert "onboarding" not in source.lower().replace("onboarding_completed", "")


# ═══════════════════════════════════════════════════════════════════════════
# 2. FLOW: DEVELOPER-FIRST COMPANY
# ═══════════════════════════════════════════════════════════════════════════


async def test_company_signup_creates_a_company_org_whose_creator_is_developer_owner():
    """Flow 2. Naming an organization produces a company workspace, and its
    founder is a Developer owner — so a Developer-first company can be set up
    and staffed without the Application product being configured at all."""
    db = FakeSession()
    _user, org, membership = await IdentityService(db).signup(
        SignupRequest(
            email="founder@p16probe.example.com",
            password="Str0ngPass!x",
            organization_name="Dignatara Space",
        )
    )

    assert org.kind == OrganizationKind.COMPANY.value
    assert org.name == "Dignatara Space"
    assert membership.developer_role == DeveloperRole.OWNER.value


async def test_a_google_signup_may_state_a_developer_first_company():
    """The Google edge reaches the same place. `workspace_kind` is consulted only
    on the create branch (P0-5), and the membership it writes is a Developer
    owner for the same bootstrap reason the password path is."""
    db = FakeSession()
    user, org_id, is_new, _completed = await IdentityService(db).google_signin(
        email="g@p16probe.example.com",
        google_sub="sub-p16-1",
        full_name="G Founder",
        workspace_kind=OrganizationKind.COMPANY,
        organization_name="Sarvam",
    )
    assert is_new is True
    membership = await _membership(db, org_id, user.id)
    assert membership.role == Role.FOUNDER
    assert membership.developer_role == DeveloperRole.OWNER.value


async def test_a_returning_google_account_resolves_its_existing_organization():
    """No second organization, ever — the P0-5 invariant, re-asserted here
    because the Developer flows are the ones most likely to break it (a
    Developer-first sign-in looks exactly like a sign-up to the client)."""
    db = FakeSession()
    svc = IdentityService(db)
    user1, org1, _new, _c = await svc.google_signin(
        email="returning@p16probe.example.com",
        google_sub="sub-p16-2",
        full_name="R",
        workspace_kind=OrganizationKind.COMPANY,
        organization_name="First Co",
    )
    user2, org2, is_new, _c = await svc.google_signin(
        email="returning@p16probe.example.com",
        google_sub="sub-p16-2",
        full_name="R",
        # A different name and kind — must be ignored on the sign-in branch.
        workspace_kind=OrganizationKind.PERSONAL,
        organization_name="Second Co",
    )
    assert is_new is False
    assert user2.id == user1.id
    assert org2 == org1


# ═══════════════════════════════════════════════════════════════════════════
# 3. FLOW: DEVELOPER TEAM  —  founder + developer + developer + viewer
# ═══════════════════════════════════════════════════════════════════════════


async def test_a_founder_can_staff_a_developer_team_with_explicit_roles():
    """Flow 3. Each teammate's Developer access is stated at invite time, and
    each resolves to exactly their own Developer role's permissions — not the
    founder's, and not each other's."""
    db = FakeSession()
    svc = MemberService(db)
    actor = uuid.uuid4()

    dev1 = await svc.invite(
        org_id=ORG_A, email="dev1@x.com", role=Role.EMPLOYEE, actor_id=actor,
        actor_permissions=FOUNDER, developer_role=DeveloperRole.DEVELOPER,
    )
    viewer = await svc.invite(
        org_id=ORG_A, email="viewer@x.com", role=Role.EMPLOYEE, actor_id=actor,
        actor_permissions=FOUNDER, developer_role=DeveloperRole.VIEWER,
    )

    assert dev1.developer_role == "developer"
    assert viewer.developer_role == "viewer"

    dev_perms = effective_permissions("employee", developer_role="developer")
    view_perms = effective_permissions("employee", developer_role="viewer")
    assert Permission.RECORD_AGENT_RUNS in dev_perms
    assert Permission.RECORD_AGENT_RUNS not in view_perms
    assert Permission.MANAGE_SERVICE_ACCOUNTS not in dev_perms
    assert Permission.MANAGE_SERVICE_ACCOUNTS not in view_perms


async def test_an_invite_without_a_developer_role_grants_no_developer_access():
    """Developer access is never acquired by omission. This is the default an
    Application-shaped invite produces, and the SCIM provisioning path too."""
    db = FakeSession()
    m = await MemberService(db).invite(
        org_id=ORG_A, email="appmember@x.com", role=Role.EMPLOYEE,
        actor_id=uuid.uuid4(), actor_permissions=FOUNDER,
    )
    assert m.developer_role is None
    resolved = effective_permissions("employee", developer_role=m.developer_role)
    assert resolved.isdisjoint(DEVELOPER_PERMISSIONS)


async def test_a_developer_role_can_be_granted_then_revoked():
    """The whole lifecycle, including revocation — which is `developer_role=None`
    and needs no separate mechanism."""
    db = FakeSession()
    svc = MemberService(db)
    uid = _seed(db, ORG_A, Role.EMPLOYEE)

    granted = await svc.assign_developer_role(
        org_id=ORG_A, user_id=uid, developer_role=DeveloperRole.DEVELOPER,
        actor_id=uuid.uuid4(), actor_permissions=FOUNDER,
    )
    assert granted.developer_role == "developer"

    revoked = await svc.assign_developer_role(
        org_id=ORG_A, user_id=uid, developer_role=None,
        actor_id=uuid.uuid4(), actor_permissions=FOUNDER,
    )
    assert revoked.developer_role is None
    assert effective_permissions("employee", developer_role=None).isdisjoint(
        DEVELOPER_PERMISSIONS
    )


async def test_revoking_developer_access_revokes_the_member_s_live_sessions():
    """A NARROWING MUST TAKE EFFECT NOW, not within the 30s Principal cache TTL.

    `revoke_all_for_user` is the existing lever; the assertion is that losing
    Developer access pulls it, because a member who has just been cut off from
    the Developer surface must not keep reading it for another half minute.
    """
    db = FakeSession()
    svc = MemberService(db)
    uid = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.ADMIN)

    user = await db.get(User, uid)
    assert user.tokens_valid_from is None

    await svc.assign_developer_role(
        org_id=ORG_A, user_id=uid, developer_role=DeveloperRole.VIEWER,
        actor_id=uuid.uuid4(), actor_permissions=FOUNDER,
    )
    assert (await db.get(User, uid)).tokens_valid_from is not None


async def test_widening_developer_access_does_not_sign_the_member_out():
    """The converse. Interrupting somebody's session to give them MORE access is
    a gratuitous logout, and the TTL picks a widening up on its own."""
    db = FakeSession()
    uid = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.VIEWER)

    await MemberService(db).assign_developer_role(
        org_id=ORG_A, user_id=uid, developer_role=DeveloperRole.ADMIN,
        actor_id=uuid.uuid4(), actor_permissions=FOUNDER,
    )
    assert (await db.get(User, uid)).tokens_valid_from is None


# ═══════════════════════════════════════════════════════════════════════════
# 4. FLOW: EXISTING APPLICATION ORGANIZATION
# ═══════════════════════════════════════════════════════════════════════════


def test_an_application_role_alone_reaches_no_developer_permission():
    """Flow 4's central requirement. Every Application role, checked."""
    for role in ("founder", "ceo", "coo", "chief_of_staff", "admin", "employee"):
        assert default_permissions_for_role(role).isdisjoint(DEVELOPER_PERMISSIONS), (
            f"{role} inherits Developer authority"
        )


async def test_an_application_admin_cannot_grant_themselves_developer_access():
    """THE ESCALATION THIS MODEL MUST REFUSE, and it needed no new rule.

    An Application `admin` holds `manage_roles`, so they reach the assignment
    route. `_ensure_grantable` then requires them to already hold every
    permission the assignment would confer — and since the split they hold no
    Developer permission at all. So the answer is 403 for every Developer role,
    including the weakest one, and including when the target is themselves.
    """
    db = FakeSession()
    svc = MemberService(db)
    admin_uid = _seed(db, ORG_A, Role.ADMIN)

    for role in (
        DeveloperRole.VIEWER, DeveloperRole.DEVELOPER,
        DeveloperRole.ADMIN, DeveloperRole.OWNER,
    ):
        with pytest.raises(MemberForbidden):
            await svc.assign_developer_role(
                org_id=ORG_A, user_id=admin_uid, developer_role=role,
                actor_id=admin_uid, actor_permissions=APP_ADMIN,
            )
    assert (await _membership(db, ORG_A, admin_uid)).developer_role is None


async def test_an_application_admin_cannot_grant_developer_access_to_anyone():
    """Not just to themselves. The bound is on the ACTOR's own permissions, so
    the target's identity does not change the answer."""
    db = FakeSession()
    svc = MemberService(db)
    admin_uid = _seed(db, ORG_A, Role.ADMIN)
    victim = _seed(db, ORG_A, Role.EMPLOYEE)

    with pytest.raises(MemberForbidden):
        await svc.assign_developer_role(
            org_id=ORG_A, user_id=victim, developer_role=DeveloperRole.VIEWER,
            actor_id=admin_uid, actor_permissions=APP_ADMIN,
        )


async def test_an_application_admin_cannot_smuggle_developer_access_into_an_invite():
    """The other door into the same escalation. One `_ensure_grantable` covers
    the COMBINED assignment, so the Developer half is bounded even though the
    Application half (`employee`) is well within an admin's gift."""
    db = FakeSession()
    with pytest.raises(MemberForbidden):
        await MemberService(db).invite(
            org_id=ORG_A, email="smuggled@x.com", role=Role.EMPLOYEE,
            actor_id=uuid.uuid4(), actor_permissions=APP_ADMIN,
            developer_role=DeveloperRole.DEVELOPER,
        )


async def test_an_application_admin_cannot_act_on_a_developer():
    """Target-aware authorization, extended to the Developer surface. An actor
    who holds no Developer authority cannot remove — or re-role — a member who
    does, because the target's permissions are not a subset of theirs."""
    db = FakeSession()
    svc = MemberService(db)
    admin_uid = _seed(db, ORG_A, Role.ADMIN)
    developer = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.DEVELOPER)

    with pytest.raises(MemberForbidden):
        await svc.remove(
            org_id=ORG_A, user_id=developer, actor_id=admin_uid,
            actor_permissions=APP_ADMIN,
        )
    with pytest.raises(MemberForbidden):
        await svc.assign_developer_role(
            org_id=ORG_A, user_id=developer, developer_role=None,
            actor_id=admin_uid, actor_permissions=APP_ADMIN,
        )


async def test_an_existing_application_founder_reaches_developer_after_the_backfill():
    """Flow 4 for the role every production membership actually holds.

    `0053_developer_access` sets `developer_role='owner'` for founders, and the
    resolution is then identical to what a founder had before the split. This is
    the test the migration's docstring cites.
    """
    db = FakeSession()
    uid = _seed(db, ORG_A, Role.FOUNDER, DeveloperRole.OWNER)
    m = await _membership(db, ORG_A, uid)
    resolved = effective_permissions(
        m.role.value, developer_role=m.developer_role
    )
    assert Permission.MANAGE_SERVICE_ACCOUNTS in resolved
    # Every permission in the vocabulary — 19 at P1-6, 20 since `manage_own_agents`
    # joined at P1-7. Asserted against the vocabulary rather than a literal so
    # this test tracks it instead of pinning a number that has to be edited.
    from navis.identity.permissions import VALID_PERMISSIONS

    assert resolved == VALID_PERMISSIONS


def test_a_founder_keeps_exactly_the_permissions_they_had_before_the_split():
    """The regression the whole change hinges on, named where the migration
    docstring points at it."""
    from navis.identity.permissions import VALID_PERMISSIONS

    assert effective_permissions("founder", developer_role="owner") == VALID_PERMISSIONS


async def test_an_existing_application_member_gets_developer_access_by_grant():
    """Flow 4's other half: a non-founder is NOT backfilled, and reaches the
    Developer surface only once a Developer owner grants it."""
    db = FakeSession()
    svc = MemberService(db)
    member = _seed(db, ORG_A, Role.CEO)

    assert (await _membership(db, ORG_A, member)).developer_role is None
    assert default_permissions_for_role("ceo").isdisjoint(DEVELOPER_PERMISSIONS)

    await svc.assign_developer_role(
        org_id=ORG_A, user_id=member, developer_role=DeveloperRole.VIEWER,
        actor_id=uuid.uuid4(), actor_permissions=FOUNDER,
    )
    m = await _membership(db, ORG_A, member)
    assert m.developer_role == "viewer"
    assert Permission.VIEW_STATE in effective_permissions(
        "ceo", developer_role=m.developer_role
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. FLOW: APPLICATION + DEVELOPER, one organization
# ═══════════════════════════════════════════════════════════════════════════


def test_the_surfaces_are_additive_and_never_subtractive():
    """One member, two surfaces, one organization. Holding a Developer role
    cannot remove an Application permission, and vice versa."""
    app_only = effective_permissions("ceo")
    dev_only = effective_permissions("employee", developer_role="developer")
    both = effective_permissions("ceo", developer_role="developer")

    assert app_only <= both
    assert DEVELOPER_ROLE_PERMISSIONS["developer"] <= both
    # The Application half is untouched by the Developer grant.
    assert both & APPLICATION_PERMISSIONS == app_only
    # And the Developer half is untouched by the Application role.
    assert both & DEVELOPER_PERMISSIONS == dev_only & DEVELOPER_PERMISSIONS


def test_an_application_only_member_holds_only_application_permissions():
    assert effective_permissions("ceo") <= APPLICATION_PERMISSIONS


def test_a_developer_only_member_holds_no_application_authority():
    """Step 8: a Developer must not acquire billing, membership or SSO powers."""
    resolved = effective_permissions("employee", developer_role="admin")
    for permission in (
        Permission.MANAGE_BILLING, Permission.MANAGE_MEMBERS,
        Permission.MANAGE_ROLES, Permission.MANAGE_SSO,
        Permission.EXECUTE_ACTIONS, Permission.MANAGE_INTEGRATIONS,
        Permission.VIEW_AUDIT,
    ):
        assert permission not in resolved


# ═══════════════════════════════════════════════════════════════════════════
# 6. GOVERNANCE — the surface must stay reachable
# ═══════════════════════════════════════════════════════════════════════════


async def test_the_last_developer_owner_cannot_be_demoted():
    """Without this an organization could lose every member able to grant
    Developer access, and the surface would be permanently closed to it."""
    db = FakeSession()
    uid = _seed(db, ORG_A, Role.FOUNDER, DeveloperRole.OWNER)

    # The owner demoting THEMSELVES, which is the realistic way an
    # organization loses its last one. Since P1-8 only a Developer owner may
    # touch the owner role at all, so an actor with no membership would now be
    # refused by succession (403) before governance (409) could answer — and
    # the rule under test here is the governance one.
    with pytest.raises(MemberConflict):
        await MemberService(db).assign_developer_role(
            org_id=ORG_A, user_id=uid, developer_role=DeveloperRole.ADMIN,
            actor_id=uid, actor_permissions=FOUNDER,
        )


async def test_the_last_developer_owner_cannot_be_removed():
    db = FakeSession()
    uid = _seed(db, ORG_A, Role.CEO, DeveloperRole.OWNER)

    with pytest.raises(MemberConflict):
        await MemberService(db).remove(
            org_id=ORG_A, user_id=uid, actor_id=uuid.uuid4(),
            actor_permissions=FOUNDER,
        )


async def test_a_second_owner_releases_the_governance_lock():
    """The rule is "keep at least one", not "the first one is permanent"."""
    db = FakeSession()
    svc = MemberService(db)
    first = _seed(db, ORG_A, Role.FOUNDER, DeveloperRole.OWNER)
    second = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.OWNER)

    # Demoted BY THE OTHER OWNER — succession (P1-8) requires the actor to be
    # a Developer owner, and here there genuinely is a second one.
    demoted = await svc.assign_developer_role(
        org_id=ORG_A, user_id=first, developer_role=DeveloperRole.ADMIN,
        actor_id=second, actor_permissions=FOUNDER,
    )
    assert demoted.developer_role == "admin"


async def test_an_organization_with_no_developer_owner_is_not_blocked():
    """An Application-only tenant is a legitimate state; the governance guard
    must not fire where there is nothing to protect."""
    db = FakeSession()
    uid = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.VIEWER)

    revoked = await MemberService(db).assign_developer_role(
        org_id=ORG_A, user_id=uid, developer_role=None,
        actor_id=uuid.uuid4(), actor_permissions=FOUNDER,
    )
    assert revoked.developer_role is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. THE CORE IS UNTOUCHED, AND THE MODEL CANNOT REACH IT
# ═══════════════════════════════════════════════════════════════════════════


def test_the_context_composer_takes_no_authorization_input():
    """Application authorization changes cannot enter the Core, structurally.

    `build_grounded_context` is the ONE task-aware composer and both `/chat` and
    `POST /developer/context` call it. It accepts an organization and a question;
    it has no parameter through which a role, a surface or a permission could
    reach it, so no change to the access model can alter what it computes.
    """
    from navis.orchestration.context import build_grounded_context

    params = set(inspect.signature(build_grounded_context).parameters)
    for banned in (
        "principal", "role", "developer_role", "application_role",
        "permission", "permissions", "surface", "kind",
    ):
        assert banned not in params

    source = inspect.getsource(build_grounded_context)
    for banned in (
        "Principal", "developer_role", "application_role",
        "effective_permissions", "DEVELOPER_ROLE_PERMISSIONS",
    ):
        assert banned not in source


def test_the_chat_route_gate_is_unchanged():
    """`/chat` is capability-gated and names no permission. The surface split
    must not have added one — an Application `employee` still legitimately uses
    the product."""
    from navis.api import chat_router

    source = inspect.getsource(chat_router)
    assert "require_permission" not in source
    assert "developer_role" not in source


def test_no_application_route_gained_a_permission_gate():
    """The split removed nine permissions from the Application role maps. That is
    only safe because no Application route gates on any of them — so this checks
    the live route table rather than trusting the claim.

    The four legacy `/identity/service-accounts` handlers are the DELIBERATE
    exception, recorded in `DEVELOPER_PERMISSIONS`: registering a machine
    identity is Developer authority whichever door it is done through.
    """
    import os

    os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
    os.environ.setdefault("SECRET_KEY", "x" * 40)
    from fastapi.routing import APIRoute

    from navis.identity.permissions import VALID_PERMISSIONS
    from navis.main import create_app

    def walk(fn, depth=0, seen=None):
        found = set()
        if seen is None:
            seen = set()
        if fn is None or id(fn) in seen or depth > 5:
            return found
        seen.add(id(fn))
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, str) and value in VALID_PERMISSIONS:
                found.add(value)
            elif callable(value):
                found |= walk(value, depth + 1, seen)
        return found

    allowed = {"/api/v1/identity/service-accounts"}
    offenders = []
    for route in create_app().routes:
        if not isinstance(route, APIRoute) or "/developer/" in route.path:
            continue
        gated = set()
        for dep in route.dependant.dependencies:
            gated |= walk(dep.call)
        gated |= walk(route.endpoint)
        leaked = gated & DEVELOPER_PERMISSIONS
        if leaked and not any(route.path.startswith(p) for p in allowed):
            offenders.append(f"{route.path} gates on {sorted(leaked)}")
    assert not offenders, "an Application route gates on Developer authority:\n" + "\n".join(
        offenders
    )


# ═══════════════════════════════════════════════════════════════════════════
# 8. AGENTS AND CREDENTIALS ARE UNCHANGED
# ═══════════════════════════════════════════════════════════════════════════


def test_a_machine_principal_has_no_developer_role_to_resolve():
    """An agent is a machine identity: its reach is `api_keys.scopes` and it
    holds no membership row. So the human surface split could not have changed
    what any existing credential can do — there is no path from a
    `developer_role` to a service-account principal."""
    from navis.identity import authz

    source = inspect.getsource(authz._principal_from_api_key)
    assert "developer_role" not in source
    assert "Membership" not in source
    assert "effective_permissions" not in source
    # And the field defaults to None on the dataclass, so a machine principal
    # reports "no Developer role" rather than a guess.
    assert authz.Principal.__dataclass_fields__["developer_role"].default is None


def test_agent_assignable_permissions_are_within_the_developer_surface():
    """A credential may be granted the Developer surface and nothing else — an
    agent has no business holding `manage_billing`.

    A SUBSET SINCE P1-8, NOT AN EQUALITY, and the difference is deliberate.
    This assertion used to read `==`, which bundled two separate rules:

        (a) no agent may hold an Application permission   — still absolute;
        (b) every Developer permission is agent-assignable — was true only
            because no HUMAN-ONLY Developer permission existed yet.

    `manage_developer_access` is the first, and must never be assignable: it
    administers people — who is on the Developer team and what access they
    hold — and a machine credential that could grant humans Developer access
    would be an escalation path out of the agent sandbox. Rule (a) is
    asserted below; rule (b) is replaced by the exception being named
    EXPLICITLY, so a future Developer permission that quietly becomes
    non-assignable still fails this test rather than slipping through a
    loosened `<=`.

    The "no Developer route is unreachable by an agent" half of the original
    docstring is not lost — it has its own test,
    `test_agent_assignable_matches_developer_routes`, which checks the
    permissions the routes actually gate on. No `/developer/` route gates on
    `manage_developer_access`, so that invariant is untouched.
    """
    from navis.identity.permissions import (
        AGENT_ASSIGNABLE_PERMISSIONS,
        Permission,
    )

    assert AGENT_ASSIGNABLE_PERMISSIONS <= DEVELOPER_PERMISSIONS
    assert DEVELOPER_PERMISSIONS - AGENT_ASSIGNABLE_PERMISSIONS == {
        Permission.MANAGE_DEVELOPER_ACCESS
    }


def test_an_agents_scopes_are_bounded_by_its_creator_s_developer_role():
    """`ensure_grantable_scopes` is unchanged and now carries the surface bound
    for free: an operator's grantable set IS their resolved permissions, so a
    Developer viewer cannot mint an agent that records runs."""
    from fastapi import HTTPException

    from navis.identity.authz import Principal, PrincipalKind, ensure_grantable_scopes

    viewer = Principal(
        kind=PrincipalKind.USER,
        org_id=ORG_A,
        subject_id=uuid.uuid4(),
        permissions=effective_permissions("employee", developer_role="viewer"),
        role="employee",
        developer_role="viewer",
    )
    # May grant what it holds.
    ensure_grantable_scopes([Permission.VIEW_STATE], viewer)
    # May not grant what it does not.
    for scope in (Permission.RECORD_AGENT_RUNS, Permission.MANAGE_SERVICE_ACCOUNTS):
        with pytest.raises(HTTPException) as raised:
            ensure_grantable_scopes([scope], viewer)
        assert raised.value.status_code == 403


# ═══════════════════════════════════════════════════════════════════════════
# 9. CROSS-ORGANIZATION AND GRANT NAMESPACING
# ═══════════════════════════════════════════════════════════════════════════


async def test_a_developer_role_in_one_org_does_not_reach_another():
    """Isolation is the same `organization_id` predicate for both surfaces. A
    member of ORG_A with Developer owner is simply not a member of ORG_B."""
    db = FakeSession()
    uid = _seed(db, ORG_A, Role.FOUNDER, DeveloperRole.OWNER)

    assert [m.user_id for m in await MemberService(db).list_members(ORG_B)] == []
    from navis.identity.members import MemberError

    with pytest.raises(MemberError):
        await MemberService(db).assign_developer_role(
            org_id=ORG_B, user_id=uid, developer_role=None,
            actor_id=uuid.uuid4(), actor_permissions=FOUNDER,
        )


def test_developer_grants_are_namespaced_away_from_the_legacy_vocabulary():
    """`role_permissions.role` is ONE column and three vocabularies share the
    strings 'admin', 'owner' and 'viewer'. Namespacing is what stops a per-org
    grant to "admin" applying to an Application admin AND a Developer admin."""
    from navis.identity.models import ApplicationRole

    assert developer_grant_key("admin") == "developer:admin"
    assert developer_grant_key(None) == ""

    legacy = {r.value for r in Role}
    application = {r.value for r in ApplicationRole}
    developer = {r.value for r in DeveloperRole}
    # The collision is real — this is why the prefix exists.
    assert developer & legacy
    assert developer & application
    # And the namespaced keys collide with nothing.
    namespaced = {developer_grant_key(r) for r in developer}
    assert namespaced.isdisjoint(legacy | application | developer)


def test_the_developer_role_map_matches_the_database_vocabulary():
    """`permissions.py` is a pure module and cannot import the ORM enum, so the
    two declarations of the Developer role vocabulary are checked against each
    other here. A value permitted by `ck_memberships_developer_role` but absent
    from the map would resolve to zero permissions — a silent denial."""
    assert set(DEVELOPER_ROLE_PERMISSIONS) == {r.value for r in DeveloperRole}


# ═══════════════════════════════════════════════════════════════════════════
# 8. P1-8 — A DEVELOPER ADMINISTRATOR CAN RUN THE DEVELOPER TEAM
# ═══════════════════════════════════════════════════════════════════════════
#
# The gap, measured: every route that administers Developer team membership
# was gated on an APPLICATION permission — `manage_members` for listing and
# inviting, `manage_roles` for assigning Developer access — and no Developer
# role holds either. A Developer owner who is an Application `employee` could
# not list their own team, invite an engineer, or grant Developer access. It
# was invisible because whoever SIGNS UP is `role=FOUNDER` and therefore holds
# both Application permissions by accident of being first.


def test_a_developer_administrator_can_administer_the_developer_team():
    """The permission exists and reaches exactly the two administrator roles."""
    for role in ("admin", "owner"):
        assert Permission.MANAGE_DEVELOPER_ACCESS in DEVELOPER_ROLE_PERMISSIONS[role]
    for role in ("developer", "viewer"):
        assert Permission.MANAGE_DEVELOPER_ACCESS not in DEVELOPER_ROLE_PERMISSIONS[role]


def test_a_developer_admin_needs_no_application_role_to_run_their_team():
    """THE WHOLE POINT OF P1-8, stated as the resolution a real member gets.

    An Application `employee` with `developer_role='admin'` — the ordinary
    shape of a tech lead at a Developer-first company — must reach Developer
    team administration, and must still reach NO Application administration.
    """
    lead = effective_permissions("employee", developer_role="admin")

    assert Permission.MANAGE_DEVELOPER_ACCESS in lead
    # ...and not one Application administration permission came with it.
    for withheld in (
        Permission.MANAGE_MEMBERS,
        Permission.MANAGE_ROLES,
        Permission.MANAGE_BILLING,
        Permission.MANAGE_INTEGRATIONS,
        Permission.MANAGE_API_KEYS,
        Permission.MANAGE_SSO,
        Permission.MANAGE_POLICIES,
    ):
        assert withheld not in lead


def test_an_application_role_alone_still_reaches_no_developer_team_administration():
    """The converse, which is the P1-6 rule this must not weaken."""
    for role in ("admin", "ceo", "coo", "chief_of_staff", "employee"):
        assert Permission.MANAGE_DEVELOPER_ACCESS not in default_permissions_for_role(role)


def test_developer_team_administration_is_never_assignable_to_an_agent():
    """A machine must not administer people.

    `manage_service_accounts` — the permission this one could most plausibly
    have reused — IS agent-assignable and a live agent key carries it. Reusing
    it would have let a credential grant humans Developer access, which is an
    escalation path out of the agent sandbox rather than a widening.
    """
    from navis.identity.permissions import AGENT_ASSIGNABLE_PERMISSIONS

    assert Permission.MANAGE_DEVELOPER_ACCESS not in AGENT_ASSIGNABLE_PERMISSIONS
    assert Permission.MANAGE_SERVICE_ACCOUNTS in AGENT_ASSIGNABLE_PERMISSIONS


async def test_a_developer_admin_can_grant_developer_access():
    """A tech lead brings a developer onto the Developer surface themselves."""
    db = FakeSession()
    lead = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.ADMIN)
    joiner = _seed(db, ORG_A, Role.EMPLOYEE, None)

    updated = await MemberService(db).assign_developer_role(
        org_id=ORG_A, user_id=joiner, developer_role=DeveloperRole.DEVELOPER,
        actor_id=lead,
        actor_permissions=effective_permissions("employee", developer_role="admin"),
    )
    assert updated.developer_role == "developer"


async def test_a_developer_admin_cannot_promote_anyone_to_owner():
    """SUCCESSION, and the escalation guard P1-8 required.

    Developer `admin` and `owner` hold identical permission SETS, so
    `_ensure_grantable`'s subset comparison cannot tell them apart — an admin
    reaching this route would otherwise pass every existing guard while
    promoting themselves to owner. Only an owner may confer the owner role.
    """
    db = FakeSession()
    lead = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.ADMIN)
    _owner = _seed(db, ORG_A, Role.FOUNDER, DeveloperRole.OWNER)
    admin_perms = effective_permissions("employee", developer_role="admin")

    # Promoting somebody else...
    other = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.DEVELOPER)
    with pytest.raises(MemberForbidden):
        await MemberService(db).assign_developer_role(
            org_id=ORG_A, user_id=other, developer_role=DeveloperRole.OWNER,
            actor_id=lead, actor_permissions=admin_perms,
        )

    # ...and promoting THEMSELVES, which is the escalation that matters.
    with pytest.raises(MemberForbidden):
        await MemberService(db).assign_developer_role(
            org_id=ORG_A, user_id=lead, developer_role=DeveloperRole.OWNER,
            actor_id=lead, actor_permissions=admin_perms,
        )


async def test_a_developer_admin_cannot_demote_an_owner():
    """The other direction of succession: an admin cannot remove the owner
    who is above them, which would otherwise be a way to take the surface."""
    db = FakeSession()
    lead = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.ADMIN)
    owner = _seed(db, ORG_A, Role.FOUNDER, DeveloperRole.OWNER)
    _spare = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.OWNER)  # governance is satisfied

    with pytest.raises(MemberForbidden):
        await MemberService(db).assign_developer_role(
            org_id=ORG_A, user_id=owner, developer_role=DeveloperRole.ADMIN,
            actor_id=lead,
            actor_permissions=effective_permissions("employee", developer_role="admin"),
        )


async def test_a_developer_owner_can_appoint_a_successor():
    """What Developer owner uniquely controls, now that it controls something."""
    db = FakeSession()
    owner = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.OWNER)
    heir = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.ADMIN)

    updated = await MemberService(db).assign_developer_role(
        org_id=ORG_A, user_id=heir, developer_role=DeveloperRole.OWNER,
        actor_id=owner,
        actor_permissions=effective_permissions("employee", developer_role="owner"),
    )
    assert updated.developer_role == "owner"


async def test_a_developer_owner_seat_in_another_org_confers_nothing_here():
    """Succession is scoped to the organization the actor is acting in."""
    db = FakeSession()
    outsider = _seed(db, ORG_B, Role.EMPLOYEE, DeveloperRole.OWNER)
    db.add(
        Membership(
            user_id=outsider, organization_id=ORG_A,
            role=Role.EMPLOYEE, developer_role=DeveloperRole.ADMIN.value,
        )
    )
    target = _seed(db, ORG_A, Role.EMPLOYEE, DeveloperRole.DEVELOPER)

    with pytest.raises(MemberForbidden):
        await MemberService(db).assign_developer_role(
            org_id=ORG_A, user_id=target, developer_role=DeveloperRole.OWNER,
            actor_id=outsider,
            actor_permissions=effective_permissions("employee", developer_role="admin"),
        )


def test_a_credential_can_never_carry_developer_team_administration():
    """THE SERVER RULE, not the console default.

    Caught in live verification: leaving `manage_developer_access` out of
    `AGENT_ASSIGNABLE_PERMISSIONS` was NOT sufficient, because that set is
    explicitly "a default for a management surface, not a server rule". A
    Developer owner holds the permission, so `ensure_grantable_scopes`'
    creator-bound check passed it and issued a credential that would then
    have satisfied the member-administration gates.

    The refusal is now at the minting chokepoint, and it applies to the
    OWNER — the one principal who holds the permission and could therefore
    pass every other check.
    """
    from fastapi import HTTPException

    from navis.identity.authz import Principal, PrincipalKind, ensure_grantable_scopes
    from navis.identity.permissions import HUMAN_ONLY_PERMISSIONS

    owner = Principal(
        kind=PrincipalKind.USER,
        org_id=ORG_A,
        subject_id=uuid.uuid4(),
        permissions=effective_permissions("founder", developer_role="owner"),
        role="founder",
    )
    assert Permission.MANAGE_DEVELOPER_ACCESS in owner.permissions

    with pytest.raises(HTTPException) as refused:
        ensure_grantable_scopes([Permission.MANAGE_DEVELOPER_ACCESS], owner)
    assert refused.value.status_code == 403

    # The rest of the Developer surface is still mintable by that owner —
    # this bounds one permission, it does not narrow agent credentials.
    ensure_grantable_scopes(
        [Permission.VIEW_STATE, Permission.RECORD_AGENT_RUNS,
         Permission.MANAGE_SERVICE_ACCOUNTS],
        owner,
    )
    assert HUMAN_ONLY_PERMISSIONS == {Permission.MANAGE_DEVELOPER_ACCESS}
