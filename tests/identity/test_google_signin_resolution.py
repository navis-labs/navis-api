"""Google sign-in must resolve an EXISTING account, never create a second one.

Regression cover for a Developer Console report: signing into the console
with a Google account that already had a Navis organization appeared to
resolve "the wrong organization". The investigation found the identity
chain was correct — the console was displaying a different column — but it
also found that nothing tested the property the whole report depended on:

    the same Google account must resolve to the same user, the same
    membership and the same organization, creating nothing.

These tests pin that. They exercise `IdentityService.google_signin`
against the in-memory session the other Identity unit tests use, so they
run without Postgres.

The org-CREATION path (a genuinely new Google account) is deliberately not
covered here: it writes through `OrgLifecycleSeam` and the audit service,
which the fake session does not model. It is exercised end-to-end in the
Postgres integration suite.
"""
from __future__ import annotations

import uuid

import pytest

from navis.identity.models import Membership, Organization, OrganizationKind, Role, User
from navis.identity.service import IdentityService
from tests.identity.fakes import FakeSession

EMAIL = "founder@example.test"
GOOGLE_SUB = "1095_stable_google_subject"


def _seed(db: FakeSession, *, google_sub: str | None) -> tuple[User, Organization, Membership]:
    """One user, one organization, one membership — the ordinary account."""
    user = User(
        id=uuid.uuid4(),
        email=EMAIL,
        full_name="A Founder",
        google_sub=google_sub,
        is_active=True,
        is_verified=True,
    )
    org = Organization(id=uuid.uuid4(), name="A Founder", slug="a-founder-1234abcd")
    membership = Membership(
        id=uuid.uuid4(), user_id=user.id, organization_id=org.id, role=Role.FOUNDER
    )
    db.add(user)
    db.add(org)
    db.add(membership)
    return user, org, membership


@pytest.mark.asyncio
async def test_a_returning_google_account_resolves_its_existing_user_and_org():
    """The property the Developer Console depends on: nothing is created."""
    db = FakeSession()
    user, org, membership = _seed(db, google_sub=GOOGLE_SUB)
    users_before = len(db.store.get(User, []))
    orgs_before = len(db.store.get(Organization, []))
    memberships_before = len(db.store.get(Membership, []))

    resolved, org_id, is_new_user, _onboarding = await IdentityService(db).google_signin(
        email=EMAIL, google_sub=GOOGLE_SUB, full_name="A Founder"
    )

    # Same canonical ids — not merely the same names.
    assert resolved.id == user.id
    assert org_id == org.id
    assert org_id == membership.organization_id
    assert is_new_user is False

    # Nothing was created. This is the assertion the report was really about.
    assert len(db.store.get(User, [])) == users_before
    assert len(db.store.get(Organization, [])) == orgs_before
    assert len(db.store.get(Membership, [])) == memberships_before


@pytest.mark.asyncio
async def test_a_returning_account_ignores_the_workspace_arguments_entirely():
    """P0-5 CASE A, and the property that makes one Google button safe.

    The console cannot know whether a Google click is a sign-in or a sign-up
    until the Core answers, so it sends the workspace choice every time. That
    is only safe if a returning account cannot be affected by it. Here the
    caller asks for a PERSONAL workspace called something else entirely, and
    the account resolves to exactly the company workspace it already had.

    If this ever fails, a Sarvam engineer signing in with the personal button
    gets a second tenant with none of their organization's data in it.
    """
    db = FakeSession()
    user, org, membership = _seed(db, google_sub=GOOGLE_SUB)
    orgs_before = len(db.store.get(Organization, []))

    resolved, org_id, is_new_user, _ = await IdentityService(db).google_signin(
        email=EMAIL,
        google_sub=GOOGLE_SUB,
        full_name="A Founder",
        workspace_kind=OrganizationKind.PERSONAL,
        organization_name="Some Other Company",
    )

    assert resolved.id == user.id
    assert org_id == org.id == membership.organization_id
    assert is_new_user is False
    # Not renamed, not reclassified, not duplicated.
    assert org.name == "A Founder"
    assert org.kind == OrganizationKind.COMPANY.value
    assert len(db.store.get(Organization, [])) == orgs_before


@pytest.mark.asyncio
async def test_the_same_account_resolves_identically_every_time():
    """Two sign-ins in a row must not diverge — no drift, no second org."""
    db = FakeSession()
    _seed(db, google_sub=GOOGLE_SUB)
    service = IdentityService(db)

    first_user, first_org, _, _ = await service.google_signin(
        email=EMAIL, google_sub=GOOGLE_SUB, full_name="A Founder"
    )
    second_user, second_org, is_new_user, _ = await service.google_signin(
        email=EMAIL, google_sub=GOOGLE_SUB, full_name="A Founder"
    )

    assert first_user.id == second_user.id
    assert first_org == second_org
    assert is_new_user is False
    assert len(db.store.get(Organization, [])) == 1


@pytest.mark.asyncio
async def test_a_password_account_signing_in_with_google_is_linked_not_duplicated():
    """Email match backfills `google_sub` on the EXISTING user.

    Without this, a founder who signed up with a password and later used
    Google would get a second user and a second organization — which is
    exactly the failure the console report suspected.
    """
    db = FakeSession()
    user, org, _ = _seed(db, google_sub=None)

    resolved, org_id, is_new_user, _ = await IdentityService(db).google_signin(
        email=EMAIL, google_sub=GOOGLE_SUB, full_name="A Founder"
    )

    assert resolved.id == user.id
    assert resolved.google_sub == GOOGLE_SUB  # linked, not duplicated
    assert org_id == org.id
    assert is_new_user is False
    assert len(db.store.get(User, [])) == 1
    assert len(db.store.get(Organization, [])) == 1


@pytest.mark.asyncio
async def test_the_current_organization_is_the_oldest_membership():
    """The canonical multi-org rule, pinned.

    `primary_membership` orders by `created_at` then `id`, so a user with
    several seats resolves to the same one on every request — and the
    Developer API, which reads the same membership row, cannot disagree
    with the application about which organization is current.
    """
    db = FakeSession()
    user, first_org, first_membership = _seed(db, google_sub=GOOGLE_SUB)

    second_org = Organization(id=uuid.uuid4(), name="Second", slug="second-9876fedc")
    second_membership = Membership(
        id=uuid.uuid4(), user_id=user.id, organization_id=second_org.id, role=Role.EMPLOYEE
    )
    db.add(second_org)
    db.add(second_membership)

    _resolved, org_id, _, _ = await IdentityService(db).google_signin(
        email=EMAIL, google_sub=GOOGLE_SUB, full_name="A Founder"
    )

    assert org_id == first_membership.organization_id
    assert org_id == first_org.id
    assert org_id != second_org.id


@pytest.mark.asyncio
async def test_a_disabled_account_is_refused_rather_than_re_created():
    from navis.identity.service import AuthError

    db = FakeSession()
    user, _org, _ = _seed(db, google_sub=GOOGLE_SUB)
    user.is_active = False

    with pytest.raises(AuthError):
        await IdentityService(db).google_signin(
            email=EMAIL, google_sub=GOOGLE_SUB, full_name="A Founder"
        )

    assert len(db.store.get(User, [])) == 1


# ── Sign-in must not provision a tenant (P0-8) ──────────────────────────
#
# The console's sign-in page has one Google button, and until P0-8 a click on
# it by somebody with no Navis account created a `company` organization named
# after them, made them its founder and its Developer owner, and told them
# nothing. Three of the four organizations in production were created that
# way. `create_workspace=False` is what distinguishes "sign me in" from
# "found a company"; these pin both halves of it.


@pytest.mark.asyncio
async def test_sign_in_refuses_an_unknown_identity_instead_of_founding_a_company():
    from navis.identity.service import NoSuchAccountError

    db = FakeSession()
    # Deliberately empty: no user, no organization, no invitation.

    with pytest.raises(NoSuchAccountError):
        await IdentityService(db).google_signin(
            email="nobody@example.com",
            google_sub="a_subject_navis_has_never_seen",
            full_name="Nobody",
            create_workspace=False,
        )

    # The whole point: the refusal is not merely a different status code, it
    # is the absence of three writes.
    assert db.store.get(User, []) == []
    assert db.store.get(Organization, []) == []
    assert db.store.get(Membership, []) == []


@pytest.mark.asyncio
async def test_an_invited_person_signs_in_normally_with_creation_disabled():
    """The flow `create_workspace=False` must NOT break.

    `MemberService.invite` writes a real `users` row with no password and no
    google_sub, so an invitee is found by the EMAIL lookup and resolves to the
    organization that invited them. If the refusal were keyed on "has never
    signed in" rather than "has no user row", this is the person it would lock
    out of the organization that just invited them.
    """
    db = FakeSession()
    invited = User(id=uuid.uuid4(), email=EMAIL, full_name=None, is_active=True)
    org = Organization(id=uuid.uuid4(), name="Inviting Company", slug="inviting-co-1234abcd")
    membership = Membership(
        id=uuid.uuid4(), user_id=invited.id, organization_id=org.id, role=Role.EMPLOYEE
    )
    db.add(invited)
    db.add(org)
    db.add(membership)

    resolved, org_id, is_new_user, _ = await IdentityService(db).google_signin(
        email=EMAIL, google_sub=GOOGLE_SUB, full_name="An Invitee", create_workspace=False
    )

    assert resolved.id == invited.id
    assert org_id == org.id
    assert is_new_user is False
    # Their Google subject is linked to the stub the invite created, rather
    # than a second account being made for the same address.
    assert resolved.google_sub == GOOGLE_SUB
    assert len(db.store.get(Organization, [])) == 1


@pytest.mark.asyncio
async def test_creation_remains_the_default_for_every_existing_caller():
    """`create_workspace` defaults to True, so the Application is unchanged.

    Asserted on the SIGNATURE rather than by creating an org, because the
    create branch writes through `OrgLifecycleSeam` and the audit service,
    which `FakeSession` does not model (see this module's docstring).
    """
    import inspect

    default = inspect.signature(IdentityService.google_signin).parameters[
        "create_workspace"
    ].default
    assert default is True
