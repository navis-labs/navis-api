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

from navis.identity.models import Membership, Organization, Role, User
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
