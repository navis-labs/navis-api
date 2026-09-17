"""Microsoft (Entra) sign-in: claim handling and the account-linking rule.

The linking rule is the reason this file exists. Entra's `email` claim is
attacker-controlled for personal Microsoft accounts, so linking a Microsoft
identity to an existing Navis account on an email match — which is what the
Google path does, safely, because Google only asserts verified emails — is an
account-takeover primitive here.

Every test below that mentions verification is guarding that boundary.
"""
from __future__ import annotations

import base64
import json

import pytest

from navis.identity.microsoft_auth import (
    CONSUMER_TENANT_ID,
    MICROSOFT_SIGNIN_SCOPES,
    MicrosoftAuthError,
    _decode_claims,
    _email_from_claims,
    _email_verified,
)
from tests.identity.fakes import FakeSession


def _id_token(**claims) -> str:
    """Build an unsigned JWT-shaped token. The service reads the payload without
    verifying the signature — safe ONLY because the token is fetched by the
    server from Entra over TLS, never accepted from a client."""
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    return f"header.{payload.decode()}.signature"


# ---- xms_edov: the verification signal --------------------------------------


def test_absent_verification_claim_is_not_verified():
    """The claim is OPTIONAL and must be configured on the app registration.

    Defaulting to True when it is missing would silently reinstate the takeover
    path on any deployment that forgot to configure it — the failure would be
    invisible until someone exploited it. Absent means unverified.
    """
    assert _email_verified({}) is False
    assert _email_verified({"email": "ceo@target.com"}) is False


def test_verification_claim_accepts_the_spellings_entra_uses():
    assert _email_verified({"xms_edov": True}) is True
    assert _email_verified({"xms_edov": "true"}) is True
    assert _email_verified({"xms_edov": "1"}) is True
    assert _email_verified({"xms_edov": 1}) is True


def test_falsey_verification_claim_is_not_verified():
    assert _email_verified({"xms_edov": False}) is False
    assert _email_verified({"xms_edov": "false"}) is False
    assert _email_verified({"xms_edov": "0"}) is False
    assert _email_verified({"xms_edov": None}) is False
    assert _email_verified({"xms_edov": "yes-ish"}) is False


# ---- claim extraction -------------------------------------------------------


def test_email_claim_preferred_over_upn():
    claims = {"email": "Founder@Contoso.com", "preferred_username": "other@contoso.com"}
    assert _email_from_claims(claims) == "founder@contoso.com"


def test_upn_used_only_when_it_looks_like_an_address():
    """In a work tenant `preferred_username` is the UPN, which is usually an
    email and sometimes is not one at all. A non-address value would become
    half of the account identity and quietly split one person across two rows.
    """
    assert _email_from_claims({"preferred_username": "founder@contoso.com"}) == "founder@contoso.com"
    assert _email_from_claims({"preferred_username": "CONTOSO\\founder"}) is None
    assert _email_from_claims({}) is None


def test_decode_rejects_a_malformed_token():
    with pytest.raises(MicrosoftAuthError):
        _decode_claims("not-a-jwt")


def test_decode_reads_a_well_formed_token():
    claims = _decode_claims(_id_token(oid="o-1", tid="t-1", email="a@b.com"))
    assert claims["oid"] == "o-1"


# ---- scope hygiene ----------------------------------------------------------


def test_signin_never_requests_mailbox_scopes():
    """A mailbox scope reaching the sign-in consent screen is a product
    regression, not a config one — it turns a sign-up into a security review
    and collapses the two-step trust ramp the whole feature is built on."""
    joined = " ".join(MICROSOFT_SIGNIN_SCOPES).lower()
    for forbidden in ("mail.", "calendars.", "files.", "chat.", "channelmessage"):
        assert forbidden not in joined
    assert "openid" in joined


# ---- the linking rule -------------------------------------------------------


def _user(email: str, **over):
    from navis.identity.models import User

    kwargs = {"email": email, "is_active": True}
    kwargs.update(over)
    return User(**kwargs)


async def test_unverified_email_cannot_attach_to_an_existing_account():
    """The takeover this whole design exists to prevent.

    Anyone can create a free Microsoft account whose `email` claim carries a
    founder's work address. If that were enough to sign in as them, it would
    hand over the org's memory, decisions and connected mailboxes.
    """
    from navis.identity.service import AuthError, IdentityService

    db = FakeSession()
    db.add(_user("founder@contoso.com", google_sub="g-real"))
    svc = IdentityService(db)

    with pytest.raises(AuthError) as exc:
        await svc.microsoft_signin(
            email="founder@contoso.com",
            microsoft_sub="attacker-oid",
            tenant_id=CONSUMER_TENANT_ID,
            full_name="Not The Founder",
            email_verified=False,
        )
    # The message has to tell the real user what to do instead, or they are
    # simply locked out with no path forward.
    assert "settings" in str(exc.value).lower()


async def test_unverified_signin_does_not_write_the_sub_onto_the_victim():
    """Refusing is not enough if the refusal still backfilled the identifier —
    the next attempt would hit the microsoft_sub fast path and succeed."""
    from navis.identity.service import AuthError, IdentityService

    db = FakeSession()
    victim = _user("founder@contoso.com", google_sub="g-real")
    db.add(victim)
    svc = IdentityService(db)

    with pytest.raises(AuthError):
        await svc.microsoft_signin(
            email="founder@contoso.com",
            microsoft_sub="attacker-oid",
            tenant_id=CONSUMER_TENANT_ID,
            full_name=None,
            email_verified=False,
        )
    assert victim.microsoft_sub is None


async def test_verified_email_links_to_the_existing_account():
    """The legitimate case: a founder who signed up with Google and later signs
    in with their verified work Microsoft account lands on ONE Navis account."""
    from navis.identity.service import IdentityService

    db = FakeSession()
    existing = _user("founder@contoso.com", google_sub="g-real")
    db.add(existing)
    svc = IdentityService(db)

    user, _org_id, is_new_user, _done = await svc.microsoft_signin(
        email="founder@contoso.com",
        microsoft_sub="ms-oid-1",
        tenant_id="tenant-contoso",
        full_name="The Founder",
        email_verified=True,
    )
    assert user is existing
    assert is_new_user is False
    assert existing.microsoft_sub == "ms-oid-1"
    assert existing.microsoft_tid == "tenant-contoso"


async def test_known_sub_takes_the_fast_path_regardless_of_verification():
    """Once the identifier is on the account, the email claim is irrelevant —
    `oid` is immutable and not something the account holder can change, so it
    is the only lookup that is safe without a verification check."""
    from navis.identity.service import IdentityService

    db = FakeSession()
    existing = _user("founder@contoso.com", microsoft_sub="ms-oid-1")
    db.add(existing)
    svc = IdentityService(db)

    user, _org_id, is_new_user, _done = await svc.microsoft_signin(
        email="founder@contoso.com",
        microsoft_sub="ms-oid-1",
        tenant_id="tenant-contoso",
        full_name=None,
        email_verified=False,
    )
    assert user is existing
    assert is_new_user is False


async def test_disabled_account_is_refused():
    from navis.identity.service import AuthError, IdentityService

    db = FakeSession()
    db.add(_user("founder@contoso.com", microsoft_sub="ms-oid-1", is_active=False))
    svc = IdentityService(db)

    with pytest.raises(AuthError):
        await svc.microsoft_signin(
            email="founder@contoso.com",
            microsoft_sub="ms-oid-1",
            tenant_id="t",
            full_name=None,
            email_verified=True,
        )
