"""Credential expiration — the lifecycle gap the Developer audit found.

WHAT THIS PINS. A Navis API key was valid from creation until somebody revoked
it, so a key that leaked into a CI log stayed usable until a human noticed.
`api_keys.expires_at` bounds that, and these tests hold the four properties that
make it safe to ship:

  1. EXPIRY IS ENFORCED SERVER-SIDE, in `ApiKeyService.check` — the single
     function every authenticated request passes through. Not in a router, not
     in the console.
  2. NULL STILL MEANS FOREVER. Every credential minted before 0052 has a NULL
     expiry and must keep working, or the migration is an outage.
  3. EXPIRED AND REVOKED STAY DISTINGUISHABLE. They are different facts with
     different follow-ups — a forgotten rotation versus a deliberate cutoff —
     and an audit that cannot tell them apart cannot explain either.
  4. THE REFUSAL LEAKS NOTHING. A guessing caller cannot learn from the answer
     whether a prefix exists.

The DECISION lives in `check`; `verify` delegates to it. Both are exercised
here, because `verify` is what the one production caller used before this change
and its contract must not have moved.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from navis.identity.agents import (
    CREDENTIAL_EXPIRY_WARNING,
    CredentialStatus,
    credential_status_of,
)
from navis.identity.apikeys import ApiKeyService, CredentialRefusal
from tests.identity.fakes import ORG_A, FakeSession

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


async def _mint(db, **kw):
    """A credential in ORG_A. Returns (row, plaintext)."""
    return await ApiKeyService(db).create(
        org_id=ORG_A, name=kw.pop("name", "probe"), scopes=["view_memory"], **kw
    )


# ── 1. Enforcement lives in the credential service ──────────────────────


async def test_a_credential_with_no_expiry_authenticates():
    """THE COMPATIBILITY PROPERTY. Every key that existed before 0052 has a
    NULL expiry; if this fails, the migration took the platform down."""
    db = FakeSession()
    row, plaintext = await _mint(db)
    assert row.expires_at is None

    checked = await ApiKeyService(db).check(plaintext)
    assert checked.key is row
    assert checked.refusal is None
    assert await ApiKeyService(db).verify(plaintext) is row


async def test_a_future_expiry_authenticates():
    db = FakeSession()
    row, plaintext = await _mint(db, expires_at=datetime.now(timezone.utc) + timedelta(days=30))
    assert row.expires_at is not None
    assert await ApiKeyService(db).verify(plaintext) is row


async def test_an_expired_credential_is_refused():
    """The point of the whole change."""
    db = FakeSession()
    row, plaintext = await _mint(db)
    # Set the expiry PAST the guard in `create`, which correctly refuses to mint
    # a credential that is already dead. Ageing a row is what time does.
    row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    checked = await ApiKeyService(db).check(plaintext)
    assert checked.key is None
    assert checked.refusal is CredentialRefusal.EXPIRED
    assert await ApiKeyService(db).verify(plaintext) is None


async def test_expiry_is_evaluated_at_the_boundary_not_rounded():
    """A credential is usable until its expiry, and not after it."""
    db = FakeSession()
    row, plaintext = await _mint(db)

    row.expires_at = datetime.now(timezone.utc) + timedelta(seconds=30)
    assert await ApiKeyService(db).verify(plaintext) is row, "still inside its life"

    row.expires_at = datetime.now(timezone.utc)
    assert await ApiKeyService(db).verify(plaintext) is None, "expiry reached is expired"


async def test_a_naive_expiry_does_not_crash_the_auth_path():
    """The column is timestamptz so this cannot come from the database — but it
    can come from a fixture, and a TypeError on the authentication path would be
    a 500 where a decision belongs."""
    db = FakeSession()
    row, plaintext = await _mint(db)
    row.expires_at = datetime.utcnow() - timedelta(days=1)  # noqa: DTZ003 — the point
    assert await ApiKeyService(db).verify(plaintext) is None


async def test_minting_an_already_expired_credential_is_refused():
    """A credential born dead authenticates nothing, so creating one is only
    ever a support ticket. Refused in the service, under all three routes."""
    db = FakeSession()
    with pytest.raises(ValueError, match="must be in the future"):
        await _mint(db, expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))


# ── 2. Expired is not revoked ───────────────────────────────────────────


async def test_revoked_and_expired_are_reported_as_different_refusals():
    db = FakeSession()
    svc = ApiKeyService(db)

    revoked_row, revoked_key = await _mint(db, name="revoked")
    await svc.revoke(revoked_row.id, org_id=ORG_A)

    expired_row, expired_key = await _mint(db, name="expired")
    expired_row.expires_at = datetime.now(timezone.utc) - timedelta(days=1)

    assert (await svc.check(revoked_key)).refusal is CredentialRefusal.REVOKED
    assert (await svc.check(expired_key)).refusal is CredentialRefusal.EXPIRED
    # Both refused. The DISTINCTION is for the operator, not the outcome.
    assert await svc.verify(revoked_key) is None
    assert await svc.verify(expired_key) is None


async def test_revocation_outranks_expiry_when_a_row_carries_both():
    """Revocation is a decision somebody made and is terminal. Reporting
    "expired" for a key an operator deliberately cut off would misattribute why
    it stopped."""
    db = FakeSession()
    svc = ApiKeyService(db)
    row, plaintext = await _mint(db)
    row.expires_at = datetime.now(timezone.utc) - timedelta(days=2)
    await svc.revoke(row.id, org_id=ORG_A)

    assert (await svc.check(plaintext)).refusal is CredentialRefusal.REVOKED
    assert credential_status_of(row.revoked_at, row.expires_at) == CredentialStatus.REVOKED.value


async def test_expiry_does_not_revoke_the_row_or_touch_history():
    """Expiry is a property of the credential, not an event that rewrites it.
    The row, its agent binding and its scopes stay readable for audit."""
    db = FakeSession()
    agent_id = uuid.uuid4()
    row, plaintext = await _mint(db, service_account_id=agent_id)
    row.expires_at = datetime.now(timezone.utc) - timedelta(days=1)

    assert await ApiKeyService(db).verify(plaintext) is None
    assert row.revoked_at is None, "expiry must not set revoked_at"
    assert row.service_account_id == agent_id
    assert row.scopes == ["view_memory"]
    assert row.prefix and row.hashed_secret


async def test_an_expired_credential_does_not_get_its_last_used_at_bumped():
    """`last_used_at` means "last authenticated". A refused request did not."""
    db = FakeSession()
    row, plaintext = await _mint(db)
    row.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    row.last_used_at = None

    await ApiKeyService(db).verify(plaintext)
    assert row.last_used_at is None


# ── 3. The refusal leaks nothing to a guesser ───────────────────────────


async def test_an_unknown_or_malformed_key_is_indistinguishable():
    """`malformed` and `unknown` are what a guessing client produces. Neither
    may reveal whether a prefix exists; `authz` maps both to one 401."""
    db = FakeSession()
    svc = ApiKeyService(db)
    assert (await svc.check("garbage")).refusal is CredentialRefusal.MALFORMED
    assert (await svc.check("navis_0000000000000000_nope")).refusal is CredentialRefusal.UNKNOWN


async def test_a_wrong_secret_on_a_real_prefix_is_unknown_not_expired():
    """The digest compare runs BEFORE the lifecycle checks, so holding a valid
    prefix and guessing the secret cannot learn a real credential's state."""
    db = FakeSession()
    row, _ = await _mint(db)
    row.expires_at = datetime.now(timezone.utc) - timedelta(days=1)

    forged = f"navis_{row.prefix}_wrongsecret"
    assert (await ApiKeyService(db).check(forged)).refusal is CredentialRefusal.UNKNOWN


# ── 4. The status derivation the API and console share ──────────────────


@pytest.mark.parametrize(
    "revoked_at, expires_at, expected",
    [
        (None, None, "active"),
        (None, NOW + timedelta(days=30), "active"),
        (None, NOW + CREDENTIAL_EXPIRY_WARNING, "expiring"),
        (None, NOW + timedelta(days=1), "expiring"),
        (None, NOW, "expired"),
        (None, NOW - timedelta(days=1), "expired"),
        (NOW - timedelta(days=1), None, "revoked"),
        (NOW - timedelta(days=1), NOW + timedelta(days=30), "revoked"),
    ],
)
def test_credential_status_covers_every_combination(revoked_at, expires_at, expected):
    assert credential_status_of(revoked_at, expires_at, now=NOW) == expected


def test_expiring_still_authenticates():
    """`expiring` is a PROMPT TO ROTATE, not a refusal. If it ever became one,
    every credential would fail a week early."""
    soon = NOW + timedelta(days=1)
    assert credential_status_of(None, soon, now=NOW) == CredentialStatus.EXPIRING.value
    from navis.identity.apikeys import _is_expired

    assert _is_expired(soon, NOW) is False
