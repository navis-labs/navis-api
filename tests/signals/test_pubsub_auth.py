"""P1-2: Pub/Sub OIDC verification — accept genuine Google pushes, reject forgeries."""
from __future__ import annotations

import pytest

from navis.signals.google.pubsub_auth import (
    PubSubAuthError,
    verify_claims,
    verify_pubsub_request,
)

AUD = "https://navis.example/api/v1/webhooks/google/pubsub"
SA = "navis-push@proj.iam.gserviceaccount.com"


def _valid_claims():
    return {"iss": "accounts.google.com", "aud": AUD, "email": SA, "email_verified": True}


def test_valid_claims_pass():
    verify_claims(_valid_claims(), audience=AUD, sa_email=SA)  # no raise


def test_bad_issuer_rejected():
    c = _valid_claims() | {"iss": "evil.com"}
    with pytest.raises(PubSubAuthError):
        verify_claims(c, audience=AUD, sa_email=SA)


def test_audience_mismatch_rejected():
    with pytest.raises(PubSubAuthError):
        verify_claims(_valid_claims(), audience="https://other/endpoint", sa_email=SA)


def test_service_account_mismatch_rejected():
    with pytest.raises(PubSubAuthError):
        verify_claims(_valid_claims(), audience=AUD, sa_email="attacker@evil.com")


def test_unverified_email_rejected():
    c = _valid_claims() | {"email_verified": False}
    with pytest.raises(PubSubAuthError):
        verify_claims(c, audience=AUD, sa_email=SA)


def test_request_missing_bearer_rejected():
    with pytest.raises(PubSubAuthError):
        verify_pubsub_request(None, audience=AUD, sa_email=SA, decoder=lambda t, a: _valid_claims())
    with pytest.raises(PubSubAuthError):
        verify_pubsub_request("Basic xyz", audience=AUD, sa_email=SA, decoder=lambda t, a: _valid_claims())


def test_request_valid_token_passes():
    claims = verify_pubsub_request(
        "Bearer good.jwt.token",
        audience=AUD,
        sa_email=SA,
        decoder=lambda t, a: _valid_claims(),
    )
    assert claims["email"] == SA


def test_request_bad_signature_rejected():
    def bad_decoder(token, audience):
        raise ValueError("signature invalid")

    with pytest.raises(PubSubAuthError):
        verify_pubsub_request("Bearer forged", audience=AUD, sa_email=SA, decoder=bad_decoder)
