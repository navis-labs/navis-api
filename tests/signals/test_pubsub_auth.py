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


def test_subject_pinning():
    # `sub` is Google's stable canonical identity — pin on it when configured.
    c = _valid_claims() | {"sub": "100200300"}
    verify_claims(c, audience=AUD, sa_email=SA, expected_sub="100200300")  # match -> ok
    with pytest.raises(PubSubAuthError) as e:
        verify_claims(c, audience=AUD, sa_email=SA, expected_sub="999")     # mismatch -> reject
    assert e.value.reason == "bad_subject"


def test_failure_carries_reason_for_metrics():
    with pytest.raises(PubSubAuthError) as e:
        verify_claims(_valid_claims() | {"iss": "evil.com"}, audience=AUD, sa_email=SA)
    assert e.value.reason == "bad_issuer"


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


# ---- TASK 6.4.2: fail-closed configuration ----------------------------------
#
# The endpoint used to SKIP verification entirely when GOOGLE_PUBSUB_AUDIENCE was
# unset, and fall through to a `?token=` check that was itself skipped when its own
# variable was unset. Two fail-opens in series left a public endpoint enqueueing
# Google sync work for anonymous callers. These pin the closed behaviour.


@pytest.fixture()
def unconfigured(monkeypatch):
    from navis.platform.config import settings

    monkeypatch.setattr(settings, "google_pubsub_audience", None)
    monkeypatch.setattr(settings, "google_pubsub_sa_email", None)
    monkeypatch.setattr(settings, "google_pubsub_sub", None)


@pytest.fixture()
def configured(monkeypatch):
    from navis.platform.config import settings

    monkeypatch.setattr(settings, "google_pubsub_audience", AUD)
    monkeypatch.setattr(settings, "google_pubsub_sa_email", SA)
    monkeypatch.setattr(settings, "google_pubsub_sub", None)


def test_missing_audience_refuses_instead_of_skipping(unconfigured):
    """A perfectly-signed Google token must NOT be accepted when we cannot say
    which audience it was supposed to be for."""
    with pytest.raises(PubSubAuthError) as exc:
        verify_pubsub_request("Bearer good", decoder=lambda t, a: _valid_claims())
    assert exc.value.reason == "not_configured"


def test_missing_service_identity_refuses(monkeypatch):
    """Audience alone is a PARTIAL configuration. Without a pinned service account,
    any Google-issued token carrying our audience would pass — including one minted
    for an attacker's own push subscription aimed at our URL."""
    from navis.platform.config import settings

    monkeypatch.setattr(settings, "google_pubsub_audience", AUD)
    monkeypatch.setattr(settings, "google_pubsub_sa_email", None)
    monkeypatch.setattr(settings, "google_pubsub_sub", None)
    with pytest.raises(PubSubAuthError) as exc:
        verify_pubsub_request("Bearer good", decoder=lambda t, a: _valid_claims())
    assert exc.value.reason == "not_configured"


def test_pubsub_auth_configured_requires_both_halves(monkeypatch):
    from navis.platform.config import settings
    from navis.signals.google.pubsub_auth import pubsub_auth_configured

    monkeypatch.setattr(settings, "google_pubsub_sub", None)
    for aud, sa, expected in (
        (None, None, False),
        (AUD, None, False),   # audience alone is NOT a configuration
        (None, SA, False),
        (AUD, SA, True),
    ):
        monkeypatch.setattr(settings, "google_pubsub_audience", aud)
        monkeypatch.setattr(settings, "google_pubsub_sa_email", sa)
        assert pubsub_auth_configured() is expected


def test_sub_alone_satisfies_service_identity(monkeypatch):
    """`sub` is the stable OIDC subject; pinning it INSTEAD of email is valid."""
    from navis.platform.config import settings
    from navis.signals.google.pubsub_auth import pubsub_auth_configured

    monkeypatch.setattr(settings, "google_pubsub_audience", AUD)
    monkeypatch.setattr(settings, "google_pubsub_sa_email", None)
    monkeypatch.setattr(settings, "google_pubsub_sub", "1234567890")
    assert pubsub_auth_configured() is True
    claims = _valid_claims() | {"sub": "1234567890"}
    verify_pubsub_request("Bearer good", decoder=lambda t, a: claims)  # no raise


def test_configured_path_still_accepts_a_genuine_push(configured):
    claims = verify_pubsub_request("Bearer good", decoder=lambda t, a: _valid_claims())
    assert claims["email"] == SA


def test_decoder_receives_the_audience_and_never_an_empty_string(configured):
    """`_default_decoder` treats a falsy audience as 'skip audience verification',
    so the audience must always reach it."""
    seen = {}

    def decoder(token, audience):
        seen["audience"] = audience
        return _valid_claims()

    verify_pubsub_request("Bearer good", decoder=decoder)
    assert seen["audience"] == AUD


def test_default_decoder_refuses_empty_audience():
    from navis.signals.google.pubsub_auth import _default_decoder

    with pytest.raises(PubSubAuthError) as exc:
        _default_decoder("token", "")
    assert exc.value.reason == "not_configured"
