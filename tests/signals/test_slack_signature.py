"""Slack webhook signature: accept genuine deliveries, reject forgeries + replays."""
from __future__ import annotations

import hashlib
import hmac

import pytest

from navis.signals.slack.signature import SlackSignatureError, verify_slack_signature

SECRET = "8f742231b10e8888abcd99yyyzzz85a5"
TS = "1700000000"
BODY = b'{"type":"event_callback","team_id":"T1","event_id":"Ev1"}'


def _sig(secret: str = SECRET, ts: str = TS, body: bytes = BODY) -> str:
    base = b"v0:" + ts.encode() + b":" + body
    return "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()


def test_valid_signature_passes():
    verify_slack_signature(body=BODY, timestamp=TS, signature=_sig(),
                           signing_secret=SECRET, now=float(TS) + 5)  # no raise


def test_tampered_body_rejected():
    with pytest.raises(SlackSignatureError) as e:
        verify_slack_signature(body=BODY + b"x", timestamp=TS, signature=_sig(),
                               signing_secret=SECRET, now=float(TS) + 5)
    assert e.value.reason == "bad_signature"


def test_stale_timestamp_rejected():
    with pytest.raises(SlackSignatureError) as e:
        verify_slack_signature(body=BODY, timestamp=TS, signature=_sig(),
                               signing_secret=SECRET, now=float(TS) + 10_000)
    assert e.value.reason == "stale_timestamp"


def test_missing_headers_rejected():
    with pytest.raises(SlackSignatureError) as e:
        verify_slack_signature(body=BODY, timestamp=None, signature=None, signing_secret=SECRET)
    assert e.value.reason == "missing_headers"


def test_unconfigured_secret_rejected():
    with pytest.raises(SlackSignatureError) as e:
        verify_slack_signature(body=BODY, timestamp=TS, signature=_sig(), signing_secret=None)
    assert e.value.reason == "not_configured"


def test_future_timestamp_rejected_distinctly_from_stale():
    # P0: a future-dated timestamp is its OWN anomaly, not "stale".
    with pytest.raises(SlackSignatureError) as e:
        verify_slack_signature(body=BODY, timestamp=TS, signature=_sig(),
                               signing_secret=SECRET, now=float(TS) - 600)  # ts is 10min ahead
    assert e.value.reason == "future_timestamp"
    # small skew is tolerated
    verify_slack_signature(body=BODY, timestamp=TS, signature=_sig(),
                           signing_secret=SECRET, now=float(TS) - 30)  # within 60s skew -> ok


def test_unsupported_signature_version_rejected():
    # P1: fail closed on a non-v0 scheme before comparing.
    with pytest.raises(SlackSignatureError) as e:
        verify_slack_signature(body=BODY, timestamp=TS, signature="v1=deadbeef",
                               signing_secret=SECRET, now=float(TS) + 5)
    assert e.value.reason == "bad_version"


def test_secret_rotation_any_secret_validates(monkeypatch):
    # P2: a signature made with the OLD secret still verifies while the new one leads the list.
    import navis.platform.config as cfg

    old_secret = "old-secret-value"
    sig_old = _sig(secret=old_secret)
    monkeypatch.setattr(cfg.settings, "slack_signing_secrets", f"new-secret,{old_secret}")
    verify_slack_signature(body=BODY, timestamp=TS, signature=sig_old, now=float(TS) + 5)  # no raise


def test_metrics_emitted_on_success_and_failure(monkeypatch):
    from navis.platform.monitoring.metrics import metrics

    seen: list[tuple] = []
    monkeypatch.setattr(metrics, "incr", lambda name, **k: seen.append((name, k.get("labels"))))

    verify_slack_signature(body=BODY, timestamp=TS, signature=_sig(), signing_secret=SECRET, now=float(TS) + 5)
    assert ("slack.signature.success", None) in seen

    seen.clear()
    with pytest.raises(SlackSignatureError):
        verify_slack_signature(body=BODY + b"x", timestamp=TS, signature=_sig(), signing_secret=SECRET, now=float(TS) + 5)
    assert ("slack.signature.failure", {"reason": "bad_signature"}) in seen


def test_health_helper(monkeypatch):
    import navis.platform.config as cfg

    from navis.signals.slack.signature import slack_signature_configured

    monkeypatch.setattr(cfg.settings, "slack_signing_secret", None)
    monkeypatch.setattr(cfg.settings, "slack_signing_secrets", None)
    assert slack_signature_configured() is False
    monkeypatch.setattr(cfg.settings, "slack_signing_secrets", "s1,s2")
    assert slack_signature_configured() is True
