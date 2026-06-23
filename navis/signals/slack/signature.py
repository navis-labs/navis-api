"""Verify that an inbound request genuinely came from Slack.

Slack signs every Events API delivery with HMAC-SHA256 over ``v0:{timestamp}:{body}`` using
the app signing secret, in the ``X-Slack-Signature`` header. We recompute it with
``hmac.compare_digest`` (constant-time) and reject any request whose
``X-Slack-Request-Timestamp`` is outside a short window — a captured-signature replay is
worthless once the timestamp ages out, and a future-dated timestamp is rejected outright.

The secret-matching core stays pure + dependency-free (unit-testable without HTTP); the public
entry point additionally emits security metrics + a no-PII warning log. Secrets and max age are
injected (default to settings). Rotation: SLACK_SIGNING_SECRETS may list several secrets,
newest first — any one validating accepts, so a secret rotates with zero downtime.
"""
from __future__ import annotations

import hashlib
import hmac
import time

from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("signals.slack.signature")

# Clock-skew tolerance for "not in the future" — a little leeway for unsynced clocks.
_FUTURE_SKEW_SECONDS = 60


class SlackSignatureError(Exception):
    """The request could not be authenticated as a genuine Slack delivery."""

    def __init__(self, message: str, *, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = reason  # stable metrics/alert label


def _signing_secrets(override: str | None) -> list[str]:
    """Resolve the ordered secret list. An explicit override wins (tests / per-call); else the
    rotation list (SLACK_SIGNING_SECRETS, newest first), else the single legacy secret."""
    if override:
        return [override]
    raw = settings.slack_signing_secrets or settings.slack_signing_secret or ""
    return [s.strip() for s in raw.split(",") if s.strip()]


def slack_signature_configured() -> bool:
    """Readiness check: at least one signing secret is configured."""
    return bool(_signing_secrets(None))


def _verify(
    *, body: bytes, timestamp: str | None, signature: str | None,
    signing_secret: str | None, max_age_seconds: int | None, now: float | None,
) -> None:
    secrets = _signing_secrets(signing_secret)
    if not secrets:
        raise SlackSignatureError("slack signing secret not configured", reason="not_configured")
    if not timestamp or not signature:
        raise SlackSignatureError("missing slack signature headers", reason="missing_headers")

    max_age = max_age_seconds if max_age_seconds is not None else settings.slack_webhook_max_age_seconds
    now = now if now is not None else time.time()
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        raise SlackSignatureError("non-integer timestamp", reason="bad_timestamp")
    # Reject future-dated timestamps (with small skew) separately from stale ones — a clock far
    # ahead is a different anomaly than an aged-out replay.
    if ts > now + _FUTURE_SKEW_SECONDS:
        raise SlackSignatureError("timestamp in future", reason="future_timestamp")
    if now - ts > max_age:
        raise SlackSignatureError("timestamp outside replay window", reason="stale_timestamp")

    # Fail closed on unknown signature versions (future Slack scheme) before comparing.
    if not signature.startswith("v0="):
        raise SlackSignatureError("unsupported signature version", reason="bad_version")

    basestring = b"v0:" + timestamp.encode() + b":" + body
    for secret in secrets:  # rotation: accept if ANY configured secret validates
        digest = hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()
        if hmac.compare_digest(f"v0={digest}", signature):
            return
    raise SlackSignatureError("signature mismatch", reason="bad_signature")


def verify_slack_signature(
    *,
    body: bytes,
    timestamp: str | None,
    signature: str | None,
    signing_secret: str | None = None,
    max_age_seconds: int | None = None,
    now: float | None = None,
) -> None:
    """Raise SlackSignatureError unless `signature` is a valid Slack HMAC over `body`. Emits
    security metrics (slack.signature.success / failure{reason}) and a no-PII warning on
    failure — never the body, headers, or secret."""
    try:
        _verify(
            body=body, timestamp=timestamp, signature=signature,
            signing_secret=signing_secret, max_age_seconds=max_age_seconds, now=now,
        )
    except SlackSignatureError as exc:
        metrics.incr("slack.signature.failure", labels={"reason": exc.reason})
        logger.warning("slack signature verification failed reason=%s", exc.reason)
        raise
    metrics.incr("slack.signature.success")
