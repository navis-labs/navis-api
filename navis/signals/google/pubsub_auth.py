"""Verify that a Pub/Sub push genuinely came from Google.

Google signs each push with an OIDC JWT in the `Authorization: Bearer <jwt>`
header. We verify the signature (Google certs), then the claims:
  * issuer is accounts.google.com,
  * audience matches our configured push audience,
  * the service-account email matches the one we authorized (if configured),
  * email_verified is true.

The signature decoder is injectable so the claim logic is unit-testable without
network access to Google's certs.
"""
from __future__ import annotations

from typing import Any, Callable

from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("signals.pubsub_auth")

_VALID_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}

Decoder = Callable[[str, str], dict[str, Any]]  # (token, audience) -> claims


class PubSubAuthError(Exception):
    """The push could not be authenticated as a genuine Google Pub/Sub request."""

    def __init__(self, message: str, *, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = reason  # stable label for metrics/alerting (bad_audience, bad_signature, ...)


def verify_claims(
    claims: dict[str, Any], *, audience: str | None, sa_email: str | None, expected_sub: str | None = None
) -> None:
    """App-level claim checks (signature/exp already verified by the decoder).

    The push service account is pinned on `sub` (Google's stable, canonical OIDC subject id)
    when configured — `email` can be reassigned, `sub` cannot — and on `email` for defence
    in depth.
    """
    if claims.get("iss") not in _VALID_ISSUERS:
        raise PubSubAuthError(f"bad issuer: {claims.get('iss')!r}", reason="bad_issuer")
    if audience and claims.get("aud") != audience:
        raise PubSubAuthError("audience mismatch", reason="bad_audience")
    if expected_sub and claims.get("sub") != expected_sub:
        raise PubSubAuthError("service-account subject mismatch", reason="bad_subject")
    if sa_email and claims.get("email") != sa_email:
        raise PubSubAuthError("service-account email mismatch", reason="bad_email")
    if claims.get("email") and claims.get("email_verified") is False:
        raise PubSubAuthError("service-account email not verified", reason="email_unverified")


# Mirror navis.identity.google_auth._CLOCK_SKEW_SECONDS — same drift
# tolerance applies to Pub/Sub push tokens. Without it a 1-2s skew
# between Google's signer and this host surfaces as silent webhook
# drops (a worse failure mode than a noisy 401 on user login).
_PUBSUB_CLOCK_SKEW_SECONDS = 10


def _default_decoder(token: str, audience: str) -> dict[str, Any]:  # pragma: no cover - network
    """Signature + exp + audience, via Google's own certificate discovery.

    `verify_oauth2_token` fetches and caches Google's public certs, selects the key
    by the token's `kid`, and rejects anything not signed by it — so an unsigned
    token (`alg: none`) and an algorithm-downgrade attempt both fail here, and no
    rotating key material is hard-coded.

    The audience is passed UNCONDITIONALLY. It used to be `audience or None`, and
    `None` tells this function to SKIP audience verification entirely — so an
    unset `GOOGLE_PUBSUB_AUDIENCE` silently downgraded the check to "signed by
    Google", which any Google-issued OIDC token satisfies. `verify_pubsub_request`
    now refuses to call this without an audience, and this signature no longer
    admits the empty string that produced that behaviour.
    """
    from google.auth.transport import requests as g_requests
    from google.oauth2 import id_token

    if not audience:
        raise PubSubAuthError(
            "refusing to verify without an audience", reason="not_configured"
        )
    return id_token.verify_oauth2_token(
        token,
        g_requests.Request(),
        audience,
        clock_skew_in_seconds=_PUBSUB_CLOCK_SKEW_SECONDS,
    )


def verify_pubsub_request(
    authorization_header: str | None,
    *,
    audience: str | None = None,
    sa_email: str | None = None,
    expected_sub: str | None = None,
    decoder: Decoder | None = None,
) -> dict[str, Any]:
    """Verify the bearer JWT and return its claims, or raise PubSubAuthError. Every outcome
    is metered (pubsub.auth.success / pubsub.auth.failure{reason}) so an attack — a burst of
    bad audience/signature/issuer — is visible instead of silent."""
    audience = audience or settings.google_pubsub_audience
    sa_email = sa_email or settings.google_pubsub_sa_email
    expected_sub = expected_sub or settings.google_pubsub_sub

    try:
        # FAIL CLOSED ON MISSING CONFIGURATION.
        #
        # This is the defect TASK 6.4.2 exists to close. Verification used to be
        # SKIPPED entirely when the audience was unset (the router asked
        # `pubsub_auth_configured()` first), which left a public endpoint that
        # enqueues Google sync work accepting anonymous POSTs. An unconfigured
        # verifier must refuse, never wave the request through — and the check
        # lives HERE as well as at the route so no future caller can reintroduce
        # the skip by forgetting to ask.
        #
        # Both halves are required. The audience proves the token was minted for
        # THIS endpoint; the identity proves it was minted for the service account
        # we authorized. Without the identity, any Google-issued token carrying our
        # audience would pass — including one an attacker obtained for their own
        # push subscription pointed at our URL.
        if not audience:
            raise PubSubAuthError(
                "Pub/Sub push authentication is not configured "
                "(GOOGLE_PUBSUB_AUDIENCE)",
                reason="not_configured",
            )
        if not (sa_email or expected_sub):
            raise PubSubAuthError(
                "Pub/Sub push authentication is not configured "
                "(GOOGLE_PUBSUB_SA_EMAIL or GOOGLE_PUBSUB_SUB)",
                reason="not_configured",
            )
        if not authorization_header or not authorization_header.lower().startswith("bearer "):
            raise PubSubAuthError("missing bearer token", reason="missing_bearer")
        token = authorization_header.split(" ", 1)[1].strip()

        decode = decoder or _default_decoder
        try:
            claims = decode(token, audience or "")
        except PubSubAuthError:
            raise
        except Exception as exc:  # signature/exp failure from the decoder
            raise PubSubAuthError(f"invalid token signature: {exc}", reason="bad_signature") from exc

        verify_claims(claims, audience=audience, sa_email=sa_email, expected_sub=expected_sub)
    except PubSubAuthError as exc:
        metrics.incr("pubsub.auth.failure", labels={"reason": exc.reason})
        raise
    metrics.incr("pubsub.auth.success")
    return claims


def pubsub_auth_configured() -> bool:
    """Is authenticated push FULLY provisioned?

    Both halves, deliberately: the audience alone is not a configuration, it is a
    partial one, and a partial configuration used to read as "configured" and
    disable the service-account check. The route uses this to answer 503 (not yet
    provisioned) rather than 401 (you are not Google) — a distinction an operator
    needs, and one that never widens what is accepted: `verify_pubsub_request`
    re-checks the same condition itself.
    """
    return bool(
        settings.google_pubsub_audience
        and (settings.google_pubsub_sa_email or settings.google_pubsub_sub)
    )
