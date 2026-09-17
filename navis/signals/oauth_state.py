"""OAuth state: CSRF protection and authorization-code binding.

MOVED here verbatim from `navis.signals.router`, unchanged, when the Microsoft
connector needed the same primitive. Not reimplemented and not copied: this is
the mechanism that stops an attacker feeding our callback an attribution of
their choosing, and two copies of it is two chances for one of them to drift
into being subtly wrong while still looking right.

The properties that matter, all of which the Google flow already depended on:

  * The state string is 256 bits of `secrets.token_urlsafe` randomness and
    carries NO tenant data. It is an opaque handle, so a caller cannot encode
    "connect this as org X" into it.

  * The binding (org, user, provider) lives SERVER-SIDE in Redis. The callback
    resolves attribution only from this store, never from its own query string.
    That is what makes a forged callback URL useless.

  * Consumption is atomic fetch-and-delete (GETDEL), so a state is single-use
    and a replayed callback misses.

  * Both sides fail CLOSED. If Redis is unreachable we refuse to start the flow
    and refuse to validate one, rather than proceeding without CSRF protection —
    an OAuth callback that cannot check state is worse than an OAuth flow that
    is temporarily unavailable.

  * Missing, expired and already-used states are indistinguishable to the
    caller on purpose: distinguishing them tells a prober which of its guesses
    was previously valid.
"""
from __future__ import annotations

import json
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status
from redis.exceptions import RedisError

from navis.platform.cache import redis_client
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("signals.oauth_state")

OAUTH_STATE_PREFIX = "oauth:state:"
OAUTH_STATE_TTL = 600  # 10 minutes


async def create_oauth_state(
    *, org_id: uuid.UUID, user_id: uuid.UUID, provider: str
) -> str:
    """Mint a 256-bit random state and bind (org, user, provider, created_at) to it in
    Redis with a TTL. The state string itself carries NO tenant data — it's an opaque,
    unguessable handle, so the callback can't be fed attacker-chosen attribution."""
    state = secrets.token_urlsafe(32)
    payload = json.dumps({
        "org_id": str(org_id),
        "user_id": str(user_id),
        "provider": provider,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    try:
        await redis_client.set(f"{OAUTH_STATE_PREFIX}{state}", payload, ex=OAUTH_STATE_TTL)
    except RedisError as exc:  # fail CLOSED: no CSRF token store -> refuse to start OAuth
        logger.error("oauth state store unavailable: %r", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "oauth temporarily unavailable")
    return state


async def consume_oauth_state(state: str | None) -> dict[str, Any]:
    """Atomically fetch-and-delete the state (GETDEL) — single-use, so replays of a used
    or expired state miss and are rejected. Returns the bound attribution dict."""
    if not state:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "missing oauth state")
    try:
        raw = await redis_client.getdel(f"{OAUTH_STATE_PREFIX}{state}")
    except RedisError as exc:  # fail CLOSED: can't validate CSRF -> reject
        logger.error("oauth state validate failed: %r", exc)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid oauth state")
    if raw is None:
        # missing / expired (TTL) / already-used (single-use) — indistinguishable on purpose
        metrics.incr("signals.oauth.state_rejected")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid, expired, or used oauth state")
    return json.loads(raw)
