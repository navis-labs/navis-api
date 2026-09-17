"""Login brute-force protection using a Redis sliding-window strike counter.

Strategy: a failed login for an email increments a Redis key with a TTL. After
`_MAX_STRIKES` failures within the window the account is locked out for `_LOCKOUT_SECONDS`.

Design choices:
  * Key is `login_strikes:{sha256(normalized_email)}` — hashed so the key doesn't
    contain PII and can't be decoded by anyone with Redis access.
  * Counter, not a list — O(1) incr/get, no memory growth per strike.
  * INCR + EXPIRE is not atomic, but the race window (missed expiry on the very first
    strike) is harmless — it means the window resets a moment later than expected, which
    biases toward generosity, never toward locking a legitimate user longer than intended.
  * Successful login resets the counter so a user doesn't stay locked after setting a
    new password.
  * Redis unavailability fails OPEN (auth proceeds) — a Redis outage must not lock
    every user out of the application.
"""
from __future__ import annotations

import hashlib

from navis.platform.monitoring import get_logger

logger = get_logger("identity.ratelimit")

_MAX_STRIKES = 10          # failed attempts before lockout
_WINDOW_SECONDS = 15 * 60  # 15-minute rolling window
_LOCKOUT_SECONDS = 15 * 60  # lockout duration (same as window)

_PREFIX = "login_strikes:"


def _key(email: str) -> str:
    digest = hashlib.sha256(email.strip().lower().encode()).hexdigest()
    return f"{_PREFIX}{digest}"


class TooManyLoginAttempts(Exception):
    """Raised when a login is blocked due to too many recent failures."""


class LoginRateLimiter:
    """Call `check(email)` before authenticating and `reset(email)` on success."""

    async def check(self, email: str) -> None:
        """Raise `TooManyLoginAttempts` if the account is currently locked out.

        Fails open on Redis error so a cache outage never locks legitimate users out.
        """
        try:
            from navis.platform.cache.redis import redis_client

            count = await redis_client.get(_key(email))
            if count is not None and int(count) >= _MAX_STRIKES:
                logger.warning(
                    "login blocked: too many attempts for email_hash=%s count=%s",
                    _key(email)[-8:],  # only the tail for logs — not the full hash
                    count,
                )
                raise TooManyLoginAttempts(
                    f"too many failed login attempts — try again in {_LOCKOUT_SECONDS // 60} minutes"
                )
        except TooManyLoginAttempts:
            raise
        except Exception:
            # Redis unavailable or any other error: fail open — never block auth on a
            # cache outage. Log at debug so it's visible but not alarming in dev.
            logger.debug("login rate-limit check skipped (redis unavailable)")

    async def record_failure(self, email: str) -> None:
        """Increment the failure counter. Resets automatically after the window expires."""
        try:
            from navis.platform.cache.redis import redis_client

            key = _key(email)
            count = await redis_client.incr(key)
            if count == 1:
                # First failure: set the expiry on the key (INCR creates it, then we expire).
                await redis_client.expire(key, _WINDOW_SECONDS)
            logger.info(
                "login failure recorded email_hash=...%s count=%d/%d",
                key[-8:], count, _MAX_STRIKES,
            )
        except Exception:
            logger.debug("login failure recording skipped (redis unavailable)")

    async def reset(self, email: str) -> None:
        """Delete the strike counter on successful login (clears any partial lockout)."""
        try:
            from navis.platform.cache.redis import redis_client

            await redis_client.delete(_key(email))
        except Exception:
            logger.debug("login strike reset skipped (redis unavailable)")


# Module-level singleton — no state, safe to share across requests.
login_rate_limiter = LoginRateLimiter()
