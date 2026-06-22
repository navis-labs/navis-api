"""Redis-backed brute-force / credential-stuffing protection for /auth/login.

Two independent counters per attempt — by email and by client IP — are incremented
atomically in Redis (INCR + sliding EXPIRE), so the throttle is multi-instance safe and
horizontally scalable with no shared in-process state. Once either counter reaches
`login_max_attempts` the identifier is locked until `login_lockout_seconds` of inactivity.

Fail-OPEN by design: any Redis error allows the login to proceed (availability over
enforcement) and emits a security metric. A Redis outage must never lock every user out
or break authentication. The email counter still defends a single account when an
attacker rotates IPs; the IP counter defends against one host spraying many accounts.

The service NEVER reveals lock state to the caller — callers map every failure (bad
password, unknown email, disabled account, locked) to one generic 401 (no enumeration).
"""
from __future__ import annotations

from redis.asyncio import Redis
from redis.exceptions import RedisError

from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("identity.throttle")


def _mask(email: str) -> str:
    """Mask an email for security logs (avoid persisting full PII in log streams)."""
    local, _, domain = email.partition("@")
    head = local[:1] if local else ""
    return f"{head}***@{domain}" if domain else "***"


class LoginThrottle:
    def __init__(self, redis: Redis, *, max_attempts: int | None = None, lockout_seconds: int | None = None) -> None:
        self.redis = redis
        self.max_attempts = max_attempts if max_attempts is not None else settings.login_max_attempts
        self.lockout_seconds = lockout_seconds if lockout_seconds is not None else settings.login_lockout_seconds

    @staticmethod
    def _keys(email: str, ip: str) -> tuple[str, str]:
        return (f"login:fail:email:{email.strip().lower()}", f"login:fail:ip:{ip}")

    async def is_locked(self, email: str, ip: str) -> bool:
        """True if either the email or IP counter has reached the threshold.

        Fail-open: on any Redis error, return False (allow) and emit a security metric."""
        try:
            ek, ik = self._keys(email, ip)
            ev, iv = await self.redis.get(ek), await self.redis.get(ik)
            email_hits = int(ev) if ev is not None else 0
            ip_hits = int(iv) if iv is not None else 0
            locked = email_hits >= self.max_attempts or ip_hits >= self.max_attempts
            if locked:
                metrics.incr("login.locked")
                logger.warning("login.locked email=%s ip=%s", _mask(email), ip)
            return locked
        except RedisError:
            metrics.incr("login.throttle.redis_error", labels={"op": "check"})
            logger.warning("login.throttle redis unavailable on check — failing open")
            return False

    async def record_failure(self, email: str, ip: str) -> None:
        """Increment both counters atomically and (re)set the sliding lock window."""
        try:
            for key in self._keys(email, ip):
                await self.redis.incr(key)
                await self.redis.expire(key, self.lockout_seconds)
            metrics.incr("login.failed")
            logger.info("login.failed email=%s ip=%s", _mask(email), ip)
        except RedisError:
            metrics.incr("login.throttle.redis_error", labels={"op": "record"})
            logger.warning("login.throttle redis unavailable on record — failing open")

    async def reset(self, email: str, ip: str) -> None:
        """Clear both counters after a successful login. Emits login.unlocked if the
        identifier had accumulated failures (i.e. a real unlock occurred)."""
        try:
            cleared = await self.redis.delete(*self._keys(email, ip))
            if cleared:
                metrics.incr("login.unlocked")
                logger.info("login.unlocked email=%s ip=%s", _mask(email), ip)
        except RedisError:
            metrics.incr("login.throttle.redis_error", labels={"op": "reset"})
