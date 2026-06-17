"""Optional Sentry error reporting. No-op unless SENTRY_DSN is set and the SDK
is installed — so it never becomes a hard dependency."""
from __future__ import annotations

from navis.platform.config import settings
from navis.platform.monitoring.logging import get_logger

logger = get_logger("monitoring")


def init_sentry() -> bool:
    if not settings.sentry_dsn:
        return False
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment,
            release=settings.app_version,
            traces_sample_rate=0.0,  # enable when you want perf tracing
        )
        logger.info("sentry initialized (env=%s)", settings.environment)
        return True
    except Exception:  # pragma: no cover - sentry optional
        logger.warning("SENTRY_DSN set but sentry-sdk unavailable; error reporting off")
        return False
