"""Gmail watch lifecycle helpers + persistable watch state.

Gmail push works by calling users.watch() to register a Cloud Pub/Sub topic;
Google then posts a notification (with a new historyId) whenever the mailbox
changes. Watches expire in ~7 days and MUST be renewed. The watch state
(historyId, expiration, renewal timestamp, topic) is persisted in
`Integration.sync_state["watch"]` (PostgreSQL) so the lifecycle survives restarts.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

# Renew when within this window of expiry.
WATCH_RENEW_BEFORE = timedelta(hours=24)


def build_watch_state(
    resp: dict[str, Any], *, topic: str | None, now: datetime | None = None
) -> dict[str, Any]:
    """Normalize a users.watch() response into persistable watch state.

    Gmail returns `expiration` as epoch milliseconds (string).
    """
    now = now or datetime.now(timezone.utc)
    expiration_ms = resp.get("expiration")
    expires_at = None
    if expiration_ms is not None:
        expires_at = datetime.fromtimestamp(int(expiration_ms) / 1000, tz=timezone.utc).isoformat()
    return {
        "history_id": str(resp.get("historyId")) if resp.get("historyId") else None,
        "expires_at": expires_at,
        "raw_expiration_ms": expiration_ms,
        "renewed_at": now.isoformat(),
        "topic": topic,
    }


def is_watch_expiring(watch_state: dict[str, Any] | None, *, now: datetime | None = None) -> bool:
    """True if there is no watch or it expires within the renew window."""
    now = now or datetime.now(timezone.utc)
    if not watch_state:
        return True
    expires_at = watch_state.get("expires_at")
    if not expires_at:
        return True
    try:
        exp = datetime.fromisoformat(expires_at)
    except ValueError:
        return True
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return now >= (exp - WATCH_RENEW_BEFORE)
