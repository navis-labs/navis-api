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


class WatchStateError(Exception):
    """A users.watch() response was malformed (missing the historyId that the whole sync
    pipeline keys off). Raised so the failure surfaces NOW — and the renewal task retries —
    instead of persisting a watch that exists but can never advance, where the breakage is
    only discovered hours later when the mailbox has silently gone dark."""


def build_watch_state(
    resp: dict[str, Any], *, topic: str | None, now: datetime | None = None
) -> dict[str, Any]:
    """Normalize a users.watch() response into persistable watch state.

    Gmail returns `historyId` (the sync anchor) and `expiration` as epoch milliseconds.
    A response without a historyId is rejected loudly: persisting it would register a live
    watch whose incremental_sync can never start, turning a transient bad response into a
    silent, hours-delayed outage.
    """
    now = now or datetime.now(timezone.utc)
    history_id = resp.get("historyId")
    if not history_id:
        raise WatchStateError("users.watch() response missing historyId")
    expiration_ms = resp.get("expiration")
    expires_at = None
    if expiration_ms is not None:
        expires_at = datetime.fromtimestamp(int(expiration_ms) / 1000, tz=timezone.utc).isoformat()
    return {
        "history_id": str(history_id),
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
