"""P0-2: Gmail watch state — fields, expiry math, malformed-response guard, restart recovery."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from navis.signals.base import SignalSource, SyncResult, WorkspaceConnector
from navis.signals.gmail.watch import WatchStateError, build_watch_state, is_watch_expiring
from navis.signals.reliability.watch_recovery import WatchRecoveryService
from tests.signals.fakes import make_ctx

NOW = datetime(2026, 6, 17, tzinfo=timezone.utc)


def test_build_watch_state_captures_all_fields():
    resp = {"historyId": "12345", "expiration": "1781000000000"}  # epoch ms
    state = build_watch_state(resp, topic="projects/x/topics/gmail", now=NOW)
    assert state["history_id"] == "12345"
    assert state["expires_at"] is not None  # parsed expiration
    assert state["renewed_at"] == NOW.isoformat()
    assert state["topic"] == "projects/x/topics/gmail"


def test_build_watch_state_rejects_missing_history_id():
    # A malformed watch response must fail LOUD, not persist a watch that can never sync.
    with pytest.raises(WatchStateError):
        build_watch_state({"expiration": "1781000000000"}, topic="t", now=NOW)
    with pytest.raises(WatchStateError):
        build_watch_state({"historyId": None, "expiration": "1781000000000"}, topic="t", now=NOW)


def test_is_watch_expiring():
    assert is_watch_expiring(None, now=NOW) is True  # no watch
    fresh = {"expires_at": (NOW + timedelta(days=5)).isoformat()}
    assert is_watch_expiring(fresh, now=NOW) is False
    soon = {"expires_at": (NOW + timedelta(hours=2)).isoformat()}
    assert is_watch_expiring(soon, now=NOW) is True  # inside 24h renew window


class _StubGmail(WorkspaceConnector):
    source = SignalSource.GMAIL

    def __init__(self):
        self.created = 0

    async def full_sync(self, ctx, client):  # pragma: no cover
        return SyncResult()

    async def incremental_sync(self, ctx, client, cursor):  # pragma: no cover
        return SyncResult()

    async def create_watch(self, ctx, client):
        self.created += 1
        return build_watch_state(
            {"historyId": "999", "expiration": "1781000000000"},
            topic="projects/x/topics/gmail",
            now=NOW,
        )


async def test_recovery_renews_when_expiring_and_persists_state():
    connector = _StubGmail()
    svc = WatchRecoveryService()
    # Expiring watch → renew (returns fresh state to persist).
    expiring = {"expires_at": (NOW + timedelta(hours=1)).isoformat()}
    new_state = await svc.ensure_watch(make_ctx(), connector, client=None, current_watch=expiring, now=NOW)
    assert connector.created == 1
    assert new_state and new_state["history_id"] == "999"


async def test_recovery_noop_when_watch_is_fresh():
    connector = _StubGmail()
    fresh = {"expires_at": (NOW + timedelta(days=5)).isoformat()}
    result = await WatchRecoveryService().ensure_watch(
        make_ctx(), connector, client=None, current_watch=fresh, now=NOW
    )
    assert result is None  # no renewal needed
    assert connector.created == 0
