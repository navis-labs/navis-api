"""OAuth v2 scope migration: Calendar write scope is granted, read-only
grants surface as reconsent-required, and the sender fails LOUD rather
than silently 403ing at Google.

These are pure-function tests — no HTTP, no DB. The reconsent path lives
in `_senders.calendar_sender` which we test at unit level by pointing it
at a stub integration with only read-only scopes."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from navis.signals.google.oauth import (
    CALENDAR_WRITE_SCOPES,
    GOOGLE_SCOPES,
    GOOGLE_SCOPE_VERSION,
    calendar_write_granted,
)


def test_google_scopes_include_calendar_events():
    """Scheduling requires `calendar.events` (or the broader `calendar`).
    Regressions that revert to `calendar.readonly` must fail this test."""
    assert "https://www.googleapis.com/auth/calendar.events" in GOOGLE_SCOPES


def test_google_scopes_include_gmail_send_but_nothing_broader():
    """Gmail send IS now requested — and still nothing wider than it.

    This test previously asserted the opposite, on the stated grounds that
    "Gmail send is a later feature — asking for it now would gate every org's
    reconnect on a permission we can't yet use." That condition no longer holds:
    proactive replies use it, and without the scope a reviewed, edited, approved
    draft 403s at the last step and dead-letters.

    `gmail.modify` stays out. It would also permit sending, but it grants label
    mutation and deletion of existing mail that Navis has no use for — and a
    consent screen asking for more than the product uses is both a trust cost
    and a security one. `gmail.send` is the narrowest scope that does the job.
    """
    assert "https://www.googleapis.com/auth/gmail.send" in GOOGLE_SCOPES
    assert "https://www.googleapis.com/auth/gmail.modify" not in GOOGLE_SCOPES
    assert "https://mail.google.com/" not in GOOGLE_SCOPES


def test_google_scope_version_bumped():
    """The version tag is how an org connected under an OLDER scope set is
    detected downstream, so it must move whenever GOOGLE_SCOPES does. v4 is the
    grant that added gmail.send."""
    assert GOOGLE_SCOPE_VERSION >= "v4"


# ── calendar_write_granted ─────────────────────────────────────────────


def test_calendar_write_granted_on_events_scope():
    assert calendar_write_granted(
        "https://www.googleapis.com/auth/calendar.events "
        "https://www.googleapis.com/auth/gmail.readonly"
    )


def test_calendar_write_granted_on_full_calendar_scope():
    assert calendar_write_granted(
        "openid https://www.googleapis.com/auth/calendar"
    )


def test_calendar_write_not_granted_on_readonly():
    assert not calendar_write_granted(
        "https://www.googleapis.com/auth/calendar.readonly "
        "https://www.googleapis.com/auth/gmail.readonly"
    )


def test_calendar_write_not_granted_on_empty():
    assert not calendar_write_granted(None)
    assert not calendar_write_granted("")


def test_all_declared_write_scopes_grant_write():
    """Round-trip: every string in CALENDAR_WRITE_SCOPES must itself
    grant write (no drift between the constant and the helper)."""
    for s in CALENDAR_WRITE_SCOPES:
        assert calendar_write_granted(s), f"{s} should grant calendar write"


# ── Sender-side pre-flight ─────────────────────────────────────────────


async def test_calendar_sender_rejects_readonly_integration(monkeypatch):
    """Legacy `calendar.readonly` integrations must fail LOUD at
    calendar_sender BEFORE any Google API call."""
    from navis.execution.providers import _senders

    from navis.identity.models import IntegrationProvider

    class _StubIntegration:
        provider = IntegrationProvider.GOOGLE_CALENDAR
        access_token = "ct"
        refresh_token = "cr"
        token_expires_at = None
        scopes = "https://www.googleapis.com/auth/calendar.readonly"
        external_account = "founder@acme.com"

    class _StubScalars:
        """`.scalars(...)` returns a result object with `.all()`."""

        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class _StubSessionCM:
        async def __aenter__(self):
            return SimpleNamespace(
                scalar=self._scalar,
                scalars=self._scalars,
                commit=self._noop,
                rollback=self._noop,
            )

        async def __aexit__(self, *a):
            return False

        async def _scalar(self, _stmt):
            return _StubIntegration()

        async def _scalars(self, _stmt):
            return _StubScalars([_StubIntegration()])

        async def _noop(self):
            return None

    monkeypatch.setattr(
        _senders, "session_scope", lambda: _StubSessionCM()
    )
    # The sender resolves the integration through the shared
    # `resolve_calendar_integration` (which also self-heals a missing
    # google_calendar row), so the stub session must answer `scalars(...)`.
    #
    # There is no longer a decrypt step to stub out: the sender snapshots the
    # tokens AS STORED and hands the ciphertext to TokenManager, which is what
    # decrypts. Nothing touches these stub tokens before the scope check refuses.
    payload = {
        "__ctx__": {
            "org_id": "11111111-1111-1111-1111-111111111111",
            "execution_id": "22222222-2222-2222-2222-222222222222",
            "decision_id": "33333333-3333-3333-3333-333333333333",
        },
        "title": "Chat with Rahul",
        "start": "2026-08-11T15:00:00+00:00",
        "end": "2026-08-11T15:30:00+00:00",
        "attendees": ["rahul@partner.com"],
        "time_zone": "UTC",
    }
    # The sender resolves through the generic integration boundary, so the
    # Google-specific CalendarScopeInsufficient is translated into the
    # provider-neutral IntegrationUnavailable the Execution layer understands.
    # The message still names the missing capability so the audit row tells the
    # user what to do, and `reconsent_required` distinguishes "reconnect to grant
    # more" from "connect Calendar" — a stronger assertion than the old match.
    from navis.execution.integrations import IntegrationUnavailable

    with pytest.raises(IntegrationUnavailable, match="write scope") as exc:
        await _senders.calendar_sender(payload)
    assert exc.value.provider == "calendar"
    assert exc.value.reconsent_required is True
