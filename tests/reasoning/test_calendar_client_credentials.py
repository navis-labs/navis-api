"""Regression suite for the REAL calendar credential path.

Every other calendar test injects a fake `calendar_client_factory`, so
`_ManagedCalendarClient` — the thing that actually turns a stored Integration
row into an authenticated Google client — had no coverage at all. That is how
the double-decrypt below reached production.

THE BUG THIS PINS
-----------------
`__aenter__` used to decrypt the stored tokens in place onto the ORM row before
handing it to `TokenManager`. But `TokenManager.get_token` decrypts what it is
given, so the tokens were decrypted TWICE. While
`signals_reject_legacy_plaintext` was off, the second decrypt saw an untagged
value and returned it as-is, so nothing looked wrong. With the sunset switch on
it raises `TokenDecryptError` instead — and since the scheduling agent wraps
freebusy in a bare `except Exception`, every availability check collapsed into
the generic "I couldn't check your calendar availability just now. Try
reconnecting Google Calendar" clarification, on orgs whose Google grant was
healthy and correctly scoped.

Assigning plaintext onto the row was also an encryption-at-rest leak: it marks
the row dirty, so any commit on that session writes real tokens back in the
clear.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from navis.platform.config.settings import settings
from navis.reasoning.scheduling import availability as availability_mod
from navis.signals.crypto import TokenDecryptError, token_cipher

ACCESS_PLAINTEXT = "ya29.access-token-plaintext"
REFRESH_PLAINTEXT = "1//refresh-token-plaintext"


class _FakeIntegration:
    """The fields `TokenManager` duck-types, holding tokens AS STORED."""

    def __init__(self) -> None:
        self.id = uuid.uuid4()
        self.access_token = token_cipher.encrypt(ACCESS_PLAINTEXT)
        self.refresh_token = token_cipher.encrypt(REFRESH_PLAINTEXT)
        # Far future, so `get_token` takes its un-expired branch — the one
        # that raised on the second decrypt. (A None expiry counts as
        # expired and would send the test to Google for a real refresh.)
        self.token_expires_at = datetime.now(timezone.utc) + timedelta(days=30)


class _FakeSessionCM:
    """Stands in for `session_scope()` without touching a database."""

    def __init__(self) -> None:
        self.committed = 0
        self.rolled_back = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def commit(self):
        self.committed += 1

    async def rollback(self):
        self.rolled_back += 1


@pytest.fixture
def sunset_on(monkeypatch):
    """Run with the plaintext sunset switch ON — production's setting."""
    monkeypatch.setattr(settings, "signals_reject_legacy_plaintext", True)


@pytest.fixture
def managed(monkeypatch):
    """A `_ManagedCalendarClient` wired to a fake session + fake integration."""
    integration = _FakeIntegration()
    session = _FakeSessionCM()

    async def _resolve(db, org_id, **kwargs):
        return integration

    monkeypatch.setattr(availability_mod, "session_scope", lambda: session)
    monkeypatch.setattr(
        availability_mod, "resolve_calendar_integration", _resolve
    )
    cm = availability_mod._ManagedCalendarClient()._set_org_id(uuid.uuid4())
    return cm, integration, session


async def test_token_provider_yields_plaintext_under_sunset(
    sunset_on, managed, monkeypatch
):
    """The end-to-end guard: a healthy encrypted grant must produce a usable
    access token even with legacy plaintext rejected.

    Before the fix this raised `TokenDecryptError` — the exact exception the
    scheduling agent reported as "try reconnecting Google Calendar".
    """
    cm, _integration, _session = managed
    await cm.__aenter__()
    try:
        token = await cm._token_manager.token_provider()()
    finally:
        await cm.__aexit__(None, None, None)

    assert token == ACCESS_PLAINTEXT


async def test_stored_tokens_are_never_mutated_to_plaintext(sunset_on, managed):
    """Encryption at rest: the ORM row must still hold ciphertext afterwards.

    The old decrypt-in-place left plaintext on a dirty row, so any commit on
    that session persisted unencrypted credentials.
    """
    cm, integration, _session = managed
    await cm.__aenter__()
    try:
        assert integration.access_token != ACCESS_PLAINTEXT
        assert integration.refresh_token != REFRESH_PLAINTEXT
        assert token_cipher.decrypt(integration.access_token) == ACCESS_PLAINTEXT
    finally:
        await cm.__aexit__(None, None, None)

    assert integration.access_token != ACCESS_PLAINTEXT
    assert integration.refresh_token != REFRESH_PLAINTEXT


async def test_double_decrypt_would_still_be_caught(sunset_on):
    """Pins the primitive the whole bug rested on, so a future refactor that
    re-introduces a pre-decrypt fails loudly here rather than in the product."""
    once = token_cipher.decrypt(token_cipher.encrypt(ACCESS_PLAINTEXT))
    assert once == ACCESS_PLAINTEXT
    with pytest.raises(TokenDecryptError):
        token_cipher.decrypt(once)


async def test_session_is_not_left_open_when_calendar_unresolvable(monkeypatch):
    """The unwind path still closes the session it opened."""
    session = _FakeSessionCM()

    async def _boom(db, org_id, **kwargs):
        raise RuntimeError("no calendar scope")

    monkeypatch.setattr(availability_mod, "session_scope", lambda: session)
    monkeypatch.setattr(availability_mod, "resolve_calendar_integration", _boom)

    cm = availability_mod._ManagedCalendarClient()._set_org_id(uuid.uuid4())
    with pytest.raises(RuntimeError):
        await cm.__aenter__()
    assert session.rolled_back == 1
