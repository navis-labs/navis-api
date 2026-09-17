"""Google OAuth token lifecycle: refresh-token rotation, revocation (invalid_grant ->
ReconsentRequired), and the missing-token re-consent path."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from navis.signals.google import oauth
from navis.signals.google.oauth import ReconsentRequired, TokenManager


def _integration(*, refresh_token: str | None = "r-old", expired: bool = True):
    """Tokens are held ENCRYPTED, as every stored Integration is.

    `TokenManager` decrypts internally, so the assertions below still compare
    PLAINTEXT values — but a bare-plaintext fixture would only be readable
    through `TokenCipher`'s legacy passthrough, which raises once
    `SIGNALS_REJECT_LEGACY_PLAINTEXT` is on (TASK 6.4.1).
    """
    from navis.signals.crypto import token_cipher

    exp = datetime.now(timezone.utc) + timedelta(hours=-1 if expired else 5)
    return SimpleNamespace(
        id="int-1",
        access_token=token_cipher.encrypt("a-old"),
        refresh_token=token_cipher.encrypt(refresh_token),
        token_expires_at=exp,
    )


async def test_refresh_token_rotation_is_captured():
    async def refresher(_rt):
        return {"access_token": "a-new", "expires_in": 3600, "refresh_token": "r-NEW"}

    tm = TokenManager(_integration(), refresher=refresher)
    tok = await tm.get_token()
    assert tok.refreshed is True
    assert tok.access_token == "a-new"
    assert tok.refresh_token == "r-NEW"        # rotated token surfaced for the caller to persist


async def test_unrotated_refresh_keeps_old_token():
    async def refresher(_rt):
        return {"access_token": "a-new", "expires_in": 3600}  # no refresh_token in response

    tm = TokenManager(_integration(), refresher=refresher)
    tok = await tm.get_token()
    assert tok.refreshed is True and tok.refresh_token is None  # None => keep the stored one


async def test_missing_refresh_token_requires_reconsent():
    async def refresher(_rt):  # pragma: no cover - never reached
        return {}

    tm = TokenManager(_integration(refresh_token=None), refresher=refresher)
    with pytest.raises(ReconsentRequired):
        await tm.get_token()
    assert issubclass(ReconsentRequired, PermissionError)  # backward compatible


async def test_valid_token_is_not_refreshed():
    async def refresher(_rt):  # pragma: no cover - must not be called
        raise AssertionError("should not refresh a still-valid token")

    tm = TokenManager(_integration(expired=False), refresher=refresher)
    tok = await tm.get_token()
    assert tok.refreshed is False


async def test_invalidate_forces_refresh_of_valid_token():
    # A still-valid (by local expiry) token that the server rejected (401) must refresh on
    # the next get_token after invalidate() — the 401 self-heal path.
    async def refresher(_rt):
        return {"access_token": "a-fresh", "expires_in": 3600}

    tm = TokenManager(_integration(expired=False), refresher=refresher)
    assert (await tm.get_token()).refreshed is False  # valid -> reused
    tm.invalidate()
    tok = await tm.get_token()
    assert tok.refreshed is True and tok.access_token == "a-fresh"


# ---- invalid_grant -> ReconsentRequired at the HTTP boundary ----

class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("raise_for_status reached for a handled status")


class _Client:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        return self._resp


async def test_invalid_grant_becomes_reconsent_required(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client(_Resp(400, {"error": "invalid_grant"})))
    with pytest.raises(ReconsentRequired):
        await oauth.refresh_access_token("r-dead")


async def test_successful_refresh_returns_response(monkeypatch):
    import httpx

    payload = {"access_token": "a", "expires_in": 3600, "refresh_token": "r-NEW"}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client(_Resp(200, payload)))
    out = await oauth.refresh_access_token("r-old")
    assert out["access_token"] == "a" and out["refresh_token"] == "r-NEW"
