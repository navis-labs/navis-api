"""Microsoft (Entra) OAuth token lifecycle and scope handling.

The tests that matter here are the two places Microsoft is NOT Google:

  * refresh tokens ROTATE, so the manager must always surface one for the
    caller to persist — including when Entra echoes the same value back;
  * Entra echoes granted scopes back in an inconsistent spelling, so the
    grant check must normalise before comparing.

Both failures are silent and delayed in production, which is exactly the kind
that needs a test rather than a code review.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from navis.signals.microsoft.oauth import (
    MicrosoftReconsentRequired,
    TokenManager,
    calendar_write_granted,
    mail_send_granted,
    normalize_scopes,
)


def _integration(*, refresh_token: str | None = "r-old", expired: bool = True):
    """Tokens are held ENCRYPTED, as every stored Integration is."""
    from navis.signals.crypto import token_cipher

    exp = datetime.now(timezone.utc) + timedelta(hours=-1 if expired else 5)
    return SimpleNamespace(
        id="int-ms-1",
        access_token=token_cipher.encrypt("a-old"),
        refresh_token=token_cipher.encrypt(refresh_token),
        token_expires_at=exp,
    )


# ---- refresh-token rotation -------------------------------------------------


async def test_rotated_refresh_token_is_surfaced():
    async def refresher(_rt):
        return {"access_token": "a-new", "expires_in": 3600, "refresh_token": "r-NEW"}

    tm = TokenManager(_integration(), refresher=refresher)
    tok = await tm.get_token()
    assert tok.refreshed is True
    assert tok.access_token == "a-new"
    assert tok.refresh_token == "r-NEW"


async def test_missing_rotation_still_surfaces_a_live_token():
    """The Google-shaped bug this guards against.

    Google omits `refresh_token` on re-consent and the caller is expected to
    KEEP the stored one, so Google's manager returns None to mean "unchanged".
    If Microsoft's did the same, `sync_integration`'s
    `if last_token.refresh_token:` guard would skip the write — and because
    Entra invalidates the used token regardless, the stored value would already
    be dead. The integration would keep working until the access token expired
    and then fail permanently, ~90 days after anyone touched this code.

    So the manager always carries a usable value, even when Entra sent none.
    """
    async def refresher(_rt):
        return {"access_token": "a-new", "expires_in": 3600}

    tm = TokenManager(_integration(), refresher=refresher)
    tok = await tm.get_token()
    assert tok.refreshed is True
    assert tok.refresh_token == "r-old"  # never None
    assert tok.refresh_token is not None


async def test_valid_token_is_not_refreshed():
    async def refresher(_rt):  # pragma: no cover - must not be called
        raise AssertionError("should not refresh a live token")

    tm = TokenManager(_integration(expired=False), refresher=refresher)
    tok = await tm.get_token()
    assert tok.refreshed is False
    assert tok.access_token == "a-old"


async def test_invalidate_forces_a_refresh():
    """A 401 self-heal: the transport invalidates, the next call must refresh
    even though local expiry still says the token is live."""
    calls = []

    async def refresher(rt):
        calls.append(rt)
        return {"access_token": "a-new", "expires_in": 3600, "refresh_token": "r-NEW"}

    tm = TokenManager(_integration(expired=False), refresher=refresher)
    assert (await tm.get_token()).refreshed is False
    tm.invalidate()
    assert (await tm.get_token()).refreshed is True
    assert calls == ["r-old"]


async def test_missing_refresh_token_requires_reconsent():
    async def refresher(_rt):  # pragma: no cover - must not be called
        raise AssertionError("nothing to refresh with")

    tm = TokenManager(_integration(refresh_token=None), refresher=refresher)
    with pytest.raises(MicrosoftReconsentRequired):
        await tm.get_token()
    # Backward compatible with any caller catching the broad type, and with the
    # Google side's ReconsentRequired.
    assert issubclass(MicrosoftReconsentRequired, PermissionError)


# ---- scope normalisation ----------------------------------------------------


def test_normalize_accepts_both_spellings_entra_returns():
    """Entra echoes scopes back either fully qualified or bare, and does not
    guarantee casing. A literal comparison rejects a grant the user actually
    gave — which surfaces to them as "Outlook didn't grant every permission"
    on a consent screen where they ticked every box."""
    qualified = "https://graph.microsoft.com/Mail.Read https://graph.microsoft.com/Mail.Send"
    bare = "Mail.Read Mail.Send"
    mixed_case = "mail.READ MAIL.send"
    assert normalize_scopes(qualified) == {"mail.read", "mail.send"}
    assert normalize_scopes(bare) == normalize_scopes(qualified)
    assert normalize_scopes(mixed_case) == normalize_scopes(qualified)


def test_normalize_handles_absent_and_empty():
    assert normalize_scopes(None) == set()
    assert normalize_scopes("") == set()
    assert normalize_scopes("   ") == set()


def test_oidc_scopes_pass_through_unprefixed():
    assert "offline_access" in normalize_scopes("openid profile offline_access")


def test_send_and_write_grant_checks():
    granted = (
        "openid offline_access https://graph.microsoft.com/Mail.Read "
        "https://graph.microsoft.com/Mail.Send "
        "https://graph.microsoft.com/Calendars.ReadWrite"
    )
    assert mail_send_granted(granted) is True
    assert calendar_write_granted(granted) is True

    read_only = "https://graph.microsoft.com/Mail.Read"
    assert mail_send_granted(read_only) is False
    assert calendar_write_granted(read_only) is False
    assert mail_send_granted(None) is False


def test_signin_scopes_never_request_mailbox_access():
    """The trust ramp: signing in and handing over a mailbox are two separate
    decisions made at two different moments. A mailbox scope reaching the
    sign-in consent screen is a product regression, not a config one, so it is
    asserted here rather than left to review."""
    from navis.signals.microsoft.oauth import MICROSOFT_SIGNIN_SCOPES

    joined = " ".join(MICROSOFT_SIGNIN_SCOPES).lower()
    for forbidden in ("mail.", "calendars.", "files.", "chat.", "channelmessage"):
        assert forbidden not in joined
