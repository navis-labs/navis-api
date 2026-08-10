"""Capability-aware Gmail sender selection.

The bug: both the sender and the approve-gate took `ORDER BY created_at DESC
LIMIT 1` and then checked the scope on whatever came back. On the workspace
where this was found, a read-only Google account had been connected roughly two
hours AFTER a send-capable one — so every approved draft failed a scope
pre-flight while a perfectly good sender sat one row down.

The rule these tests pin: capability is filtered FIRST; recency only ever breaks
ties among grants that can actually send.

Pure DB-shape tests — a stub session returns rows, no Postgres, no HTTP.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from navis.identity.models import IntegrationProvider
from navis.signals.google.resolve import (
    GmailNotConnected,
    GmailSendScopeInsufficient,
    resolve_gmail_send_integration,
)

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)

READONLY = "https://www.googleapis.com/auth/gmail.readonly openid"
SEND = "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.send"
MODIFY = "https://www.googleapis.com/auth/gmail.modify"
FULL_MAIL = "https://mail.google.com/"


def _grant(
    account: str,
    scopes: str,
    *,
    age_hours: float = 0.0,
    provider: IntegrationProvider = IntegrationProvider.GMAIL,
    tokens: bool = True,
):
    return SimpleNamespace(
        id=uuid.uuid5(uuid.NAMESPACE_DNS, account),
        organization_id=ORG,
        provider=provider,
        external_account=account,
        scopes=scopes,
        is_active=True,
        created_at=NOW - timedelta(hours=age_hours),
        access_token="at" if tokens else None,
        refresh_token="rt" if tokens else None,
        token_expires_at=None,
    )


class _StubDB:
    """Answers `scalars(...)` with a fixed row set — the only call the resolver makes."""

    def __init__(self, rows):
        self._rows = rows

    async def scalars(self, _stmt):
        return SimpleNamespace(all=lambda: list(self._rows))


# ── 1. the actual production bug ───────────────────────────────────────


async def test_newer_grant_without_send_never_beats_older_grant_with_send():
    """THE regression. Newest-wins picked `readonly@` here and the send failed."""
    db = _StubDB([
        _grant("readonly@navis.ai", READONLY, age_hours=0),   # newest
        _grant("sender@navis.ai", SEND, age_hours=2),         # older, capable
    ])
    picked = await resolve_gmail_send_integration(db, ORG)
    assert picked.external_account == "sender@navis.ai"


# ── 2. recency still applies WITHIN the capable set ────────────────────


async def test_newest_grant_wins_when_it_is_send_capable():
    """Capability-first must not turn into oldest-wins. When the newest grant
    CAN send, it is the one the user most recently chose — it wins."""
    db = _StubDB([
        _grant("new-sender@navis.ai", SEND, age_hours=0),
        _grant("old-sender@navis.ai", SEND, age_hours=48),
    ])
    picked = await resolve_gmail_send_integration(db, ORG)
    assert picked.external_account == "new-sender@navis.ai"


# ── 3. determinism ─────────────────────────────────────────────────────


async def test_selection_is_deterministic_across_row_orderings():
    """Same grants, different row order out of the DB -> same sender, every
    time. Includes an exact `created_at` tie, which is what forces the id
    tiebreak to exist: without it the winner would depend on the planner."""
    tied_a = _grant("tie-a@navis.ai", SEND, age_hours=5)
    tied_b = _grant("tie-b@navis.ai", SEND, age_hours=5)
    other = _grant("older@navis.ai", MODIFY, age_hours=9)

    orderings = [
        [tied_a, tied_b, other],
        [other, tied_b, tied_a],
        [tied_b, other, tied_a],
    ]
    picked = {
        (await resolve_gmail_send_integration(_StubDB(rows), ORG)).external_account
        for rows in orderings
    }
    assert len(picked) == 1, f"non-deterministic selection: {picked}"


async def test_dedicated_gmail_row_outranks_a_sibling_google_grant():
    """One Google consent covers every scope, so a `drive` row's token can send —
    but the Gmail row is the sender identity of record, so it wins even when the
    sibling is newer."""
    db = _StubDB([
        _grant("via-drive@navis.ai", SEND, age_hours=0, provider=IntegrationProvider.DRIVE),
        _grant("mailbox@navis.ai", SEND, age_hours=6),
    ])
    picked = await resolve_gmail_send_integration(db, ORG)
    assert picked.external_account == "mailbox@navis.ai"


# ── 4. no capable grant -> clear, actionable error ─────────────────────


async def test_no_send_capable_grant_raises_clear_permission_error():
    db = _StubDB([
        _grant("a@navis.ai", READONLY, age_hours=0),
        _grant("b@navis.ai", READONLY, age_hours=3),
    ])
    with pytest.raises(GmailSendScopeInsufficient, match="Reconnect Google"):
        await resolve_gmail_send_integration(db, ORG)


async def test_no_google_grant_at_all_is_a_distinct_error():
    """`not connected` and `connected but cannot send` need different words —
    telling someone to RE-connect an account they never connected is a dead end."""
    with pytest.raises(GmailNotConnected):
        await resolve_gmail_send_integration(_StubDB([]), ORG)


async def test_send_capable_grant_without_tokens_is_not_selected():
    with pytest.raises(GmailSendScopeInsufficient, match="tokens"):
        await resolve_gmail_send_integration(
            _StubDB([_grant("stale@navis.ai", SEND, tokens=False)]), ORG
        )


# ── 5 & 6. broader scopes we do not REQUEST but must honour ────────────


async def test_gmail_modify_grant_is_treated_as_send_capable():
    """We never ask for `gmail.modify` (see test_oauth_scope_write), but an org
    that granted it — via an older grant or an admin-wide policy — can send.
    Refusing would be a self-inflicted failure."""
    db = _StubDB([
        _grant("readonly@navis.ai", READONLY, age_hours=0),
        _grant("modify@navis.ai", MODIFY, age_hours=4),
    ])
    picked = await resolve_gmail_send_integration(db, ORG)
    assert picked.external_account == "modify@navis.ai"


async def test_full_mail_google_com_grant_is_treated_as_send_capable():
    db = _StubDB([
        _grant("readonly@navis.ai", READONLY, age_hours=0),
        _grant("full@navis.ai", FULL_MAIL, age_hours=4),
    ])
    picked = await resolve_gmail_send_integration(db, ORG)
    assert picked.external_account == "full@navis.ai"


# ── 7. read-only keeps working; it is only barred from SENDING ─────────


async def test_readonly_grant_stays_usable_for_reading_but_never_sends():
    """A read-only grant is a first-class integration for ingestion. This fix
    must not turn "cannot send" into "not connected" — the org's mail keeps
    syncing; only the send path refuses.
    """
    readonly = _grant("reader@navis.ai", READONLY, age_hours=0)
    capable = _grant("sender@navis.ai", SEND, age_hours=8)

    # It is a real, active grant with usable tokens...
    assert readonly.is_active and readonly.access_token

    # ...and it is still never chosen to send when a capable grant exists...
    assert (
        await resolve_gmail_send_integration(_StubDB([readonly, capable]), ORG)
    ).external_account == "sender@navis.ai"

    # ...nor when it is the ONLY grant, where it errors instead of being used.
    with pytest.raises(GmailSendScopeInsufficient):
        await resolve_gmail_send_integration(_StubDB([readonly]), ORG)


# ── the approve-gate asks the same question as the sender ──────────────


async def test_router_send_gate_agrees_with_the_resolver():
    """`_gmail_send_ready` used to run its OWN newest-wins query, so the gate and
    the sender could disagree. Sharing the resolver makes them one question."""
    from navis.execution.router import _gmail_send_ready

    assert await _gmail_send_ready(
        _StubDB([
            _grant("readonly@navis.ai", READONLY, age_hours=0),
            _grant("sender@navis.ai", SEND, age_hours=2),
        ]),
        ORG,
    )
    assert not await _gmail_send_ready(
        _StubDB([_grant("readonly@navis.ai", READONLY)]), ORG
    )
    assert not await _gmail_send_ready(_StubDB([]), ORG)
