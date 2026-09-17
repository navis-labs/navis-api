"""Capability-aware Gmail sender selection.

The bug: both the sender and the approve-gate took `ORDER BY created_at DESC
LIMIT 1` and then checked the scope on whatever came back. On the workspace
where this was found, a read-only Google account had been connected roughly two
hours AFTER a send-capable one — so every approved draft failed a scope
pre-flight while a perfectly good sender sat one row down.

The rule these tests pin: capability is filtered FIRST; recency only ever breaks
ties among grants that can actually send.

TASK 6.5.1 (T-11) narrowed that second clause. Recency may break a tie between
ROWS OF ONE ACCOUNT (the Google family writes a gmail/calendar/drive row per
grant, all carrying the same address) — but it may NOT choose between two
DIFFERENT mailboxes. Sending a reply from the wrong address is a real, visible
failure that no retry corrects, so an organization with two send-capable accounts
and an action that does not say which one to use is refused
(`GmailSendAccountAmbiguous`) rather than ranked. Actions created after T-11
record `execution_requests.integration_id` and never reach this resolver at all.

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
    GmailSendAccountAmbiguous,
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


async def test_two_send_capable_accounts_are_refused_not_ranked():
    """T-11. THIS TEST USED TO ASSERT THE OPPOSITE — "newest grant wins when it is
    send-capable" — and that was the defect: with Gmail A and Gmail B both
    connected, `created_at` decided which identity a reply was sent from. Right
    tenant, wrong mailbox, wrong thread, no error.

    Recency cannot answer "which of these two people is this reply from?", so the
    resolver no longer pretends it can.
    """
    db = _StubDB([
        _grant("new-sender@navis.ai", SEND, age_hours=0),
        _grant("old-sender@navis.ai", SEND, age_hours=48),
    ])
    with pytest.raises(GmailSendAccountAmbiguous, match="more than one"):
        await resolve_gmail_send_integration(db, ORG)


async def test_the_refusal_tells_the_user_how_to_resolve_it():
    """The org is connected AND entitled — the ACTION is under-specified — so the
    remedy is different from 'reconnect Google' and the message must say so."""
    db = _StubDB([
        _grant("a@navis.ai", SEND, age_hours=0),
        _grant("b@navis.ai", SEND, age_hours=1),
    ])
    with pytest.raises(GmailSendAccountAmbiguous, match="recreate the action"):
        await resolve_gmail_send_integration(db, ORG)


# ── 3. determinism ─────────────────────────────────────────────────────


async def test_selection_is_deterministic_across_row_orderings():
    """Same grants, different row order out of the DB -> same sender, every
    time. Includes an exact `created_at` tie, which is what forces the id
    tiebreak to exist: without it the winner would depend on the planner.

    All rows are ONE account (T-11): a single Google consent writes a
    gmail/calendar/drive row per grant, all carrying the same address, so a tie
    among them is a tie between ROWS and recency legitimately breaks it. A tie
    between two different mailboxes is refused instead — see the test above.
    """
    tied_a = _grant("mailbox@navis.ai", SEND, age_hours=5)
    tied_b = _grant("mailbox@navis.ai", SEND, age_hours=5,
                    provider=IntegrationProvider.DRIVE)
    other = _grant("mailbox@navis.ai", MODIFY, age_hours=9,
                   provider=IntegrationProvider.GOOGLE_CALENDAR)
    # Distinct ids, so an id-based tiebreak is actually exercised.
    tied_b.id = uuid.uuid5(uuid.NAMESPACE_DNS, "mailbox-drive")
    other.id = uuid.uuid5(uuid.NAMESPACE_DNS, "mailbox-calendar")

    orderings = [
        [tied_a, tied_b, other],
        [other, tied_b, tied_a],
        [tied_b, other, tied_a],
    ]
    picked = {
        (await resolve_gmail_send_integration(_StubDB(rows), ORG)).id
        for rows in orderings
    }
    assert len(picked) == 1, f"non-deterministic selection: {picked}"


async def test_dedicated_gmail_row_outranks_a_sibling_google_grant():
    """One Google consent covers every scope, so a `drive` row's token can send —
    but the Gmail row is the sender identity of record, so it wins even when the
    sibling is newer.

    Both rows carry the SAME address, which is what the Google callback actually
    writes (`signals/router.py` upserts one row per granted provider from a single
    grant, all with the same `external_account`). The original fixture used two
    different addresses, which is not a shape the connector can produce — and
    under T-11 that shape is now correctly refused as two accounts.
    """
    db = _StubDB([
        _grant("mailbox@navis.ai", SEND, age_hours=0, provider=IntegrationProvider.DRIVE),
        _grant("mailbox@navis.ai", SEND, age_hours=6),
    ])
    picked = await resolve_gmail_send_integration(db, ORG)
    assert picked.provider is IntegrationProvider.GMAIL


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


async def test_approve_preflight_agrees_with_the_resolver():
    """The approve gate used to run its OWN newest-wins query, so the gate and
    the sender could disagree. Sharing the resolver makes them one question.

    The check now reaches the approve handler through `ActionSpec.preflight`
    rather than being inlined in the router (which is what made a generic handler
    know about Google), and signals refusal by raising rather than returning
    False. Same resolver, same three cases.
    """
    from navis.execution.actions import ActionPreconditionFailed, require_spec

    preflight = require_spec("reply_to_email").preflight
    assert preflight is not None, "the reply action must carry a send preflight"

    # A send-capable grant wins even when a newer read-only grant exists.
    await preflight(
        _StubDB([
            _grant("readonly@navis.ai", READONLY, age_hours=0),
            _grant("sender@navis.ai", SEND, age_hours=2),
        ]),
        ORG,
    )
    with pytest.raises(ActionPreconditionFailed):
        await preflight(_StubDB([_grant("readonly@navis.ai", READONLY)]), ORG)
    with pytest.raises(ActionPreconditionFailed):
        await preflight(_StubDB([]), ORG)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 6.5.1 — the SAME rule, for Calendar
# ═════════════════════════════════════════════════════════════════════════════
#
# `resolve_calendar_integration` had the asymmetry this file's Gmail tests
# already closed: it ordered by `created_at DESC` and took the first
# `google_calendar` row, so an organization with two connected Google accounts
# silently used whichever was connected most recently. For a read that is a wrong
# answer; on the meeting-scheduling path (`require_write=True`) it books the
# event on the wrong person's calendar and invites from the wrong identity.
#
# Kept in this file deliberately: the Gmail and Calendar resolvers now share one
# rule, and a future change to either should have to walk past the other.

CAL_READ = "https://www.googleapis.com/auth/calendar.readonly"
CAL_WRITE = (
    "https://www.googleapis.com/auth/calendar "
    "https://www.googleapis.com/auth/calendar.events"
)


async def test_two_google_accounts_with_calendars_are_refused_not_ranked():
    """The Calendar twin of `test_two_send_capable_accounts_are_refused_not_ranked`."""
    from navis.signals.google.resolve import (
        CalendarAccountAmbiguous,
        resolve_calendar_integration,
    )

    db = _StubDB([
        _grant("ops@navis.ai", CAL_WRITE, age_hours=0,
               provider=IntegrationProvider.GOOGLE_CALENDAR),
        _grant("founder@navis.ai", CAL_WRITE, age_hours=48,
               provider=IntegrationProvider.GOOGLE_CALENDAR),
    ])
    with pytest.raises(CalendarAccountAmbiguous, match="more than one"):
        await resolve_calendar_integration(db, ORG, require_write=True)


async def test_calendar_ambiguity_is_refused_on_the_read_path_too():
    """Not only on writes. A read that returns the wrong founder's availability
    is how the wrong meeting gets proposed in the first place."""
    from navis.signals.google.resolve import (
        CalendarAccountAmbiguous,
        resolve_calendar_integration,
    )

    db = _StubDB([
        _grant("a@navis.ai", CAL_READ, age_hours=0,
               provider=IntegrationProvider.GOOGLE_CALENDAR),
        _grant("b@navis.ai", CAL_READ, age_hours=1,
               provider=IntegrationProvider.GOOGLE_CALENDAR),
    ])
    with pytest.raises(CalendarAccountAmbiguous):
        await resolve_calendar_integration(db, ORG)


async def test_one_google_account_with_several_family_rows_is_not_ambiguous():
    """THE CASE THAT MUST KEEP WORKING, and the reason the check is keyed on the
    ACCOUNT rather than the row count.

    One Google consent writes a gmail row AND a calendar row AND a drive row for
    the same address, all sharing a token. Three rows, one identity — refusing
    that would break every normally-connected organization.
    """
    from navis.signals.google.resolve import resolve_calendar_integration

    db = _StubDB([
        _grant("solo@navis.ai", CAL_WRITE, age_hours=0,
               provider=IntegrationProvider.GOOGLE_CALENDAR),
        _grant("solo@navis.ai", CAL_WRITE, age_hours=0,
               provider=IntegrationProvider.GMAIL),
        _grant("solo@navis.ai", CAL_WRITE, age_hours=0,
               provider=IntegrationProvider.DRIVE),
    ])
    picked = await resolve_calendar_integration(db, ORG, require_write=True)
    assert picked.external_account == "solo@navis.ai"
    assert picked.provider == IntegrationProvider.GOOGLE_CALENDAR


async def test_a_single_calendar_account_still_resolves():
    """The control — the fix must refuse ambiguity, not refuse everything."""
    from navis.signals.google.resolve import resolve_calendar_integration

    db = _StubDB([
        _grant("only@navis.ai", CAL_WRITE, age_hours=0,
               provider=IntegrationProvider.GOOGLE_CALENDAR),
    ])
    picked = await resolve_calendar_integration(db, ORG, require_write=True)
    assert picked.external_account == "only@navis.ai"


async def test_calendar_ambiguity_reaches_execution_as_a_PERMANENT_failure():
    """The worker classifies failures by TYPE. A retryable classification here
    would burn the retry budget rediscovering that the action still does not say
    which calendar it means — so the credential resolver must translate this into
    `IntegrationUnavailable`, which the worker treats as permanent."""
    from navis.execution.integrations import IntegrationUnavailable
    from navis.execution.providers._senders import resolve_calendar_credential

    db = _StubDB([
        _grant("a@navis.ai", CAL_WRITE, age_hours=0,
               provider=IntegrationProvider.GOOGLE_CALENDAR),
        _grant("b@navis.ai", CAL_WRITE, age_hours=1,
               provider=IntegrationProvider.GOOGLE_CALENDAR),
    ])
    with pytest.raises(IntegrationUnavailable) as exc:
        await resolve_calendar_credential(db, ORG)
    assert exc.value.provider == "calendar"
    # NOT a reconsent: both accounts are connected fine, the action is the
    # thing that is under-specified. The UI must not say "reconnect Google".
    assert exc.value.reconsent_required is False
