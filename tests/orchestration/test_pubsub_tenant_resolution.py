"""The webhook→tenant boundary, for BOTH authenticated push channels.

TASK 6.4.2 covered Google Pub/Sub; TASK 6.5.1 (T-1) brought the Slack Events path
to the same standard. They are tested together on purpose: authentication proves
WHO the caller is (Google, or Slack via HMAC) and says nothing about WHICH tenant
the notification concerns. That comes from resolving an external account —
a mailbox address, a workspace id — against persisted `Integration` rows, and
`uq_org_provider_account` is unique per ORGANIZATION, so the resolution is
one-to-many and can genuinely be ambiguous.

Every failure mode below must fail CLOSED. There is no ordering — newest, oldest,
lowest id, whatever `db.scalar()` happens to return — that makes an arbitrary
choice correct, because which tenant an event belongs to is a fact about the
world and not a preference.
"""
from __future__ import annotations

import contextlib
import uuid
from types import SimpleNamespace

import pytest

from navis.identity.models import IntegrationProvider
from navis.orchestration import sync as S


class _FakeScalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows
        self.queried = False

    async def scalars(self, _stmt):
        self.queried = True
        return _FakeScalars(self._rows)


@pytest.fixture()
def fake_db(monkeypatch):
    """Install a session_scope returning a caller-chosen row set."""

    def install(rows) -> _FakeDB:
        db = _FakeDB(rows)

        @contextlib.asynccontextmanager
        async def _scope():
            yield db

        monkeypatch.setattr(S, "session_scope", _scope)
        return db

    return install


def _integration(org_id=None, email="founder@acme.com"):
    return SimpleNamespace(
        id=uuid.uuid4(),
        organization_id=org_id or uuid.uuid4(),
        external_account=email,
        is_active=True,
    )


async def test_notification_without_a_mailbox_address_is_refused(fake_db, monkeypatch):
    """The `if email:` filter used to be optional, so a payload with no
    `emailAddress` dropped it and synced SOME arbitrary active integration."""
    install = fake_db
    db = install([_integration(), _integration()])
    called = []
    monkeypatch.setattr(S, "sync_integration", lambda i: called.append(i))

    out = await S._sync_account("gmail", None, "42")

    assert out == {"matched": False, "reason": "no_address"}
    assert called == [], "must not sync an arbitrarily-chosen tenant"
    assert db.queried is False, "must refuse before querying for a row to act on"


async def test_unknown_mailbox_is_refused(fake_db, monkeypatch):
    install = fake_db
    install([])
    called = []
    monkeypatch.setattr(S, "sync_integration", lambda i: called.append(i))

    out = await S._sync_account("gmail", "nobody@acme.com", "42")

    assert out == {"matched": False, "reason": "no_match"}
    assert called == []


async def test_mailbox_shared_by_two_organizations_is_refused(fake_db, monkeypatch):
    """`uq_org_provider_account` is unique per ORG, not globally, so two tenants
    can connect the same mailbox. The notification names only the address, so
    nothing identifies the tenant — picking one would silently attribute one
    organization's mail to the other."""
    install = fake_db
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    install([_integration(org_a), _integration(org_b)])
    called = []
    monkeypatch.setattr(S, "sync_integration", lambda i: called.append(i))

    out = await S._sync_account("gmail", "shared@acme.com", "42")

    assert out == {"matched": False, "reason": "ambiguous_tenant"}
    assert called == [], "must not guess a tenant"


async def test_single_match_syncs_that_integration(fake_db, monkeypatch):
    install = fake_db
    row = _integration()
    install([row])
    called = []

    async def _sync(i):
        called.append(i)
        return {"ok": True}

    monkeypatch.setattr(S, "sync_integration", _sync)

    out = await S._sync_account("gmail", "founder@acme.com", "42")

    assert out == {"ok": True}
    assert called == [row]


async def test_org_scope_comes_from_the_persisted_row_not_the_task_payload():
    """PHASE 8: the Celery signature has no organization parameter at all, so a
    webhook payload cannot introduce one. `sync_integration` re-reads the row and
    builds SignalContext from `fresh.organization_id`."""
    import inspect

    params = set(inspect.signature(S._sync_account).parameters)
    assert params == {"provider", "email", "history_id"}
    assert not params & {"org_id", "organization_id", "integration_id"}

    src = inspect.getsource(S.sync_integration)
    assert "org_id=fresh.organization_id" in src


# ── T-1: the Slack Events path ──────────────────────────────────────────────
#
# Reproduced before the fix: `_process_slack_event` resolved the install with
# `db.scalar(...)`, which returns the FIRST row of an unordered result set. With
# two organizations connected to one workspace, `SignalContext.org_id` came back
# as whichever row the planner produced first and the message was captured,
# mapped and ingested under it — one company's Slack conversations written into
# another company's memory, state and brief, irreversibly, because Memory is
# append-only.


def _slack_integration(org_id=None, team="T_ACME", *, bot_user_id=None, app_id=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        organization_id=org_id or uuid.uuid4(),
        user_id=uuid.uuid4(),
        provider=IntegrationProvider.SLACK,
        external_account=team,
        is_active=True,
        sync_state={"slack": {"bot_id": "B1", "bot_user_id": bot_user_id, "app_id": app_id}},
    )


def _envelope(team="T_ACME", *, authorizations=None, api_app_id=None, inner="message"):
    env = {
        "team_id": team,
        "event_id": "Ev1",
        "event": {"type": inner, "text": "hello", "channel": "C1", "ts": "1.0"},
    }
    if authorizations is not None:
        env["authorizations"] = authorizations
    if api_app_id is not None:
        env["api_app_id"] = api_app_id
    return env


@pytest.fixture()
def slack_capture(monkeypatch):
    """Record any SignalContext the ingest path is handed. Empty == refused."""
    seen: list = []

    class _FakeSync:
        def __init__(self, _db):
            pass

        async def ingest_events(self, ctx, _source, events):
            seen.append(ctx)
            return SimpleNamespace(ingested=len(events), duplicates=0, dead_lettered=0)

    monkeypatch.setattr(S, "SignalSyncService", _FakeSync)
    import navis.signals.slack.events as SE

    monkeypatch.setattr(SE, "events_from_envelope", lambda *a, **k: [{"e": 1}])
    return seen


async def test_slack_one_workspace_one_org_ingests(fake_db, slack_capture):
    org = uuid.uuid4()
    fake_db([_slack_integration(org)])

    out = await S._process_slack_event("T_ACME", _envelope())

    assert out["matched"] is True and out["captured"] == 1
    assert [c.org_id for c in slack_capture] == [org]


async def test_slack_workspace_shared_by_two_orgs_is_refused(fake_db, slack_capture):
    """THE T-1 REGRESSION. Nothing in the envelope says which tenant this is for."""
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    fake_db([_slack_integration(org_a), _slack_integration(org_b)])

    out = await S._process_slack_event("T_ACME", _envelope())

    assert out == {"matched": False, "reason": "ambiguous_tenant"}
    assert slack_capture == [], "must not create a SignalContext for a guessed tenant"


async def test_slack_shared_workspace_with_distinct_installs_resolves(
    fake_db, slack_capture
):
    """Ambiguity is refused, not assumed. When two organizations hold GENUINELY
    distinct installs in one workspace, Slack's own `authorizations[].user_id`
    names the install it delivered to, and that is matched against the bot user id
    recorded at install time — so a legitimately shared workspace keeps working."""
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    fake_db([
        _slack_integration(org_a, bot_user_id="U_A"),
        _slack_integration(org_b, bot_user_id="U_B"),
    ])

    out = await S._process_slack_event(
        "T_ACME", _envelope(authorizations=[{"user_id": "U_B", "is_bot": True}])
    )

    assert out["matched"] is True
    assert [c.org_id for c in slack_capture] == [org_b], "resolved to the WRONG install"


async def test_slack_narrower_that_matches_both_orgs_still_refuses(
    fake_db, slack_capture
):
    """A narrower may only SHRINK the candidate set; it can never make an
    ambiguous match unambiguous by agreeing with everyone."""
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    fake_db([
        _slack_integration(org_a, bot_user_id="U_SAME"),
        _slack_integration(org_b, bot_user_id="U_SAME"),
    ])

    out = await S._process_slack_event(
        "T_ACME", _envelope(authorizations=[{"user_id": "U_SAME", "is_bot": True}])
    )

    assert out == {"matched": False, "reason": "ambiguous_tenant"}
    assert slack_capture == []


async def test_slack_unknown_workspace_is_refused(fake_db, slack_capture):
    fake_db([])

    out = await S._process_slack_event("T_NOBODY", _envelope(team="T_NOBODY"))

    assert out == {"matched": False, "reason": "no_match"}
    assert slack_capture == []


async def test_slack_envelope_without_a_team_is_refused(fake_db, slack_capture):
    db = fake_db([_slack_integration()])

    out = await S._process_slack_event(None, {"event": {"type": "message"}})

    assert out == {"matched": False, "reason": "no_account"}
    assert slack_capture == []
    assert db.queried is False, "must refuse before querying for a row to act on"


async def test_slack_malformed_envelope_is_refused_not_crashed(fake_db, slack_capture):
    """A verified-but-garbage body must not raise (a 500 here is a retry storm)."""
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    fake_db([_slack_integration(org_a), _slack_integration(org_b)])

    out = await S._process_slack_event(
        "T_ACME", {"team_id": "T_ACME", "authorizations": "not-a-list"}
    )

    assert out == {"matched": False, "reason": "ambiguous_tenant"}
    assert slack_capture == []


async def test_slack_uninstall_cannot_deactivate_an_ambiguous_tenants_row(
    fake_db, slack_capture
):
    """`app_uninstalled` flips is_active=False. Under ambiguity that would have
    disabled an ARBITRARY organization's Slack integration."""
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    rows = [_slack_integration(org_a), _slack_integration(org_b)]
    fake_db(rows)

    out = await S._process_slack_event(
        "T_ACME", _envelope(inner="app_uninstalled")
    )

    assert out == {"matched": False, "reason": "ambiguous_tenant"}
    assert all(r.is_active for r in rows), "deactivated a guessed tenant's integration"


async def test_slack_resolution_never_uses_db_scalar():
    """`db.scalar()` on a multi-row predicate IS the defect. Pin its absence.

    Asserted on the PARSED body, not the source text: the function's own comment
    explains the removed `db.scalar(...)` call, so a substring check matches the
    explanation and passes for the wrong reason (it did, on first run).
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(S._process_slack_event)))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "scalar" not in called, "single-row fetch on a one-to-many tenant predicate"

    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "resolve_owning_integration" in names
