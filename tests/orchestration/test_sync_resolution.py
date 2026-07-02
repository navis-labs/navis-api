"""Regression: webhook -> Integration tenant resolution.

(organization_id, provider, external_account) is unique PER ORG, so the same Slack team_id
or Gmail address may hold ACTIVE integrations in more than one org. The resolution paths must
never pick one arbitrarily:

  * Slack event CONTENT is ingested under the resolved org — an arbitrary pick is cross-tenant
    injection. Ambiguity must fail CLOSED (drop, log) and workspace-level revocations must
    deactivate EVERY matching row.
  * A Gmail Pub/Sub push names a mailbox; every org that connected that mailbox must sync
    (each integration is self-scoped — it syncs its own org with its own token), and a push
    with no mailbox must be dropped, never routed by guesswork.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

from navis.orchestration import sync as sync_mod

ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")


def _integration(org, n: int, *, user: bool = True, provider="slack") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.UUID(f"00000000-0000-0000-0000-00000000000{n}"),
        organization_id=org,
        user_id=uuid.uuid5(uuid.NAMESPACE_DNS, f"user{n}") if user else None,
        provider=provider,
        is_active=True,
        sync_state={},
    )


class _FakeDB:
    """Just enough AsyncSession: scalars(...).all() returns the preset rows."""

    def __init__(self, rows):
        self.rows = rows

    async def scalars(self, stmt):
        return SimpleNamespace(all=lambda: list(self.rows))


def _patch_session(monkeypatch, rows):
    @asynccontextmanager
    async def fake_scope():
        yield _FakeDB(rows)

    monkeypatch.setattr(sync_mod, "session_scope", fake_scope)


class _SpySyncService:
    """Stands in for SignalSyncService; records which tenant ingestion ran under."""

    calls: list = []

    def __init__(self, db):
        pass

    async def ingest_events(self, ctx, source, events):
        _SpySyncService.calls.append(ctx)
        return SimpleNamespace(ingested=len(events), duplicates=0, dead_lettered=0)


# ---- Slack: _process_slack_event -------------------------------------------------------------

async def test_slack_single_match_ingests_under_that_org(monkeypatch):
    integ = _integration(ORG_A, 1)
    _patch_session(monkeypatch, [integ])
    _SpySyncService.calls = []
    monkeypatch.setattr(sync_mod, "SignalSyncService", _SpySyncService)
    monkeypatch.setattr(
        "navis.signals.slack.events.events_from_envelope",
        lambda ctx, envelope, our_bot_id=None, our_bot_user_id=None: [object()],
    )
    out = await sync_mod._process_slack_event("T1", {"event": {"type": "message"}})
    assert out["matched"] is True and out["captured"] == 1
    assert [c.org_id for c in _SpySyncService.calls] == [ORG_A]


async def test_slack_ambiguous_team_fails_closed(monkeypatch):
    # Two orgs hold active rows for one team_id: content must NOT be ingested under either.
    _patch_session(monkeypatch, [_integration(ORG_A, 1), _integration(ORG_B, 2)])
    _SpySyncService.calls = []
    monkeypatch.setattr(sync_mod, "SignalSyncService", _SpySyncService)
    out = await sync_mod._process_slack_event("T1", {"event": {"type": "message"}})
    assert out == {"matched": False, "ambiguous": True}
    assert _SpySyncService.calls == []  # nothing crossed a tenant boundary


async def test_slack_uninstall_deactivates_every_matching_row(monkeypatch):
    # Revocation is a workspace-level fact — every org's row for the team goes inactive.
    rows = [_integration(ORG_A, 1), _integration(ORG_B, 2)]
    _patch_session(monkeypatch, rows)
    out = await sync_mod._process_slack_event("T1", {"event": {"type": "app_uninstalled"}})
    assert out == {"matched": True, "deactivated": 2}
    assert all(r.is_active is False for r in rows)


async def test_slack_no_match_and_missing_user_id(monkeypatch):
    _patch_session(monkeypatch, [])
    assert (await sync_mod._process_slack_event("T1", {}))["matched"] is False

    _patch_session(monkeypatch, [_integration(ORG_A, 1, user=False)])
    out = await sync_mod._process_slack_event("T1", {"event": {"type": "message"}})
    assert out["matched"] is False


# ---- Gmail Pub/Sub: _sync_account -------------------------------------------------------------

async def test_pubsub_without_mailbox_is_dropped_not_guessed(monkeypatch):
    def _explode():  # the DB must not even be consulted
        raise AssertionError("session_scope must not be entered for an unroutable push")

    monkeypatch.setattr(sync_mod, "session_scope", _explode)
    out = await sync_mod._sync_account("gmail", None, "42")
    assert out == {"matched": False}


async def test_pubsub_syncs_every_org_that_connected_the_mailbox(monkeypatch):
    rows = [_integration(ORG_A, 1, provider="gmail"), _integration(ORG_B, 2, provider="gmail")]
    _patch_session(monkeypatch, rows)
    synced: list = []

    async def fake_sync(integration):
        synced.append(integration.organization_id)
        return {"provider": "gmail", "ingested": 0, "duplicates": 0, "dead_lettered": 0}

    monkeypatch.setattr(sync_mod, "sync_integration", fake_sync)
    out = await sync_mod._sync_account("gmail", "a@acme.com", "42")
    assert synced == [ORG_A, ORG_B]  # both tenants sync, deterministic id order
    assert out["matched"] is True and out["synced"] == 2


async def test_pubsub_single_match_keeps_original_return_shape(monkeypatch):
    _patch_session(monkeypatch, [_integration(ORG_A, 1, provider="gmail")])

    async def fake_sync(integration):
        return {"provider": "gmail", "ingested": 3, "duplicates": 0, "dead_lettered": 0}

    monkeypatch.setattr(sync_mod, "sync_integration", fake_sync)
    out = await sync_mod._sync_account("gmail", "a@acme.com", "42")
    assert out == {"provider": "gmail", "ingested": 3, "duplicates": 0, "dead_lettered": 0}


async def test_pubsub_no_match(monkeypatch):
    _patch_session(monkeypatch, [])
    assert (await sync_mod._sync_account("gmail", "a@acme.com", None))["matched"] is False
