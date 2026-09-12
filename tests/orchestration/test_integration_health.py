"""Integration auth health — recording, gating, and deterministic recovery.

WHAT THESE PROTECT. Before 0050 a revoked Google grant produced a
`ReconsentRequired`, which `SignalSyncService.sync` caught with a broad
`except`, dead-lettered, and flattened into `failed=True` — indistinguishable
from a timeout. The row kept `is_active=True`, the fan-out retried it on every
beat forever, and `/developer/context` told agents the source was connected
while it had ingested nothing for weeks.

Every test below fails if any link in that chain regresses:

    classify   an auth failure is distinguishable from a transport failure
    record     it reaches `integrations.auth_state`, and `is_active` does not move
    preserve   the refresh token and the sync cursor survive the failure
    gate       a known-dead grant is not retried
    recover    a new grant clears it, from either direction
    separate   Slack's two events stop being one event
"""
from __future__ import annotations

import contextlib
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from navis.identity.models import IntegrationAuthState
from navis.signals.google.oauth import ReconsentRequired
from navis.signals.slack.client import SlackReconsentRequired
from navis.signals.sync import SyncOutcome, _auth_failure_reason

ORG = uuid.UUID("33333333-3333-3333-3333-333333333333")
CURSOR = {"history_id": "231965"}


# ── 1. Classification: a dead credential is not a slow network ──────────────


def test_a_google_invalid_grant_is_recognised_as_an_auth_failure():
    """`invalid_grant` is terminal. Retrying it produces the same refusal until
    a human re-authorises, so it must be distinguishable from a blip."""
    reason = _auth_failure_reason(
        ReconsentRequired("google refresh rejected (invalid_grant)")
    )
    assert reason == "google refresh rejected (invalid_grant)"


def test_a_slack_token_revocation_is_recognised_as_an_auth_failure():
    assert _auth_failure_reason(SlackReconsentRequired("token_revoked", status=200))


def test_a_timeout_is_not_an_auth_failure():
    """THE FALSE POSITIVE THAT WOULD BE WORSE THAN THE BUG. Marking a grant
    `reconsent_required` on a network blip stops syncing a working account and
    puts a re-consent prompt in front of a user who has nothing to fix."""
    assert _auth_failure_reason(TimeoutError("connector exceeded 30s")) is None
    assert _auth_failure_reason(RuntimeError("provider returned 503")) is None


def test_a_failure_reason_is_bounded():
    """An exception message is a classic place for a credentialled URL to
    appear, and the column is VARCHAR(500)."""
    reason = _auth_failure_reason(ReconsentRequired("x" * 5000))
    assert reason is not None and len(reason) <= 300


# ── 2. Recording: the row learns what happened ──────────────────────────────


def _integration(**kw):
    """An `integrations` row as `sync_integration` re-reads it inside the lock."""
    fields = {
        "id": uuid.uuid4(),
        "organization_id": ORG,
        "user_id": uuid.uuid4(),
        "is_active": True,
        "auth_state": IntegrationAuthState.OK.value,
        "last_attempt_at": None,
        "last_success_at": None,
        "last_failure_reason": None,
        "access_token": "enc-access",
        "refresh_token": "enc-refresh",
        "token_expires_at": None,
        "sync_state": {"cursor": CURSOR},
    }
    fields.update(kw)
    return SimpleNamespace(**fields)


async def _run_sync(monkeypatch, integration, outcome):
    """Drive `sync_integration` to completion with a stubbed sync engine.

    Everything below the row is faked — the point is what the FUNCTION writes to
    the row, not what the connector does.
    """
    import navis.orchestration.sync as S

    integration.provider = S.IntegrationProvider.GMAIL

    class _DB:
        async def get(self, _model, _id):
            return integration

    @contextlib.asynccontextmanager
    async def _scope():
        yield _DB()

    @contextlib.asynccontextmanager
    async def _lock(_db, _id):
        yield True

    class _Transport:
        async def aclose(self):
            pass

    class _Engine:
        def __init__(self, _db):
            pass

        async def sync(self, _ctx, _connector, _client, cursor):
            # The engine is handed the row's cursor; assert it, because a
            # regression that dropped it would otherwise look like success.
            assert cursor == CURSOR
            return outcome

    monkeypatch.setattr(S, "session_scope", _scope)
    monkeypatch.setattr(S, "integration_advisory_lock", _lock)
    monkeypatch.setattr(S, "scheduled_work_allowed", lambda *a, **k: _true())
    monkeypatch.setattr(
        S, "_build_client",
        lambda _i: (object(), object(), (_Transport(), SimpleNamespace(last_token=None))),
    )
    monkeypatch.setattr(S, "SignalSyncService", _Engine)
    return await S.sync_integration(integration)


async def _true():
    return True


async def test_a_reconsent_failure_is_recorded_on_the_row(monkeypatch):
    """THE BUG THIS PHASE EXISTS TO FIX. The failure used to reach the DLQ and
    nothing else, so the row went on claiming to be a healthy connection."""
    integration = _integration()

    await _run_sync(monkeypatch, integration, SyncOutcome(
        failed=True, cursor=CURSOR,
        auth_failed=True,
        auth_failure_reason="google refresh rejected (invalid_grant)",
    ))

    assert integration.auth_state == IntegrationAuthState.RECONSENT_REQUIRED.value
    assert integration.last_failure_reason == "google refresh rejected (invalid_grant)"
    assert integration.last_attempt_at is not None
    assert integration.last_success_at is None, "a failure is not a success"


async def test_a_reconsent_failure_does_not_touch_the_user_s_intent(monkeypatch):
    """`is_active` means "the user wants this connected" and Google rejecting a
    token is not the user changing their mind. Flipping it here made an auth
    failure indistinguishable from a disconnect AND hid the row from the
    reconnect UI that exists to fix it."""
    integration = _integration()

    await _run_sync(monkeypatch, integration, SyncOutcome(
        failed=True, cursor=CURSOR, auth_failed=True, auth_failure_reason="invalid_grant",
    ))

    assert integration.is_active is True


async def test_a_reconsent_failure_preserves_the_credential_and_the_cursor(monkeypatch):
    """Neither may be discarded. The refresh token may be revived by re-consent,
    and the cursor is where ingestion must RESUME — dropping it would turn every
    re-consent into a full historical resync."""
    integration = _integration()

    await _run_sync(monkeypatch, integration, SyncOutcome(
        failed=True, cursor=CURSOR, auth_failed=True, auth_failure_reason="invalid_grant",
    ))

    assert integration.refresh_token == "enc-refresh"
    assert integration.access_token == "enc-access"
    assert integration.sync_state == {"cursor": CURSOR}


async def test_a_transient_failure_leaves_auth_state_alone(monkeypatch):
    """A timeout says nothing about the credential. Demoting a working grant on
    a network blip would stop syncing an account that is fine."""
    integration = _integration()

    await _run_sync(monkeypatch, integration, SyncOutcome(failed=True, cursor=CURSOR))

    assert integration.auth_state == IntegrationAuthState.OK.value
    assert integration.last_attempt_at is not None
    assert integration.last_failure_reason  # still reported, just not as auth


async def test_a_successful_pass_records_success_and_clears_a_stale_failure(monkeypatch):
    """A pass that reached the provider clears the failure record.

    The row here starts `ok` with a leftover `last_failure_reason` from an
    earlier TRANSIENT failure — which is the only state this branch can actually
    reach, because the auth gate refuses a `reconsent_required` row before the
    sync runs. See the next test, which pins that.
    """
    integration = _integration(last_failure_reason="sync failed; see dead-letter queue")
    new_cursor = {"history_id": "999999"}

    await _run_sync(monkeypatch, integration, SyncOutcome(
        ingested=3, cursor=new_cursor,
    ))

    assert integration.auth_state == IntegrationAuthState.OK.value
    assert integration.last_failure_reason is None
    assert integration.last_success_at is not None
    assert integration.last_attempt_at == integration.last_success_at, (
        "one wall-clock reading per pass — a success stamped before the attempt "
        "that produced it reads as a clock bug"
    )
    assert integration.sync_state["cursor"] == new_cursor


async def test_a_sync_cannot_rescue_a_grant_that_needs_reconsent(monkeypatch):
    """RECOVERY HAS EXACTLY ONE PATH, and this is the test that says so.

    `reconsent_required` is terminal until a new OAuth grant: the gate refuses
    the row, so no amount of retrying can flip it back to `ok`. That is not an
    oversight — a gate that let the row through "just in case" would be the
    retry storm again, and a token Google has rejected does not start working
    because it was asked a fourth time.

    The consequence is worth stating plainly: if the OAuth callback ever stops
    clearing `auth_state`, a re-consenting user would be stuck permanently. That
    is what `test_the_oauth_callback_clears_auth_state` protects.
    """
    integration = _integration(auth_state=IntegrationAuthState.RECONSENT_REQUIRED.value)

    out = await _run_sync(monkeypatch, integration, SyncOutcome(ingested=3))

    assert out == {"skipped": "auth_reconsent_required"}
    assert integration.auth_state == IntegrationAuthState.RECONSENT_REQUIRED.value
    assert integration.last_success_at is None


# ── 3. The gate: a known-dead grant is not retried ──────────────────────────


async def test_a_grant_needing_reconsent_is_not_attempted(monkeypatch):
    """THE RETRY STORM. Attempting it burns a worker, a provider rate-limit slot
    and a DLQ row to learn what the row already records."""
    import navis.orchestration.sync as S

    integration = _integration(auth_state=IntegrationAuthState.RECONSENT_REQUIRED.value)

    class _Boom:
        def __init__(self, _db):
            pass

        async def sync(self, *_a, **_k):
            raise AssertionError("retried a credential already known to be dead")

    monkeypatch.setattr(S, "SignalSyncService", _Boom)
    out = await _run_sync(monkeypatch, integration, SyncOutcome())

    assert out == {"skipped": "auth_reconsent_required"}


def test_the_scheduled_fan_out_excludes_unhealthy_rows():
    """`_dispatch_all` must filter in SQL, not enqueue and refuse. A dead grant
    that reaches the worker has already taken the queue slot the gate exists to
    save."""
    import inspect
    import textwrap

    import navis.orchestration.sync as S

    src = textwrap.dedent(inspect.getsource(S._dispatch_all))
    assert "auth_state" in src, "the fan-out does not filter on auth health"
    assert "is_active" in src, "the fan-out must still respect the user's intent"


# ── 4. Recovery: deterministic, and from the one event that proves it ───────


def test_the_oauth_callback_clears_auth_state():
    """A completed grant is the ONE event that proves a credential works, so it
    is the one event allowed to clear `reconsent_required` — and it must do so
    in the same statement that stores the new token. A follow-up write could
    leave a fresh token behind a stale broken flag, which keeps a working
    account out of the fan-out indefinitely with nothing to show why."""
    import inspect
    import textwrap

    import navis.signals.router as R

    src = textwrap.dedent(inspect.getsource(R))
    assert '"auth_state": IntegrationAuthState.OK.value' in src
    assert '"last_failure_reason": None' in src


def test_the_oauth_callback_does_not_reset_the_sync_cursor():
    """`sync_state` must NOT appear in the upsert's `set_`: re-consent resumes
    ingestion, it does not replay history."""
    import inspect
    import textwrap

    import navis.signals.router as R

    src = textwrap.dedent(inspect.getsource(R))
    upsert = src[src.index("on_conflict_do_update"):]
    upsert = upsert[: upsert.index(".returning")]
    assert "sync_state" not in upsert


# ── 5. Slack: two events that were one event ────────────────────────────────


async def _slack_event(monkeypatch, integration, inner_type):
    import navis.orchestration.sync as S

    class _DB:
        pass

    @contextlib.asynccontextmanager
    async def _scope():
        yield _DB()

    async def _resolve(_db, **_kw):
        return integration, None

    monkeypatch.setattr(S, "session_scope", _scope)
    monkeypatch.setattr(
        S, "_slack_install_narrower", lambda _e: (lambda rows: rows)
    )
    import navis.identity.org_service as O

    monkeypatch.setattr(O, "resolve_owning_integration", _resolve)
    return await S._process_slack_event(
        "T123", {"event": {"type": inner_type}}
    )


async def test_slack_token_revocation_moves_auth_not_intent(monkeypatch):
    """THE OVERLOAD THIS PHASE REMOVES. Nobody asked to disconnect — an admin
    rotated a token. Deactivating the row made that look like a user
    disconnect and stopped inbound events resolving at all, so the workspace
    went quiet with no explanation."""
    integration = _integration(provider=None)
    out = await _slack_event(monkeypatch, integration, "tokens_revoked")

    assert integration.auth_state == IntegrationAuthState.REVOKED.value
    assert integration.is_active is True, "intent must not move on an auth failure"
    assert out["auth_revoked"] is True
    assert out["deactivated"] is False


async def test_slack_app_uninstall_is_a_real_disconnect(monkeypatch):
    """Removing the app from the workspace IS a person choosing to disconnect,
    so this one legitimately moves intent — and the token is dead too."""
    integration = _integration(provider=None)
    out = await _slack_event(monkeypatch, integration, "app_uninstalled")

    assert integration.is_active is False
    assert integration.auth_state == IntegrationAuthState.REVOKED.value
    assert out["deactivated"] is True
