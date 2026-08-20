# pyright: reportArgumentType=false, reportOptionalMemberAccess=false
"""The Slack execution connector, end to end through the generic engine.

Slack is the first connector added AFTER the universal contract was finished, so
these tests double as the proof that adding one requires no change to generic
Execution: a spec, a credential resolver, a sender. Nothing else.

The Slack HTTP API is substituted at the transport boundary (httpx) — everything
above it is the real code: the real worker, the real authorization re-check, the
real integration boundary, the real `resolve_slack_credential`, the real
`SlackClient`.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from navis.execution.actions import require_spec
from navis.execution.integrations import (
    IntegrationUnavailable,
    has_integration_resolver,
    resolve_integration,
)
from navis.execution.providers import production_providers
from navis.execution.providers.base import ProviderValidationError
from navis.execution.providers.slack import SlackProvider
from navis.execution.retry import ExecutionRetryPolicy
from navis.execution.service import ExecutionService
from navis.execution.status import ExecutionStatus
from navis.execution.worker import execute_pass
from navis.organization.billing import BillingService
from navis.organization.contract import Capability, Plan
from navis.organization.entitlements import EntitlementService
from navis.platform.events import EventType
from navis.signals.slack.oauth import SLACK_SCOPES, slack_write_granted
from tests.execution.fakes import ORG_A, ORG_B, USER, FakeExecutionRepo
from tests.organization.fakes import FakeOrgRepo

FAST = ExecutionRetryPolicy(max_attempts=1, initial_backoff=0.0, max_backoff=0.0, jitter=0.0)

READ_ONLY = "channels:history,channels:read,users:read"
WRITE = "channels:history,channels:read,users:read,chat:write"


# ── ActionSpec + scope vocabulary ──────────────────────────────────────


def test_the_slack_spec_uses_the_existing_capability_vocabulary():
    spec = require_spec("send_slack")
    assert spec.provider == "slack"
    assert spec.capability is Capability.SLACK
    assert has_integration_resolver("slack")


def test_chat_write_is_requested_but_not_required_to_install():
    """Capture is the product floor; posting is a capability on top of it.

    A workspace admin who will not grant write must still be able to install
    Navis and keep every read feature — so `chat:write` is REQUESTED at install
    but is not in the required set. `slack_write_granted` is what refuses the
    send, with a reconnect instruction.
    """
    from navis.signals.slack.oauth import _REQUIRED_SCOPES

    assert "chat:write" in SLACK_SCOPES
    assert "chat:write" not in _REQUIRED_SCOPES


def test_write_scope_detection_uses_slacks_comma_delimiter():
    """Slack stores scopes comma-separated; Google uses spaces. Splitting on the
    wrong one reads the whole grant as a single unknown scope and refuses every
    send."""
    assert slack_write_granted(WRITE) is True
    assert slack_write_granted(READ_ONLY) is False
    assert slack_write_granted(None) is False
    assert slack_write_granted("") is False


# ── validate() ─────────────────────────────────────────────────────────


async def test_a_blank_slack_message_is_refused_before_the_api():
    """Same defect class as the blank-email bug: validate() checked only
    `channel`, so an approved action could post an empty message."""
    with pytest.raises(ProviderValidationError, match="channel"):
        await SlackProvider().validate({"text": "hi"})
    with pytest.raises(ProviderValidationError, match="blank message"):
        await SlackProvider().validate({"channel": "#general", "text": "   "})
    await SlackProvider().validate({"channel": "#general", "text": "real"})


# ── credential resolution ──────────────────────────────────────────────


class _StubDB:
    """Answers the one `select(Integration)` the Slack resolver issues."""

    def __init__(self, rows):
        self._rows = rows

    async def scalars(self, _stmt):
        rows = self._rows

        class _R:
            def all(self):
                return rows

        return _R()


def _install(
    *, scopes=WRITE, token="sealed-bot-token", active=True, created=1.0,
    workspace="T123",
):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from navis.identity.models import IntegrationProvider

    return SimpleNamespace(
        id=uuid.uuid4(),
        organization_id=ORG_A,
        provider=IntegrationProvider.SLACK,
        scopes=scopes,
        access_token=token,
        # A Slack bot token is long-lived: no refresh, no expiry. Present and
        # None (rather than absent) because the sender snapshots the same fields
        # off every provider's row, and a stub missing them would pass a shape
        # production never produces.
        refresh_token=None,
        token_expires_at=None,
        is_active=active,
        # The Slack TEAM id. Two installs sharing one team id are the same
        # workspace re-installed; two different ids are two different
        # workspaces, which is what `resolve_slack_credential` now refuses to
        # choose between.
        external_account=workspace,
        created_at=datetime.fromtimestamp(created, tz=timezone.utc),
    )


async def test_a_read_only_slack_install_cannot_post_and_says_why():
    from navis.execution.providers._senders import resolve_slack_credential

    with pytest.raises(IntegrationUnavailable) as exc:
        await resolve_slack_credential(_StubDB([_install(scopes=READ_ONLY)]), ORG_A)
    assert exc.value.provider == "slack"
    assert exc.value.reconsent_required is True
    assert "chat:write" in str(exc.value)


async def test_no_slack_install_is_a_distinct_error_from_a_read_only_one():
    from navis.execution.providers._senders import resolve_slack_credential

    with pytest.raises(IntegrationUnavailable) as exc:
        await resolve_slack_credential(_StubDB([]), ORG_A)
    # "connect Slack", not the wrong "reconnect Slack".
    assert exc.value.reconsent_required is False


async def test_a_write_capable_install_beats_a_newer_read_only_one():
    """The Gmail ordering bug, avoided by construction: capability first,
    recency only as a tiebreak among grants that can actually post.

    Both rows are the SAME workspace (one `external_account`) — a re-install, not
    a second Slack — so there is no ambiguity about WHERE the message goes, only
    about which row carries the usable token.
    """
    from navis.execution.providers._senders import resolve_slack_credential

    newer_readonly = _install(scopes=READ_ONLY, created=99.0)
    older_writable = _install(scopes=WRITE, token="the-right-token", created=1.0)
    row = await resolve_slack_credential(_StubDB([newer_readonly, older_writable]), ORG_A)
    assert row.access_token == "the-right-token"


# ── TASK 6.5.1 — two workspaces are refused, not ranked ────────────────


async def test_two_connected_workspaces_are_refused_not_ranked():
    """THE FIX. This resolver used to document "newest install wins, ties broken
    by id" as a virtue — deterministic, which is not the same as correct.

    `uq_org_provider_account` is unique per (org, provider, external_account), so
    one organization may legitimately connect an internal workspace AND a shared
    customer workspace. Ranking by recency posted every message to whichever was
    installed last: right tenant, wrong workspace, possibly in front of a
    customer, and no retry corrects it. Gmail and Calendar already refuse this;
    Slack was the odd one out.
    """
    from navis.execution.providers._senders import resolve_slack_credential

    internal = _install(workspace="T-INTERNAL", token="internal-token", created=1.0)
    customer = _install(workspace="T-CUSTOMER", token="customer-token", created=99.0)
    with pytest.raises(IntegrationUnavailable) as exc:
        await resolve_slack_credential(_StubDB([internal, customer]), ORG_A)
    assert exc.value.provider == "slack"
    assert "more than one connected Slack workspace" in str(exc.value)
    # The accounts are connected fine — the ACTION is under-specified — so the
    # UI must not tell the user to reconnect Slack.
    assert exc.value.reconsent_required is False


async def test_the_slack_refusal_names_the_remedy():
    from navis.execution.providers._senders import resolve_slack_credential

    with pytest.raises(IntegrationUnavailable, match="recreate the action"):
        await resolve_slack_credential(
            _StubDB([_install(workspace="T-A"), _install(workspace="T-B")]), ORG_A
        )


async def test_a_second_workspace_that_cannot_post_creates_no_ambiguity():
    """Capability is filtered BEFORE the ambiguity check, so a read-only second
    workspace does not block a send from the only one that can actually post.
    Refusing there would be a regression dressed up as safety."""
    from navis.execution.providers._senders import resolve_slack_credential

    poster = _install(workspace="T-POSTER", token="the-right-token")
    lurker = _install(workspace="T-LURKER", scopes=READ_ONLY, created=99.0)
    row = await resolve_slack_credential(_StubDB([poster, lurker]), ORG_A)
    assert row.access_token == "the-right-token"


async def test_a_single_workspace_still_resolves():
    """The control — refuse ambiguity, not everything."""
    from navis.execution.providers._senders import resolve_slack_credential

    row = await resolve_slack_credential(
        _StubDB([_install(workspace="T-ONLY", token="only-token")]), ORG_A
    )
    assert row.access_token == "only-token"


# ── the full path, with only the Slack HTTP boundary substituted ───────


async def _entitled(*orgs, plan=Plan.ENTERPRISE):
    org_repo = FakeOrgRepo()
    ent = EntitlementService(org_repo)
    for org in orgs:
        await BillingService(org_repo, ent).set_plan(org, plan, billing_authorized=True)
    return org_repo, ent


async def _approved_slack(repo, org_id=ORG_A, payload=None):
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        org_id, uuid.uuid4(), "send_slack",
        payload or {"channel": "C123", "text": "Standup moved to 10am"},
        reasoning_signal_ids=["s1"], reasoning_version=2, state_watermark="w-7",
    )
    await svc.approve(req.id, USER)
    return req


@pytest.fixture()
def slack_transport(monkeypatch):
    """Substitute ONLY the network. Resolver, token decrypt, SlackClient, retry
    classification and the worker are all the real implementations."""
    from navis.execution.providers import _senders

    calls: list[dict] = []
    state = {"body": {"ok": True, "ts": "1723987200.001900", "channel": "C123"}, "status": 200}

    class _Resp:
        status_code = 200

        def __init__(self, body, status):
            self.status_code = status
            self._body = body
            self.headers = {}

        def json(self):
            return self._body

    class _HTTP:
        def __init__(self, *a, **kw):
            pass

        async def post(self, url, data=None, headers=None):
            calls.append({"url": url, "data": data, "headers": headers})
            return _Resp(state["body"], state["status"])

        async def aclose(self):
            pass

    monkeypatch.setattr("navis.signals.slack.client.httpx.AsyncClient", _HTTP)
    monkeypatch.setattr(
        _senders, "resolve_integration",
        lambda db, org_id, provider, **_: _resolved(_install(scopes=WRITE)),
    )
    monkeypatch.setattr(_senders.token_cipher, "decrypt", lambda t: "xoxb-real-token")

    class _Session:
        """Mirrors the session surface `_load_grant` actually uses."""

        async def __aenter__(self):
            return SimpleNamespace(commit=self._noop, rollback=self._noop)

        async def __aexit__(self, *a):
            return False

        async def _noop(self):
            return None

    monkeypatch.setattr(_senders, "session_scope", lambda: _Session())
    return calls, state


async def _resolved(value):
    return value


async def test_slack_executes_end_to_end_through_the_production_provider(slack_transport):
    """Decision → ActionSpec → approval → runtime authorization → credential
    boundary → SlackProvider → chat.postMessage → ProviderReference → outbox."""
    calls, _state = slack_transport
    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled(ORG_A)
    repo.entitlements = ent
    req = await _approved_slack(repo)

    result = await execute_pass(
        repo, providers=production_providers(), policy=FAST, max_attempts=3
    )

    assert result["executed"] == 1
    assert req.status == ExecutionStatus.EXECUTED.value
    # A REAL Slack ts, not a dry-run id.
    assert req.provider_reference == {
        "provider": "slack", "external_id": "1723987200.001900",
    }
    assert "dry_run" not in req.provider_reference

    # The call actually went to chat.postMessage with the org's own token.
    assert len(calls) == 1
    assert calls[0]["url"].endswith("/chat.postMessage")
    assert calls[0]["data"]["channel"] == "C123"
    assert calls[0]["headers"]["Authorization"] == "Bearer xoxb-real-token"

    # Observation gets the connector-neutral envelope, same as Gmail/Calendar.
    executed = [e for e in repo.events if e.event_type == EventType.ACTION_EXECUTED.value]
    assert len(executed) == 1
    outcome = executed[0].payload["outcome"]
    assert outcome["org_id"] == str(ORG_A)
    assert outcome["action_type"] == "send_slack"
    assert outcome["reasoning_signal_ids"] == ["s1"]
    assert outcome["provider_reference"]["provider"] == "slack"


async def test_no_slack_token_reaches_the_row_outbox_or_dead_letter(slack_transport):
    _calls, _state = slack_transport
    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled(ORG_A)
    repo.entitlements = ent
    req = await _approved_slack(repo)

    await execute_pass(repo, providers=production_providers(), policy=FAST, max_attempts=3)

    surfaces = [
        str(req.payload), str(req.provider_reference), str(req.last_error),
        str([e.payload for e in repo.events]),
        str([(d.error, d.payload) for d in repo.dead_letters]),
    ]
    for surface in surfaces:
        assert "xoxb" not in surface
        assert "sealed-bot-token" not in surface
        assert "Bearer" not in surface


async def test_a_revoked_slack_capability_blocks_the_post_and_the_credential(
    slack_transport,
):
    """Runtime authorization runs BEFORE credential resolution: a revoked SLACK
    capability means zero credential lookups and zero Slack API calls."""
    calls, _state = slack_transport
    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled(ORG_A)
    repo.entitlements = ent
    req = await _approved_slack(repo)

    await ent.set(ORG_A, Capability.SLACK, False)

    await execute_pass(repo, providers=production_providers(), policy=FAST, max_attempts=3)

    assert calls == [], "no Slack API call may happen for an unauthorized org"
    assert req.status == ExecutionStatus.FAILED.value
    assert "slack" in (req.last_error or "")
    assert not req.provider_reference


async def test_a_dead_slack_token_dead_letters_immediately_instead_of_retrying(
    slack_transport,
):
    """`token_revoked` is terminal — the install needs re-consent. It maps to
    IntegrationUnavailable, which the worker treats as PERMANENT, so it fails once
    with a reconnect instruction rather than burning the retry budget."""
    calls, state = slack_transport
    state["body"] = {"ok": False, "error": "token_revoked"}

    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled(ORG_A)
    repo.entitlements = ent
    req = await _approved_slack(repo)

    await execute_pass(repo, providers=production_providers(), policy=FAST, max_attempts=3)

    assert req.status == ExecutionStatus.FAILED.value
    assert len(calls) == 1, "a dead token must not be retried against Slack"
    assert req.attempts >= 3, "permanent: the retry budget is spent, not looped"
    assert len(repo.dead_letters) == 1
    assert "reconnect" in repo.dead_letters[0].error.lower()
    assert "token" not in repo.dead_letters[0].error.replace("token_revoked", "")


async def test_a_permanent_slack_business_error_names_the_cause(slack_transport):
    """`not_in_channel` is the founder's most likely real failure: Navis was never
    invited. The audit row must say so — and must not retry."""
    calls, state = slack_transport
    state["body"] = {"ok": False, "error": "not_in_channel"}

    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled(ORG_A)
    repo.entitlements = ent
    req = await _approved_slack(repo)

    await execute_pass(repo, providers=production_providers(), policy=FAST, max_attempts=3)

    assert req.status == ExecutionStatus.FAILED.value
    assert "not_in_channel" in (req.last_error or "")
    assert len(calls) == 1


async def test_org_isolation_holds_for_slack(slack_transport):
    calls, _state = slack_transport
    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled(ORG_A, ORG_B)
    repo.entitlements = ent
    req_a = await _approved_slack(repo, ORG_A)
    req_b = await _approved_slack(repo, ORG_B)

    await ent.set(ORG_A, Capability.SLACK, False)

    await execute_pass(repo, providers=production_providers(), policy=FAST, max_attempts=3)

    assert req_a.status == ExecutionStatus.FAILED.value
    assert req_b.status == ExecutionStatus.EXECUTED.value
    assert len(calls) == 1, "exactly one org reached Slack"
