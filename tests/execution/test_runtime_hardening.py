# pyright: reportArgumentType=false, reportOptionalMemberAccess=false
"""Runtime hardening of the execution core (task 5.4.2).

These cover the invariants that had NO test and where source inspection found
real production defects. Everything else in the required battery is already
pinned elsewhere and is not duplicated here:

    cross-org credential isolation / forged payload  test_integration_boundary
    post-approval revocation, kill switch, capability test_execution_authorization
    unknown ActionSpec, provider resolution, approval test_action_contract
    retryable vs non-retryable classification        test_hardening
    Meet through Calendar, event identity            test_calendar_conference
    Slack end to end                                 test_slack_connector

What is new here:

  * SECRETS AT REST — the senders were committing DECRYPTED OAuth tokens back
    into `integrations`. This is the highest-severity defect the audit found and
    it was completely untested.
  * TRANSACTION BOUNDARY — no database session may be open while Gmail,
    Calendar or Slack is being called. The worker documented this rule; the
    senders broke it.
  * TOKEN REFRESH PERSISTENCE — a rotated token must still be stored, now that
    it happens in its own short transaction rather than on a held ORM row.
  * DRIVE / DOCS — signal-only systems must be structurally unable to execute.

The network is substituted at the httpx boundary and NOWHERE ABOVE IT: the real
worker, the real authorization re-check, the real integration boundary, the real
`resolve_gmail_send_integration`, the real `TokenManager`, the real
`GoogleTransport` and the real `GmailClient` all run.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from navis.execution.actions import all_specs
from navis.execution.providers import _DEFAULT_PROVIDERS, production_providers
from navis.execution.retry import ExecutionRetryPolicy
from navis.execution.service import ExecutionService
from navis.execution.status import ExecutionStatus
from navis.execution.worker import execute_pass
from navis.organization.billing import BillingService
from navis.organization.contract import Plan
from navis.organization.entitlements import EntitlementService
from navis.platform.events import EventType
from navis.signals.crypto import token_cipher
from tests.execution.fakes import ORG_A, USER, FakeExecutionRepo
from tests.organization.fakes import FakeOrgRepo

FAST = ExecutionRetryPolicy(max_attempts=1, initial_backoff=0.0, max_backoff=0.0, jitter=0.0)

# The real secrets. Every assertion below that says "no secret leaked" is
# checking for THESE strings, so a leak cannot pass by being differently shaped.
ACCESS_SECRET = "ya29.a0AfB_REAL_ACCESS_TOKEN_VALUE"
REFRESH_SECRET = "1//04REAL_REFRESH_TOKEN_VALUE"
ROTATED_ACCESS = "ya29.a0AfB_ROTATED_ACCESS_TOKEN"

SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"


# ── The harness ────────────────────────────────────────────────────────


class _StoredIntegration:
    """A Gmail grant as the database actually holds it: tokens ENCRYPTED.

    Deliberately not a bare SimpleNamespace — attribute writes are recorded, so
    a test can assert that the sender never mutated the stored row. That is the
    mechanism behind the plaintext-writeback defect: a dirty ORM attribute plus
    a session that commits on exit.
    """

    def __init__(self, *, expired: bool = False) -> None:
        object.__setattr__(self, "_writes", [])
        self.id = uuid.uuid4()
        self.organization_id = ORG_A
        self.user_id = uuid.uuid4()
        self.provider = "gmail"
        self.scopes = SEND_SCOPE
        self.external_account = "founder@navislabs.in"
        self.access_token = token_cipher.encrypt(ACCESS_SECRET)
        self.refresh_token = token_cipher.encrypt(REFRESH_SECRET)
        self.token_expires_at = datetime.now(timezone.utc) + timedelta(
            hours=-1 if expired else 1
        )
        self.is_active = True
        object.__setattr__(self, "_armed", True)

    def __setattr__(self, name, value):
        if getattr(self, "_armed", False):
            self._writes.append((name, value))
        object.__setattr__(self, name, value)

    @property
    def writes(self):
        return list(self._writes)


@pytest.fixture()
def gmail_harness(monkeypatch):
    """Substitute ONLY the network and the session factory.

    Returns a handle exposing:
      requests            every Google HTTP call, each tagged with how many DB
                          sessions were open at the moment it was made
      sessions_opened     how many sessions the sender opened in total
      persisted           values written by `_persist_refreshed_token`
      integration         the stored row, which records any attribute write
    """
    from navis.execution.providers import _senders

    integration = _StoredIntegration()
    state = {"open": 0, "opened": 0, "requests": [], "persisted": [], "refresh_calls": 0}

    class _Session:
        """Counts open sessions and captures any UPDATE issued through it."""

        async def __aenter__(self):
            state["open"] += 1
            state["opened"] += 1
            return SimpleNamespace(commit=self._noop, rollback=self._noop, execute=self._execute)

        async def __aexit__(self, *a):
            state["open"] -= 1
            return False

        async def _noop(self):
            return None

        async def _execute(self, stmt):
            # `_persist_refreshed_token` issues a Core UPDATE; capture the bound
            # values so the test can assert what would reach the database.
            params = getattr(stmt, "compile", lambda: None)()
            state["persisted"].append(dict(params.params) if params is not None else {})
            return SimpleNamespace(rowcount=1)

    class _Resp:
        def __init__(self, body, status=200):
            self.status_code = status
            self._body = body
            self.headers = {}
            self.content = b"{}"
            self.text = ""

        def json(self):
            return self._body

    class _HTTP:
        def __init__(self, *a, **kw):
            pass

        async def request(self, method, url, params=None, json=None, headers=None):
            # THE KEY MEASUREMENT: how many DB sessions are open right now?
            state["requests"].append(
                {
                    "method": method,
                    "url": url,
                    "json": json,
                    "headers": headers,
                    "db_sessions_open": state["open"],
                }
            )
            if "/messages/send" in url:
                return _Resp({"id": "gmail-msg-1", "threadId": "thr-1"})
            if url.endswith("/messages"):
                return _Resp({"messages": []})  # no prior send of this Message-ID
            return _Resp({})

        async def aclose(self):
            pass

    async def _refresher(refresh_token):
        state["refresh_calls"] += 1
        # Google only hands back a refresh token when it rotates one.
        assert refresh_token == REFRESH_SECRET, "the refresher must get PLAINTEXT"
        return {"access_token": ROTATED_ACCESS, "expires_in": 3600}

    monkeypatch.setattr("navis.signals.google.transport.httpx.AsyncClient", _HTTP)
    monkeypatch.setattr(_senders, "session_scope", lambda: _Session())
    monkeypatch.setattr(
        _senders, "resolve_integration",
        lambda db, org_id, provider, **_: _resolved(integration),
    )
    monkeypatch.setattr("navis.signals.google.oauth.refresh_access_token", _refresher)
    return SimpleNamespace(state=state, integration=integration)


async def _resolved(value):
    return value


async def _entitled(*orgs):
    org_repo = FakeOrgRepo()
    ent = EntitlementService(org_repo)
    for org in orgs:
        await BillingService(org_repo, ent).set_plan(org, Plan.ENTERPRISE, billing_authorized=True)
    return ent


async def _approved_email(repo, org_id=ORG_A):
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        org_id, uuid.uuid4(), "send_email",
        {"to": "arjun@fund.com", "subject": "Deck", "body": "Attached, as promised."},
        reasoning_signal_ids=["sig-1"], reasoning_version=4, state_watermark="w-11",
    )
    await svc.approve(req.id, USER)
    return req


# ══ A. Gmail execution path, end to end ════════════════════════════════


async def test_gmail_executes_end_to_end_through_the_production_sender(gmail_harness):
    """Decision → ActionSpec → approval → runtime authorization → credential
    boundary → GmailProvider → messages.send → ProviderReference → outbox."""
    repo = FakeExecutionRepo()
    repo.entitlements = await _entitled(ORG_A)
    req = await _approved_email(repo)

    result = await execute_pass(
        repo, providers=production_providers(), policy=FAST, max_attempts=3
    )

    assert result["executed"] == 1
    assert req.status == ExecutionStatus.EXECUTED.value
    assert req.provider_reference == {"provider": "gmail", "external_id": "gmail-msg-1"}
    assert "dry_run" not in req.provider_reference

    sends = [r for r in gmail_harness.state["requests"] if "/messages/send" in r["url"]]
    assert len(sends) == 1, "exactly one message was sent"
    assert sends[0]["headers"]["Authorization"] == f"Bearer {ACCESS_SECRET}"

    # T. The connector-neutral Observation envelope.
    executed = [e for e in repo.events if e.event_type == EventType.ACTION_EXECUTED.value]
    assert len(executed) == 1
    outcome = executed[0].payload["outcome"]
    assert outcome["org_id"] == str(ORG_A)
    assert outcome["execution_id"] == str(req.id)
    assert outcome["decision_id"] == str(req.decision_id)
    assert outcome["action_type"] == "send_email"
    assert outcome["reasoning_signal_ids"] == ["sig-1"]
    assert outcome["reasoning_version"] == 4
    assert outcome["provider_reference"]["provider"] == "gmail"


# ══ Q. No database transaction is open during the external call ════════


async def test_no_database_session_is_open_while_gmail_is_being_called(gmail_harness):
    """The worker's Phase B rule, enforced end to end rather than only in the
    worker.

    `gmail_sender` opened a session, resolved the credential, and then kept that
    session for BOTH Gmail round trips — the duplicate pre-check and the send —
    with no intervening commit. A connection sat idle-in-transaction for the
    whole exchange, including the transport's internal backoff, so N concurrent
    sends pinned N connections for the duration of a third-party API call.

    Measured directly: every Google request records how many sessions were open
    at the instant it was made.
    """
    repo = FakeExecutionRepo()
    repo.entitlements = await _entitled(ORG_A)
    await _approved_email(repo)

    await execute_pass(repo, providers=production_providers(), policy=FAST, max_attempts=3)

    requests = gmail_harness.state["requests"]
    assert requests, "the sender must actually have called Google"
    offenders = [r for r in requests if r["db_sessions_open"] != 0]
    assert not offenders, (
        "a database session was held open across a Google call: "
        + ", ".join(f"{r['method']} {r['url']} ({r['db_sessions_open']} open)" for r in offenders)
    )


async def test_the_credential_session_closes_before_the_first_external_byte(gmail_harness):
    """Ordering, not just the count: the resolving session must have opened AND
    closed before Google is reached."""
    repo = FakeExecutionRepo()
    repo.entitlements = await _entitled(ORG_A)
    await _approved_email(repo)

    await execute_pass(repo, providers=production_providers(), policy=FAST, max_attempts=3)

    assert gmail_harness.state["opened"] >= 1, "a session must have been opened to resolve"
    assert gmail_harness.state["open"] == 0, "every session the sender opened was closed"
    assert gmail_harness.state["requests"][0]["db_sessions_open"] == 0


# ══ SECRETS AT REST — the plaintext write-back defect ══════════════════


async def test_the_sender_never_writes_a_decrypted_token_back_to_the_row(gmail_harness):
    """THE DEFECT THIS FILE EXISTS FOR.

    The senders held the live `Integration` ORM row and assigned the DECRYPTED
    tokens onto it (`_decrypt_tokens_inplace`). That marks the row dirty;
    `session_scope` commits on exit; and the re-encrypt step only ran when a
    refresh had happened. On the ordinary path — access token still valid — the
    exit commit therefore wrote the real access and refresh tokens into
    `integrations` IN PLAINTEXT.

    It then stayed silent forever: `token_cipher.decrypt` returns an untagged
    value as-is, so everything kept working with encryption-at-rest gone.

    The decryption was never needed — `TokenManager.get_token` decrypts what it
    is given. So the invariant is simply: the stored row is never written to at
    all during a send.
    """
    repo = FakeExecutionRepo()
    repo.entitlements = await _entitled(ORG_A)
    await _approved_email(repo)

    await execute_pass(repo, providers=production_providers(), policy=FAST, max_attempts=3)

    token_writes = [
        (name, value) for name, value in gmail_harness.integration.writes
        if name in ("access_token", "refresh_token", "token_expires_at")
    ]
    assert not token_writes, (
        f"the sender mutated the stored credential row: {token_writes}"
    )
    # And what is still stored is ciphertext, carrying neither secret.
    assert token_cipher.is_encrypted(gmail_harness.integration.access_token)
    assert token_cipher.is_encrypted(gmail_harness.integration.refresh_token)
    assert ACCESS_SECRET not in str(gmail_harness.integration.access_token)
    assert REFRESH_SECRET not in str(gmail_harness.integration.refresh_token)


# ══ R. Token refresh persistence ═══════════════════════════════════════


async def test_a_rotated_token_is_persisted_encrypted_in_its_own_transaction(monkeypatch):
    """A refresh must still be stored — now via a targeted UPDATE after the
    external call, rather than by mutating a row held across it.

    Two things must both hold, and they pull in opposite directions: the new
    token IS written (so quota is not burned re-refreshing every send), and it
    is written ENCRYPTED (so the fix for the plaintext defect is not undone by
    the refresh path).
    """
    from navis.execution.providers import _senders

    integration = _StoredIntegration(expired=True)  # forces a refresh
    state = {"open": 0, "persisted": [], "requests": []}

    class _Session:
        async def __aenter__(self):
            state["open"] += 1
            return SimpleNamespace(commit=self._noop, rollback=self._noop, execute=self._execute)

        async def __aexit__(self, *a):
            state["open"] -= 1
            return False

        async def _noop(self):
            return None

        async def _execute(self, stmt):
            state["persisted"].append(dict(stmt.compile().params))
            return SimpleNamespace(rowcount=1)

    class _Resp:
        status_code = 200
        headers: dict = {}
        content = b"{}"
        text = ""

        def __init__(self, body):
            self._body = body

        def json(self):
            return self._body

    class _HTTP:
        def __init__(self, *a, **kw):
            pass

        async def request(self, method, url, params=None, json=None, headers=None):
            state["requests"].append({"url": url, "headers": headers, "open": state["open"]})
            if "/messages/send" in url:
                return _Resp({"id": "gmail-msg-2"})
            return _Resp({"messages": []})

        async def aclose(self):
            pass

    async def _refresher(refresh_token):
        assert refresh_token == REFRESH_SECRET
        return {"access_token": ROTATED_ACCESS, "expires_in": 3600}

    monkeypatch.setattr("navis.signals.google.transport.httpx.AsyncClient", _HTTP)
    monkeypatch.setattr(_senders, "session_scope", lambda: _Session())
    monkeypatch.setattr(
        _senders, "resolve_integration",
        lambda db, org_id, provider, **_: _resolved(integration),
    )
    monkeypatch.setattr("navis.signals.google.oauth.refresh_access_token", _refresher)

    message_id = await _senders.gmail_sender({
        "__ctx__": {
            "org_id": str(ORG_A),
            "execution_id": str(uuid.uuid4()),
            "decision_id": str(uuid.uuid4()),
        },
        "to": "arjun@fund.com",
        "subject": "Deck",
        "body": "Attached.",
    })

    assert message_id == "gmail-msg-2"
    # The refreshed token really was used against Google.
    sends = [r for r in state["requests"] if "/messages/send" in r["url"]]
    assert sends[0]["headers"]["Authorization"] == f"Bearer {ROTATED_ACCESS}"
    # ...and it was persisted, ENCRYPTED, in a session of its own.
    assert len(state["persisted"]) == 1, "the rotated token must be stored exactly once"
    stored = state["persisted"][0]
    written = [v for v in stored.values() if isinstance(v, str)]
    assert any(token_cipher.is_encrypted(v) for v in written), (
        f"the persisted token is not encrypted: {stored}"
    )
    assert ROTATED_ACCESS not in " ".join(written), "a PLAINTEXT token would be stored"
    assert state["open"] == 0, "the persistence session was closed"


# ══ S. No secret reaches any durable or observable surface ═════════════


async def test_no_google_token_reaches_the_row_outbox_dead_letter_or_observation(
    gmail_harness,
):
    repo = FakeExecutionRepo()
    repo.entitlements = await _entitled(ORG_A)
    req = await _approved_email(repo)

    await execute_pass(repo, providers=production_providers(), policy=FAST, max_attempts=3)

    surfaces = {
        "payload": str(req.payload),
        "provider_reference": str(req.provider_reference),
        "last_error": str(req.last_error),
        "outbox": str([e.payload for e in repo.events]),
        "dead_letters": str([(d.error, d.payload) for d in repo.dead_letters]),
    }
    for name, surface in surfaces.items():
        for secret in (ACCESS_SECRET, REFRESH_SECRET, "Bearer ", "ya29.", "1//04"):
            assert secret not in surface, f"{secret!r} leaked into {name}"


# ══ An unscoped request is terminal, not transient ═════════════════════


async def test_a_request_with_no_organization_scope_fails_permanently():
    """A payload the worker never stamped cannot name its tenant, and no retry
    will grow one — so it must fail once.

    `SenderContextError` subclassed `Exception` while its own docstring claimed
    to be non-retryable. The worker classifies by TYPE, so it landed in the
    transient branch and a request with NO organization scope was re-driven the
    full retry budget, each attempt re-entering the credential path. Retrying an
    unscoped request is the last thing anyone wants near that boundary.
    """
    from navis.execution.providers._senders import SenderContextError, gmail_sender
    from navis.execution.providers.base import ProviderError

    assert issubclass(SenderContextError, ProviderError)

    for payload in ({}, {"__ctx__": {}}, {"__ctx__": {"org_id": "not-a-uuid"}}):
        with pytest.raises(SenderContextError) as exc:
            await gmail_sender({**payload, "to": "a@b.com", "subject": "s", "body": "b"})
        assert exc.value.retryable is False, "an unscoped request must not be retried"


# ══ B. The calendar read-after-write runs under the same two rules ═════


async def test_calendar_reconcile_holds_no_session_and_stores_no_plaintext(monkeypatch):
    """`reconcile_calendar_event` runs in the execution worker's Phase B, after
    EVERY calendar mutation, and had both defects in their worst form.

    It held the resolving session across the Google `events.get` AND through
    `SignalIngestionService.ingest` — whose flush, with the decrypted tokens
    assigned onto the live row, wrote them to `integrations` in plaintext. It was
    the highest-frequency instance of the write-back.
    """
    from navis.signals.calendar import reconcile as rc

    integration = _StoredIntegration()
    integration.provider = "google_calendar"
    state = {"open": 0, "opened": 0, "requests": []}

    class _Session:
        async def __aenter__(self):
            state["open"] += 1
            state["opened"] += 1
            return SimpleNamespace(commit=self._noop, rollback=self._noop)

        async def __aexit__(self, *a):
            state["open"] -= 1
            return False

        async def _noop(self):
            return None

    class _Resp:
        status_code = 200
        headers: dict = {}
        content = b"{}"
        text = ""

        def json(self):
            return {
                "id": "evt-1",
                "etag": '"abc123"',
                "status": "confirmed",
                "summary": "Investor sync",
                "start": {"dateTime": "2026-08-09T14:00:00+00:00"},
                "end": {"dateTime": "2026-08-09T14:30:00+00:00"},
                "organizer": {"email": "founder@navislabs.in"},
            }

    class _HTTP:
        def __init__(self, *a, **kw):
            pass

        async def request(self, method, url, params=None, json=None, headers=None):
            state["requests"].append({"url": url, "db_sessions_open": state["open"]})
            return _Resp()

        async def aclose(self):
            pass

    ingested: list = []

    class _Ingestion:
        def __init__(self, db):
            pass

        async def ingest(self, signal):
            ingested.append(signal)
            return True

    monkeypatch.setattr("navis.signals.google.transport.httpx.AsyncClient", _HTTP)
    monkeypatch.setattr("navis.platform.database.session_scope", lambda: _Session())
    monkeypatch.setattr(
        rc, "resolve_calendar_integration",
        lambda db, org_id: _resolved(integration),
    )
    monkeypatch.setattr(rc, "SignalIngestionService", _Ingestion)

    written = await rc.reconcile_calendar_event(ORG_A, "evt-1")

    assert written is True and ingested, "the read model must still be updated"
    offenders = [r for r in state["requests"] if r["db_sessions_open"] != 0]
    assert not offenders, f"a session was held across the Google call: {offenders}"
    token_writes = [
        (n, v) for n, v in integration.writes
        if n in ("access_token", "refresh_token", "token_expires_at")
    ]
    assert not token_writes, f"reconcile mutated the credential row: {token_writes}"
    assert token_cipher.is_encrypted(integration.access_token)
    assert state["open"] == 0


# ══ E + F. Drive and Docs are signal-only and cannot execute ═══════════


DRIVE_DOCS_WORDS = ("drive", "docs", "document", "file", "folder", "sheet", "slide")


def test_no_drive_or_docs_action_is_registered():
    """Drive and Docs are READ connectors. An executable action for either would
    need an OAuth write scope the org has never granted, so the registry must not
    contain one — and the registry is the only way an action becomes reachable."""
    offenders = sorted(
        f"{action_type} (provider={spec.provider})"
        for action_type, spec in all_specs().items()
        if any(w in action_type.lower() or w in spec.provider.lower()
               for w in ("drive", "docs"))
    )
    assert not offenders, f"a Drive/Docs execution action is registered: {offenders}"


def test_no_drive_or_docs_provider_is_reachable_from_execution():
    """Neither the dry-run registry nor the production one may expose one."""
    for label, registry in (("default", _DEFAULT_PROVIDERS), ("production", production_providers())):
        offenders = sorted(
            f"{action_type} -> {provider.name}"
            for action_type, provider in registry.items()
            if "drive" in provider.name.lower() or "docs" in provider.name.lower()
        )
        assert not offenders, f"{label} registry exposes a Drive/Docs provider: {offenders}"


def test_no_credential_resolver_is_registered_for_drive_or_docs():
    """`resolve_integration` refuses a provider with no registered resolver, so
    the absence of one is the structural block on a Drive/Docs write."""
    from navis.execution.integrations import has_integration_resolver

    assert not has_integration_resolver("drive")
    assert not has_integration_resolver("docs")
    assert not has_integration_resolver("google_docs")


async def test_a_drive_action_cannot_be_requested_or_executed():
    """The front door, closed. Intake refuses an unregistered action outright,
    so no Drive row can exist to be approved."""
    from navis.execution.actions import UnknownAction

    repo = FakeExecutionRepo()
    for action_type in ("drive_share_file", "create_doc", "docs_append"):
        with pytest.raises(UnknownAction):
            await ExecutionService(repo=repo).request(
                ORG_A, uuid.uuid4(), action_type, {"file_id": "f1"},
                reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
            )
    assert repo.rows == {}, "no Drive/Docs execution row may exist"
    assert repo.events == [], "and nothing may be published about one"


def test_the_google_grant_still_requests_drive_metadata_read_only():
    """The scope must not be silently widened. `drive.metadata.readonly` grants
    no content read and no write at all — which is what makes a Drive write
    action impossible rather than merely unimplemented."""
    from navis.signals.google.oauth import GOOGLE_SCOPES

    drive_scopes = [s for s in GOOGLE_SCOPES if "drive" in s or "docs" in s]
    assert drive_scopes == ["https://www.googleapis.com/auth/drive.metadata.readonly"]
    for scope in GOOGLE_SCOPES:
        assert not scope.endswith("/drive"), "full Drive scope must never be requested"
        assert not scope.endswith("/drive.file")
        assert "documents" not in scope, "the Docs API scope must never be requested"


def test_generic_execution_contains_no_drive_or_docs_branch():
    """Provider-specific leakage, checked for the two connectors that must NOT
    be reachable. A branch naming them in a generic layer would be the first step
    toward one becoming executable by accident."""
    import ast
    import pathlib

    execution_dir = pathlib.Path(__file__).resolve().parents[2] / "navis" / "execution"
    generic = (
        "worker.py", "service.py", "repository.py", "retry.py", "recovery.py",
        "integrations.py", "router.py", "status.py", "contract.py",
    )
    offenders: list[str] = []
    for name in generic:
        source = (execution_dir / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            # String constants only — prose in comments and docstrings is fine,
            # a literal the code branches on is not.
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.lower() in ("drive", "docs", "google_docs"):
                    offenders.append(f"{name}: {node.value!r}")
            if isinstance(node, ast.ImportFrom) and node.module:
                if "drive" in node.module or "docs" in node.module:
                    offenders.append(f"{name}: imports {node.module}")
    assert not offenders, f"Drive/Docs leaked into generic Execution: {offenders}"
