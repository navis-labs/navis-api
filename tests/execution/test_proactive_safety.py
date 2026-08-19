"""Phase 1 safety rails for PROACTIVE actions.

Three things shipped that should never have been approvable:

  1. `proposer._map` turned every risk/opportunity decision with a known contact
     into `send_email` with `body: ""`, and GmailProvider only checked `to` and
     `subject`. One click emailed a real person a blank message whose subject
     was the decision title ("Commitment slipping: I'll get you the deck by
     Friday") — sent TO the person who said it.

  2. `TasksProvider` hashed a title into a synthetic id and returned ok=True, so
     the worker recorded EXECUTED for work filed in a system that does not
     exist.

  3. `GET /actions` returned every REQUESTED row regardless of who proposed it,
     so a meeting the founder had just requested in chat also appeared on the
     "things Navis noticed" board, approvable from either surface.

These tests exist so none of the three can come back quietly.
"""
from __future__ import annotations

import uuid

import pytest

from navis.execution.models import ActionType, ExecutionOrigin
from navis.execution.providers import GmailProvider, TasksProvider, get_provider
from navis.execution.providers.base import ProviderValidationError
from navis.execution.providers.reply import (
    NEEDS_DRAFT_MESSAGE,
    ReplyToEmailProvider,
    requires_draft,
)
from navis.execution.proposer import propose_from_decisions
from navis.execution.service import ExecutionService
from navis.execution.worker import execute_pass
from tests.execution.fakes import DECISION, ORG_A, USER, FakeExecutionRepo


def _decision(category: str, **over):
    d = {
        "decision_id": str(uuid.uuid4()),
        "category": category,
        "title": "Commitment slipping: I'll get you the deck by Friday",
        "evidence": {"speaker_email": "arjun@fund.com", "person_email": "arjun@fund.com"},
        "reasoning_signal_ids": [],
        "reasoning_version": 1,
        "state_watermark": None,
    }
    d.update(over)
    return d


# ── 1. Blank email ─────────────────────────────────────────────────────


async def test_gmail_provider_refuses_an_empty_body():
    p = GmailProvider()
    with pytest.raises(ProviderValidationError, match="non-empty 'body'"):
        await p.validate({"to": "a@b.com", "subject": "Hi", "body": ""})


@pytest.mark.parametrize("body", [None, "", "   ", "\n", "\t  \n "])
async def test_whitespace_only_bodies_are_also_refused(body):
    """"Blank" is not just the empty string — a body of spaces reads as blank in
    every mail client."""
    p = GmailProvider()
    with pytest.raises(ProviderValidationError):
        await p.validate({"to": "a@b.com", "subject": "Hi", "body": body})


async def test_a_real_body_still_passes():
    await GmailProvider().validate(
        {"to": "a@b.com", "subject": "Hi", "body": "Here's the deck."}
    )


@pytest.mark.parametrize("category", ["risk", "opportunity"])
def test_proposer_no_longer_emits_blank_emails(category):
    """The exact input that used to produce the blank email."""
    proposed = propose_from_decisions([_decision(category)], ORG_A)
    assert proposed == [], (
        "risk/opportunity must propose nothing until drafting exists — "
        "an email with no body is not a proposal, it is an accident"
    )


def test_no_proposal_anywhere_carries_an_empty_sendable_body():
    """Belt and braces across EVERY category: whatever the proposer emits, it is
    never a send_email, and never sendable-but-blank."""
    categories = ["risk", "opportunity", "delegation", "priority", "context"]
    for p in propose_from_decisions([_decision(c) for c in categories], ORG_A):
        assert p.action_type != ActionType.SEND_EMAIL.value
        if not p.payload.get("requires_draft"):
            assert str(p.payload.get("body") or "").strip()


async def test_approving_a_malformed_email_row_cannot_send():
    """End-to-end: even if a blank-body send_email row exists (legacy row, bad
    caller), approving it fails validation and never reaches a sender."""
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        ORG_A, DECISION, ActionType.SEND_EMAIL.value,
        {"to": "arjun@fund.com", "subject": "Commitment slipping", "body": ""},
        reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
    )
    await svc.approve(req.id, USER)

    sent: list[dict] = []

    async def _sender(payload):
        sent.append(payload)
        return "msg-1"

    result = await execute_pass(
        repo, providers={"send_email": GmailProvider(sender=_sender)}
    )

    assert sent == [], "nothing may reach Gmail"
    assert result["failed"] == 1
    assert req.status == "failed"
    assert "body" in (req.last_error or "")
    # Non-retryable, so it dead-letters instead of retrying into existence.
    assert len(repo.dead_letters) == 1


# ── 2. Fake tasks ──────────────────────────────────────────────────────


async def test_create_task_refuses_instead_of_faking_success():
    with pytest.raises(ProviderValidationError, match="no task system"):
        await TasksProvider().validate({"title": "Delegate: ship the deck"})


async def test_create_task_never_reports_executed():
    """The precise old behaviour: EXECUTED, with a task id, for a task that was
    never created anywhere."""
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        ORG_A, DECISION, ActionType.CREATE_TASK.value, {"title": "Delegate: ship the deck"},
        reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
    )
    await svc.approve(req.id, USER)

    result = await execute_pass(repo)

    assert result["executed"] == 0
    assert req.status == "failed"
    assert not req.provider_reference, "no fabricated task id"
    assert "no task system" in (req.last_error or "")


def test_proposer_never_emits_create_task():
    for category in ("delegation", "risk", "opportunity"):
        proposed = propose_from_decisions([_decision(category)], ORG_A)
        assert all(p.action_type != ActionType.CREATE_TASK.value for p in proposed)


# ── 3. Reply candidates cannot send without a draft ────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        {"requires_draft": True, "body": "a real body"},  # flag wins
        {"body": ""},                                     # empty body
        {"body": "   "},                                  # whitespace body
        {},                                               # nothing at all
        None,
    ],
)
def test_requires_draft_detects_every_placeholder_shape(payload):
    assert requires_draft(payload) is True


def test_a_written_reply_is_not_a_placeholder():
    assert requires_draft({"body": "Sorry for the delay — Friday works."}) is False


async def test_reply_provider_refuses_a_draftless_reply():
    p = ReplyToEmailProvider()
    with pytest.raises(ProviderValidationError, match="no draft"):
        await p.validate(
            {
                "to": ["a@b.com"],
                "reply_thread_id": "t-1",
                "subject": "Re: deck",
                "body": "",
                "requires_draft": True,
            }
        )


async def test_reply_provider_requires_the_thread_it_answers():
    p = ReplyToEmailProvider()
    with pytest.raises(ProviderValidationError, match="reply_thread_id"):
        await p.validate({"to": ["a@b.com"], "subject": "Re: deck", "body": "hi"})


async def test_a_drafted_reply_validates():
    await ReplyToEmailProvider().validate(
        {
            "to": ["a@b.com"],
            "reply_thread_id": "t-1",
            "subject": "Re: deck",
            "body": "Sorry for the delay — Friday works.",
        }
    )


async def test_approving_a_draftless_reply_cannot_send():
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        ORG_A, DECISION, ActionType.REPLY_TO_EMAIL.value,
        {
            "to": ["arjun@fund.com"], "reply_thread_id": "t-1",
            "subject": "Re: deck", "body": "", "requires_draft": True,
        },
        reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
        origin=ExecutionOrigin.PROACTIVE,
    )
    await svc.approve(req.id, USER)

    sent: list[dict] = []

    async def _sender(payload):
        sent.append(payload)
        return "msg-1"

    await execute_pass(
        repo, providers={"reply_to_email": ReplyToEmailProvider(sender=_sender)}
    )

    assert sent == []
    assert req.status == "failed"
    assert NEEDS_DRAFT_MESSAGE.split(",")[0] in (req.last_error or "")


def test_reply_provider_is_registered():
    assert get_provider(ActionType.REPLY_TO_EMAIL.value) is not None


# ── 4. Chat vs proactive origin ────────────────────────────────────────


async def test_origin_defaults_to_chat():
    """The safe default: an unlabelled row stays out of the proactive queue."""
    repo = FakeExecutionRepo()
    req, _ = await ExecutionService(repo=repo).request(
        ORG_A, DECISION, ActionType.SCHEDULE_MEETING.value, {"title": "Sync", "start": "x"},
        reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
    )
    assert req.origin == ExecutionOrigin.CHAT.value


async def test_origin_is_recorded_when_supplied():
    repo = FakeExecutionRepo()
    req, _ = await ExecutionService(repo=repo).request(
        ORG_A, DECISION, ActionType.REPLY_TO_EMAIL.value, {"to": ["a@b.com"]},
        reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
        origin=ExecutionOrigin.PROACTIVE,
    )
    assert req.origin == ExecutionOrigin.PROACTIVE.value


async def test_origin_does_not_split_the_idempotency_key():
    """The same decision + payload is the SAME action whoever noticed it.

    If origin fed the key, one meeting could be booked twice — once from chat
    and once from the proactive queue — which is the opposite of what the
    idempotency machinery exists for.
    """
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    payload = {"title": "Sync", "start": "2026-08-11T10:00:00+00:00"}
    first, created_first = await svc.request(
        ORG_A, DECISION, ActionType.SCHEDULE_MEETING.value, payload,
        reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
        origin=ExecutionOrigin.CHAT,
    )
    second, created_second = await svc.request(
        ORG_A, DECISION, ActionType.SCHEDULE_MEETING.value, payload,
        reasoning_signal_ids=[], reasoning_version=None, state_watermark=None,
        origin=ExecutionOrigin.PROACTIVE,
    )
    assert created_first is True
    assert created_second is False
    assert first.id == second.id


# ── Gmail send scope (Phase 2B E2E readiness) ──────────────────────────
#
# The whole proactive-reply path was unshippable against a real account: every
# org was connected under `gmail.readonly`, so `messages.send` would 403 AFTER
# the founder had reviewed, edited and approved a draft — dead-lettering with an
# opaque Google error at the last possible step.
#
# The fix mirrors the calendar write-scope handling exactly: request the scope,
# and fail loud BEFORE calling Google.


def test_the_send_scope_is_actually_requested_at_consent():
    from navis.signals.google.oauth import GOOGLE_SCOPES, GOOGLE_SCOPE_VERSION

    assert "https://www.googleapis.com/auth/gmail.send" in GOOGLE_SCOPES
    # The version is persisted with the integration so an org connected under an
    # older set can be detected and prompted to reconnect, rather than being
    # assumed to have access it does not have.
    assert GOOGLE_SCOPE_VERSION != "v3", "adding a scope must bump the version"


def test_read_only_gmail_does_not_grant_send():
    """The exact state every existing org is in."""
    from navis.signals.google.oauth import gmail_send_granted

    assert not gmail_send_granted("https://www.googleapis.com/auth/gmail.readonly")
    assert not gmail_send_granted(None)
    assert not gmail_send_granted("")


@pytest.mark.parametrize(
    "scope",
    [
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://mail.google.com/",
    ],
)
def test_scopes_that_do_grant_send(scope):
    """Broader grants are accepted so an org that already has one is not forced
    through a pointless reconnect."""
    from navis.signals.google.oauth import gmail_send_granted

    assert gmail_send_granted(f"https://www.googleapis.com/auth/gmail.readonly {scope}")


async def test_the_sender_refuses_before_calling_google_when_scope_is_missing():
    """Fail loud, with an instruction — not a bare 403 from inside a retry."""
    from types import SimpleNamespace

    from navis.execution.providers import _senders

    class _Integration:
        provider = "gmail"
        scopes = "https://www.googleapis.com/auth/gmail.readonly"
        access_token = "ct"
        refresh_token = "cr"
        token_expires_at = None
        external_account = "founder@acme.com"

    class _CM:
        async def __aenter__(self):
            return SimpleNamespace(
                scalar=self._one, scalars=self._many, commit=self._noop, rollback=self._noop
            )

        async def __aexit__(self, *a):
            return False

        async def _one(self, _stmt):
            return _Integration()

        async def _many(self, _stmt):
            return SimpleNamespace(all=lambda: [_Integration()])

        async def _noop(self):
            return None

    sent: list = []

    async def _never(*a, **kw):
        sent.append(a)
        raise AssertionError("Google must not be called when the scope is missing")

    import pytest as _pytest

    monkey = _pytest.MonkeyPatch()
    try:
        monkey.setattr(_senders, "session_scope", lambda: _CM())
        # No decrypt step to stub: the sender snapshots the tokens AS STORED and
        # hands the ciphertext to TokenManager. Nothing reads these stub tokens
        # before the scope check refuses.
        monkey.setattr(_senders, "GmailClient", _never)
        # The guarantee under test is unchanged and is the assertion below plus
        # `_never` — refuse BEFORE any Google call.
        #
        # The sender now resolves through the generic integration boundary, so
        # the Google-specific GmailSendScopeInsufficient is translated into the
        # provider-neutral IntegrationUnavailable that the whole Execution layer
        # understands. Asserting the flag too is strictly stronger than the old
        # message match: `reconsent_required` is what makes the UI say "reconnect
        # to grant send" rather than the wrong "connect Google".
        from navis.execution.integrations import IntegrationUnavailable

        with _pytest.raises(IntegrationUnavailable, match="not allowed to send") as exc:
            await _senders.gmail_sender(
                {
                    "__ctx__": {
                        "org_id": str(ORG_A),
                        "execution_id": str(uuid.uuid4()),
                        "decision_id": str(DECISION),
                    },
                    "to": ["a@b.com"],
                    "subject": "Re: x",
                    "body": "a real body",
                    "reply_thread_id": "t-1",
                }
            )
        assert exc.value.provider == "gmail"
        assert exc.value.reconsent_required is True
        assert sent == []
    finally:
        monkey.undo()
