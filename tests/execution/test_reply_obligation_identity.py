"""ONE inbound Gmail message = ONE reply obligation = at most ONE sendable execution.

These tests pin the exact defect found in production. `idempotency_key` hashed
(org, decision_id, action_type, payload) — and for a reply BOTH of those carry
observation-scoped values:

  * decision_id = uuid5(org | reasoning_version | state_watermark | signal_id),
    and the watermark moves on every new memory record anywhere in the org;
  * the payload carries `waiting_days`, which drifts daily.

Measured on the real database, one unanswered message across three briefs:

    brief 5a537645  waiting_days=2.3  decision=842732eb  key=63418e25
    brief 62059d13  waiting_days=2.3  decision=53878738  key=4d348a78
    brief f12eab92  waiting_days=5.9  decision=cbb65303  key=e800950b

Three keys, three execution ids, three independently approvable rows into the
same Gmail thread. The first two differ ONLY by state watermark — so the
duplicate did not even need a day to pass.

Every test below is written so that it FAILS against the pre-fix key.
"""
from __future__ import annotations

import uuid

import pytest

from navis.execution.idempotency import execution_id, obligation_idempotency_key
from navis.execution.models import ExecutionOrigin
from navis.execution.obligation import reply_obligation_key
from navis.execution.proposer import propose_from_decisions
from navis.execution.service import ExecutionService
from navis.execution.status import ExecutionStatus
from tests.execution.fakes import ORG_A, FakeExecutionRepo

THREAD = "19fdb57e89581d43"
MESSAGE = "<calendar-b7706463-b4ea-4492-b1aa-89c6dfe8@google.com>"
OTHER_MESSAGE = "<CACoxYfgQQVZ1K=2OQOXvYjZ6-ms9Gp4Fxowp=s9N@mail.gmail.com>"

# The three decision_ids the real briefs actually produced for ONE message.
D1 = uuid.UUID("842732eb-49ee-5006-ac74-596f29aca2f6")
D2 = uuid.UUID("53878738-de5f-5155-b01c-cdd21f13d8d4")
D3 = uuid.UUID("cbb65303-83d4-55fb-9e4d-e45c1a17efb8")


def payload(**over) -> dict:
    """A waiting_reply payload as execution/proposer.py builds it."""
    base = {
        "to": ["mulayamabmrc@gmail.com"],
        "person_email": "mulayamabmrc@gmail.com",
        "person_name": "Mulayam",
        "thread_id": THREAD,
        "reply_thread_id": THREAD,
        "in_reply_to_message_id": MESSAGE,
        "subject": "Invitation: Meeting",
        "body": "",
        "requires_draft": True,
        "candidate_kind": "waiting_reply",
        "waiting_days": 2.3,
        "last_inbound_at": "2026-08-07T08:29:58+00:00",
        "decision_id": str(D1),
    }
    base.update(over)
    return base


async def request(svc, *, decision_id=D1, watermark="w-1", **payload_over):
    return await svc.request(
        ORG_A,
        decision_id,
        "reply_to_email",
        payload(**payload_over),
        reasoning_signal_ids=[],
        reasoning_version=2,
        state_watermark=watermark,
        origin=ExecutionOrigin.PROACTIVE,
    )


@pytest.fixture()
def svc():
    return ExecutionService(repo=FakeExecutionRepo())


# ── The identity itself ────────────────────────────────────────────────


class TestObligationKey:
    def test_names_the_message_being_answered(self):
        key = reply_obligation_key("reply_to_email", payload())
        assert key == f"reply|waiting_reply|{THREAD}|{MESSAGE.strip('<>')}"

    @pytest.mark.parametrize(
        "field, value",
        [
            ("waiting_days", 6.6),
            ("last_inbound_at", "2026-01-01T00:00:00+00:00"),
            ("decision_id", str(D3)),
            ("subject", "a completely different subject"),
            ("person_name", "Someone Else"),
        ],
    )
    def test_ignores_everything_that_is_not_the_obligation(self, field, value):
        """Observation-scoped and cosmetic fields must not change identity."""
        assert reply_obligation_key("reply_to_email", payload(**{field: value})) == (
            reply_obligation_key("reply_to_email", payload())
        )

    def test_angle_brackets_are_delimiters_not_identity(self):
        bare = reply_obligation_key("reply_to_email", payload(in_reply_to_message_id=MESSAGE.strip("<>")))
        assert bare == reply_obligation_key("reply_to_email", payload())

    def test_a_different_message_is_a_different_obligation(self):
        assert reply_obligation_key(
            "reply_to_email", payload(in_reply_to_message_id=OTHER_MESSAGE)
        ) != reply_obligation_key("reply_to_email", payload())

    def test_a_different_thread_is_a_different_obligation(self):
        assert reply_obligation_key(
            "reply_to_email", payload(thread_id="other", reply_thread_id="other")
        ) != reply_obligation_key("reply_to_email", payload())

    def test_a_different_kind_on_one_thread_is_a_different_obligation(self):
        # "they're waiting" and "chase the commitment" are genuinely separate
        # obligations that can legitimately co-exist on one thread.
        assert reply_obligation_key(
            "reply_to_email", payload(candidate_kind="decision_owed")
        ) != reply_obligation_key("reply_to_email", payload())

    @pytest.mark.parametrize("missing", ["in_reply_to_message_id", "thread_id", "candidate_kind"])
    def test_no_identity_without_every_component(self, missing):
        p = payload()
        p[missing] = None
        if missing == "thread_id":
            p["reply_thread_id"] = None
        assert reply_obligation_key("reply_to_email", p) is None

    @pytest.mark.parametrize("blank", ["", "   ", "<>"])
    def test_blank_message_id_is_not_an_identity(self, blank):
        assert reply_obligation_key("reply_to_email", payload(in_reply_to_message_id=blank)) is None

    def test_scoped_to_replies_only(self):
        # Calendar keeps the existing payload-hash semantics untouched: Google's
        # events.insert already gives scheduling an exactly-once guarantee.
        assert reply_obligation_key("schedule_meeting", payload()) is None
        assert reply_obligation_key("send_email", payload()) is None


# ── Same message, many observations -> ONE execution ───────────────────


class TestSameMessageResolvesToOneExecution:
    async def test_different_decision_id_same_execution(self, svc):
        a, created_a = await request(svc, decision_id=D1)
        b, created_b = await request(svc, decision_id=D2)
        assert a.id == b.id
        assert (created_a, created_b) == (True, False)

    async def test_different_state_watermark_same_execution(self, svc):
        # The sharpest case from production: identical evidence, identical
        # waiting_days, only the watermark moved.
        a, _ = await request(svc, decision_id=D1, watermark="w-1")
        b, created = await request(svc, decision_id=D2, watermark="w-2")
        assert a.id == b.id
        assert created is False

    async def test_different_waiting_days_same_execution(self, svc):
        a, _ = await request(svc, waiting_days=2.3)
        b, created = await request(svc, decision_id=D3, waiting_days=5.9)
        assert a.id == b.id
        assert created is False

    async def test_the_three_real_briefs_produce_one_execution(self, svc):
        """The exact production sequence: brief A -> B -> C, one message."""
        rows = [
            await request(svc, decision_id=D1, watermark="2e482801", waiting_days=2.3),
            await request(svc, decision_id=D2, watermark="e06b78b1", waiting_days=2.3),
            await request(svc, decision_id=D3, watermark="e910a36d", waiting_days=5.9),
        ]
        assert len({r.id for r, _ in rows}) == 1
        assert [created for _, created in rows] == [True, False, False]

    async def test_repeated_proposal_creates_exactly_one(self, svc):
        for i in range(10):
            await request(svc, decision_id=uuid.uuid4(), watermark=f"w-{i}", waiting_days=i / 3)
        assert len(svc.repo.rows) == 1

    async def test_lineage_is_preserved_on_the_surviving_row(self, svc):
        """decision_id stays for audit — it just no longer defines identity."""
        a, _ = await request(svc, decision_id=D1)
        assert a.decision_id == D1
        assert a.obligation_key == reply_obligation_key("reply_to_email", payload())

    async def test_a_new_inbound_message_is_a_new_execution(self, svc):
        """THE distinction that must survive: same thread, new message, new work."""
        a, _ = await request(svc)
        b, created = await request(
            svc, decision_id=D3, in_reply_to_message_id=OTHER_MESSAGE
        )
        assert a.id != b.id
        assert created is True
        assert len(svc.repo.rows) == 2


# ── Existing executions are reused, terminal ones are never resurrected ─


class TestExistingExecutionStates:
    @pytest.mark.parametrize(
        "status",
        [
            ExecutionStatus.REQUESTED,
            ExecutionStatus.APPROVED,
            ExecutionStatus.EXECUTING,
            ExecutionStatus.EXECUTED,
            ExecutionStatus.REJECTED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.FAILED,
        ],
    )
    async def test_reuses_the_row_whatever_state_it_is_in(self, svc, status):
        existing, _ = await request(svc)
        existing.status = status.value

        again, created = await request(svc, decision_id=D3, waiting_days=9.9)

        # One row, always. Never a second sendable copy, and never a rewrite of
        # a human's REJECTED/CANCELLED decision back into something approvable.
        assert again.id == existing.id
        assert created is False
        assert len(svc.repo.rows) == 1
        assert again.status == status.value

    async def test_executed_never_yields_a_second_send(self, svc):
        done, _ = await request(svc)
        done.status = ExecutionStatus.EXECUTED.value
        done.provider_reference = {"provider": "gmail", "external_id": "19feb69c55c0a6f6"}

        again, created = await request(svc, decision_id=D3, watermark="later")

        assert created is False
        assert again.id == done.id
        assert again.provider_reference["external_id"] == "19feb69c55c0a6f6"


# ── The proposer refuses what it cannot identify ───────────────────────


def decision(category: str, **evidence_over) -> dict:
    evidence = {
        "person_email": "mulayamabmrc@gmail.com",
        "person_name": "Mulayam",
        "thread_id": THREAD,
        "rfc_message_id": MESSAGE,
        "subject": "Invitation: Meeting",
        "waiting_days": 2.3,
        "last_inbound_at": "2026-08-07T08:29:58+00:00",
        "commitment": "I'll send the deck",
        "due_at": "2026-08-08T00:00:00+00:00",
        "decision": "ship or hold?",
    }
    evidence.update(evidence_over)
    return {
        "decision_id": str(D1),
        "category": category,
        "title": "Mulayam is waiting for your reply",
        "evidence": evidence,
    }


class TestProposerWithholdsUnidentifiableReplies:
    def test_waiting_reply_is_proposed(self):
        out = propose_from_decisions([decision("waiting_reply")], ORG_A)
        assert len(out) == 1
        assert reply_obligation_key(out[0].action_type, out[0].payload) is not None

    def test_decision_owed_is_proposed(self):
        assert len(propose_from_decisions([decision("decision_owed")], ORG_A)) == 1

    def test_a_reply_without_a_message_id_is_withheld(self):
        """No stable identity -> no execution, rather than an invented one.

        Refusing to act costs a nudge. Inventing an identity costs a duplicate
        email to a customer, which is strictly worse.
        """
        out = propose_from_decisions([decision("waiting_reply", rfc_message_id=None)], ORG_A)
        assert out == []

    def test_commitment_follow_up_is_identified_by_the_promise(self):
        # CommitmentState carries no rfc_message_id at all (only ThreadState and
        # DecisionState do — state/v2/models.py), so a follow-up cannot name a
        # message. It names the PROMISE instead: a content hash using the same
        # normalisation Memory applies to the same sentence. Deterministic and
        # brief-invariant, so a legitimate stable identity rather than an
        # invented one.
        out = propose_from_decisions([decision("follow_up")], ORG_A)
        assert len(out) == 1
        key = reply_obligation_key(out[0].action_type, out[0].payload)
        assert key is not None
        assert key.startswith(f"reply|commitment_follow_up|{THREAD}|commitment:")

    def test_a_follow_up_without_the_promise_text_is_withheld(self):
        assert propose_from_decisions([decision("follow_up", commitment=None)], ORG_A) == []

    def test_the_same_promise_is_one_obligation_however_overdue(self):
        """`overdue_days` drifts exactly as `waiting_days` does — it must not count."""
        a = propose_from_decisions([decision("follow_up", overdue_days=1.0)], ORG_A)[0]
        b = propose_from_decisions([decision("follow_up", overdue_days=9.0)], ORG_A)[0]
        assert reply_obligation_key(a.action_type, a.payload) == reply_obligation_key(
            b.action_type, b.payload
        )

    def test_case_and_whitespace_do_not_fork_one_promise(self):
        b = propose_from_decisions(
            [decision("follow_up", commitment="  I'LL   SEND the deck  ")], ORG_A
        )[0]
        c = propose_from_decisions(
            [decision("follow_up", commitment="I'll send the deck")], ORG_A
        )[0]
        assert reply_obligation_key(b.action_type, b.payload) == reply_obligation_key(
            c.action_type, c.payload
        )

    def test_a_different_promise_is_a_different_obligation(self):
        a = propose_from_decisions([decision("follow_up")], ORG_A)[0]
        b = propose_from_decisions(
            [decision("follow_up", commitment="I'll send the contract")], ORG_A
        )[0]
        assert reply_obligation_key(a.action_type, a.payload) != reply_obligation_key(
            b.action_type, b.payload
        )

    def test_every_proposed_reply_can_be_de_duplicated(self):
        """The invariant the whole fix rests on."""
        out = propose_from_decisions(
            [decision(c) for c in ("waiting_reply", "decision_owed", "follow_up", "risk")],
            ORG_A,
        )
        assert out  # not vacuous
        for p in out:
            assert reply_obligation_key(p.action_type, p.payload) is not None


# ── Determinism of the derived ids ─────────────────────────────────────


class TestDeterminism:
    def test_same_obligation_same_execution_id_across_processes(self):
        """No randomness, no clock: the id is a pure function of the obligation."""
        obl = reply_obligation_key("reply_to_email", payload())
        first = execution_id(ORG_A, obligation_idempotency_key(ORG_A, obl))
        second = execution_id(ORG_A, obligation_idempotency_key(ORG_A, obl))
        assert first == second

    def test_obligation_keys_do_not_collide_with_payload_hashes(self):
        from navis.execution.idempotency import idempotency_key

        obl = reply_obligation_key("reply_to_email", payload())
        assert obligation_idempotency_key(ORG_A, obl) != idempotency_key(
            ORG_A, D1, "reply_to_email", payload()
        )

    def test_one_obligation_is_scoped_per_org(self):
        from tests.execution.fakes import ORG_B

        obl = reply_obligation_key("reply_to_email", payload())
        assert obligation_idempotency_key(ORG_A, obl) != obligation_idempotency_key(ORG_B, obl)


# ── The outbound Gmail Message-ID inherits the guarantee ───────────────


class TestGmailMessageIdStillDeterministic:
    def test_retries_of_one_obligation_keep_one_message_id(self):
        """The duplicate-send pre-check keys on execution_id, which is now
        derived FROM the obligation — so it inherits the stability for free and
        needed no change of its own."""
        from navis.execution.providers._senders import navis_message_id

        obl = reply_obligation_key("reply_to_email", payload())
        eid = execution_id(ORG_A, obligation_idempotency_key(ORG_A, obl))
        assert navis_message_id(eid) == navis_message_id(eid)

    def test_a_new_inbound_message_gets_a_different_message_id(self):
        from navis.execution.providers._senders import navis_message_id

        def mid(message_id: str) -> str:
            obl = reply_obligation_key(
                "reply_to_email", payload(in_reply_to_message_id=message_id)
            )
            return navis_message_id(execution_id(ORG_A, obligation_idempotency_key(ORG_A, obl)))

        assert mid(MESSAGE) != mid(OTHER_MESSAGE)
