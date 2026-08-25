"""Decision -> ExecutionRequest bridge.

Decision PROPOSES; Execution executes. This maps decisions (the JSON dicts stored
on a DailyBrief) to candidate actions — deterministically, with NO content
generation and NO LLM. Every proposal stays REQUESTED (human approval required);
nothing here fires a side effect. Lineage (decision_id, reasoning signals,
watermark) rides along so the resulting ExecutionRequest is fully traceable.

THE RULE THIS FILE NOW ENFORCES
-------------------------------
A proactive proposal may only exist if it is either (a) executable as authored,
or (b) explicitly marked as needing a draft and structurally unable to execute
until it has one.

It used to violate that in two ways, both of which shipped to users:

  * `risk` / `opportunity` with a known contact  ->  send_email with `body: ""`.
    GmailProvider only checked `to` and `subject`, so approving that card sent a
    real, blank email to a real person, with the decision title ("Commitment
    slipping: I'll get you the deck by Friday") as the subject.

  * `delegation` (and any contactless risk)      ->  create_task, whose provider
    hashed the title into a synthetic id and reported EXECUTED. There is no task
    system. Users were told work had been filed that existed nowhere.

Both are gone. Until real drafting lands (Phase 2), the only thing this bridge
emits is a REPLY_TO_EMAIL candidate carrying evidence and NO body — see
`navis.execution.providers.reply.ReplyToEmailProvider`, which refuses to execute
without one.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from navis.execution.models import ActionType
from navis.execution.obligation import reply_obligation_key
from navis.platform.monitoring import get_logger

logger = get_logger("execution.proposer")


@dataclass(frozen=True, slots=True)
class ProposedAction:
    org_id: uuid.UUID
    decision_id: uuid.UUID
    action_type: str
    payload: dict[str, Any]
    reasoning_signal_ids: list[str]
    reasoning_version: int | None
    state_watermark: str | None


def _waiting_reply_payload(decision: dict[str, Any]) -> dict[str, Any] | None:
    """Turn a "someone is waiting on your reply" decision into a reply candidate.

    Carries the full evidence the Action Center card needs to justify itself —
    who is waiting, since when, on which thread — and deliberately carries NO
    body. `requires_draft` is the machine-readable form of "this cannot be sent
    yet"; the approve endpoint and the provider both refuse it.
    """
    evidence = decision.get("evidence", {}) or {}
    person = evidence.get("person_email")
    thread_id = evidence.get("thread_id")
    if not person or not thread_id:
        # Without a recipient and a thread there is nothing to reply TO. A
        # candidate that can't identify its own target is noise.
        return None
    return {
        "to": [person],
        "person_email": person,
        "person_name": evidence.get("person_name"),
        "thread_id": thread_id,
        "reply_thread_id": thread_id,
        "in_reply_to_message_id": evidence.get("rfc_message_id"),
        "subject": evidence.get("subject") or decision.get("title", "")[:200],
        # No body, on purpose. Phase 2's drafting stage fills this in; until
        # then the card renders "needs a draft" and cannot be approved.
        "body": "",
        "requires_draft": True,
        "candidate_kind": "waiting_reply",
        "waiting_days": evidence.get("waiting_days"),
        "last_inbound_at": evidence.get("last_inbound_at"),
        "decision_id": decision.get("decision_id"),
    }


def _follow_up_payload(decision: dict[str, Any]) -> dict[str, Any] | None:
    """"Follow up with Chrissy — product deck was due Friday."

    Everything here is quoted from the source message: the promise, the deadline
    the promiser stated, and how long ago it passed. Nothing is computed into
    prose, because prose is Phase 2B's job.
    """
    e = decision.get("evidence", {}) or {}
    person, thread_id = e.get("person_email"), e.get("thread_id")
    if not person or not thread_id:
        return None
    return {
        "to": [person],
        "person_email": person,
        "person_name": e.get("person_name"),
        "thread_id": thread_id,
        "reply_thread_id": thread_id,
        "subject": e.get("subject") or decision.get("title", "")[:200],
        "body": "",
        "requires_draft": True,
        "candidate_kind": "commitment_follow_up",
        # The specifics the card renders, and the audit row preserves.
        "commitment": e.get("commitment"),
        "due_at": e.get("due_at"),
        "due_text": e.get("due_text"),
        "overdue_days": e.get("overdue_days"),
        "promised_at": e.get("promised_at"),
        "decision_id": decision.get("decision_id"),
    }


def _decision_owed_payload(decision: dict[str, Any]) -> dict[str, Any] | None:
    """"Review Product Release decision — awaiting your response."."""
    e = decision.get("evidence", {}) or {}
    person, thread_id = e.get("person_email"), e.get("thread_id")
    if not person or not thread_id:
        return None
    return {
        "to": [person],
        "person_email": person,
        "person_name": e.get("person_name"),
        "thread_id": thread_id,
        "reply_thread_id": thread_id,
        "in_reply_to_message_id": e.get("rfc_message_id"),
        "subject": e.get("subject") or decision.get("title", "")[:200],
        "body": "",
        "requires_draft": True,
        "candidate_kind": "decision_owed",
        "decision_question": e.get("decision"),
        "age_days": e.get("age_days"),
        "asked_at": e.get("asked_at"),
        "decision_id": decision.get("decision_id"),
    }


def _map(decision: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Deterministic category -> action, or (None, {}) for "nothing to propose".

    Returning None is the DEFAULT, not the exception. A decision only becomes a
    proposal when we can author something honest; "we noticed X" is a finding
    for the brief, not an action for the queue.
    """
    category = decision.get("category")

    # Every draftable category produces the SAME action type — a reply into the
    # thread the obligation lives in — and differs only in the evidence it
    # carries. One action type, one provider, one approval path.
    builders = {
        "waiting_reply": _waiting_reply_payload,
        "follow_up": _follow_up_payload,
        "decision_owed": _decision_owed_payload,
    }
    builder = builders.get(str(category))
    if builder is not None:
        payload = builder(decision)
        if payload is None:
            return None, {}
        return ActionType.REPLY_TO_EMAIL.value, payload

    # risk / opportunity / delegation / priority / context:
    #
    # No proposal until drafting exists. These previously produced blank emails
    # and fake tasks; a decision the founder can read on the brief is strictly
    # better than a card that lies about what approving it will do.
    return None, {}


@dataclass(frozen=True, slots=True)
class DecisionAction:
    """WHAT, if anything, can be executed about one brief decision.

    `action_type` is an `ActionType` value and `obligation_key` is the stable
    identity of the thing it would act on — both exactly as the proposer would
    write them, because they come from the same `_map`. `requires_draft` says
    the candidate is a placeholder that cannot be sent as authored.

    The remaining fields are READ OFF THE REGISTERED `ActionSpec`, never decided
    here. `execution.actions` is the single declaration of what an action is —
    which provider performs it, which entitlement it consumes, how much damage it
    can do, whether a human must approve, and whether performing it refreshes
    Navis's own read model. Every one of those was already required to execute
    the action and none of it was reachable from the decision that proposes it,
    so a surface could name an action and still not say who would perform it or
    whether pressing the button sends something that cannot be unsent.

    Copied rather than re-derived: `require_spec` fails closed on an unknown
    action type, so an action can never be described here under a policy the
    worker would not apply.
    """

    action_type: str
    obligation_key: str
    requires_draft: bool
    # `Provider.name` — WHO would perform it. Provider-neutral by construction:
    # a future Jira action registers its own spec and answers this identically,
    # with no branch anywhere in Decision or Today.
    provider: str
    # The connector entitlement it consumes, checked IN ADDITION to the blanket
    # EXECUTION capability. A surface can tell "not entitled" from "not possible".
    capability: str
    # `ActionRisk` — "moves an object" vs "delivers a message that cannot be
    # unsent". The one thing a person needs before pressing approve.
    risk: str
    # True while `ApprovalPolicy.REQUIRED`. DECLARATIVE: the hard boundary is the
    # status machine (REQUESTED -> APPROVED is the single edge into APPROVED and
    # is reachable only from the approve endpoint), so this reports the policy
    # and never grants an exemption from it.
    requires_approval: bool
    # Whether performing this action pulls its own result back into
    # `signal_events` — the Execution -> Observation -> State edge, closed today
    # for the calendar verbs by `_calendar_reconcile` and not for the mail ones,
    # where the reply re-enters through the next Gmail sync instead. Published
    # because "when will Today stop showing this?" has two different answers.
    updates_read_model: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "obligation_key": self.obligation_key,
            "requires_draft": self.requires_draft,
            "provider": self.provider,
            "capability": self.capability,
            "risk": self.risk,
            "requires_approval": self.requires_approval,
            "updates_read_model": self.updates_read_model,
        }


def decision_action(decision: dict[str, Any]) -> DecisionAction | None:
    """The action a brief decision would produce, or None if it produces none.

    THE point of this function is that it runs the decision through the SAME
    `_map` the proposer uses, so "what can be done about this decision?" has
    exactly one answer in the codebase. The brief-enrichment read path and the
    execution write path cannot drift into disagreeing — if they did, a card
    would offer to send a reply the proposer refuses to create, or report "no
    reply sent" for an obligation that had been sent.

    None is the ORDINARY answer, and it is a real one: `risk`, `opportunity`
    and `delegation` decisions are findings to read, not replies to send, and a
    draftable decision whose thread carries no `rfc_message_id` has no stable
    obligation identity (see `propose_from_decisions`). Both cases must reach a
    surface as "there is nothing to execute here" rather than as an Execute
    button that would fail or duplicate.
    """
    action_type, payload = _map(decision)
    if action_type is None:
        return None
    key = reply_obligation_key(action_type, payload)
    if key is None:
        return None
    # Lazy: `execution.actions` imports the provider registry, which imports this
    # module's siblings. Local, like every other cross-module import here.
    from navis.execution.actions import ApprovalPolicy, require_spec

    spec = require_spec(action_type)
    return DecisionAction(
        action_type=action_type,
        obligation_key=key,
        requires_draft=bool(payload.get("requires_draft")),
        provider=spec.provider,
        capability=spec.capability.value,
        risk=spec.risk.value,
        requires_approval=spec.approval is ApprovalPolicy.REQUIRED,
        updates_read_model=spec.reconciler is not None,
    )


def decision_obligation_key(decision: dict[str, Any]) -> str | None:
    """The stable obligation a brief decision represents, or None if it has none.

    A projection of `decision_action` — one derivation, so the enrichment read
    path and the execution write path cannot drift into disagreeing about
    identity.
    """
    action = decision_action(decision)
    return action.obligation_key if action else None


def propose_from_decisions(decisions: list[dict], org_id: uuid.UUID) -> list[ProposedAction]:
    out: list[ProposedAction] = []
    skipped: dict[str, int] = {}
    unidentifiable: dict[str, int] = {}
    for d in decisions:
        action_type, payload = _map(d)
        if action_type is None:
            category = str(d.get("category") or "unknown")
            skipped[category] = skipped.get(category, 0) + 1
            continue
        # A reply must be able to name the message it answers. Without that
        # there is no stable identity for the obligation, and the only
        # alternatives are observation-scoped values (decision_id, watermark,
        # waiting_days) that mint a fresh "new action" for the same email every
        # time a brief is rebuilt — the duplicate-send defect.
        #
        # `commitment_follow_up` is the real case: CommitmentState carries no
        # rfc_message_id (only ThreadState and DecisionState do), so a follow-up
        # cannot say which message it is replying to. Not proposing it is the
        # honest outcome — the decision still appears on the brief as a finding
        # to read. Inventing an identity to keep the card would trade a missing
        # nudge for a possible duplicate email, which is the worse failure.
        if reply_obligation_key(action_type, payload) is None:
            unidentifiable[str(d.get("category") or "unknown")] = (
                unidentifiable.get(str(d.get("category") or "unknown"), 0) + 1
            )
            continue
        out.append(
            ProposedAction(
                org_id=org_id,
                decision_id=uuid.UUID(d["decision_id"]),
                action_type=action_type,
                payload=payload,
                reasoning_signal_ids=d.get("reasoning_signal_ids", []),
                reasoning_version=d.get("reasoning_version"),
                state_watermark=d.get("state_watermark"),
            )
        )
    if skipped:
        # Visible, not silent: an empty Action Center should be explainable from
        # the logs rather than looking like a broken pipeline.
        logger.info(
            "proposer org=%s proposed=%d skipped=%s (no draftable action for "
            "those categories yet)", org_id, len(out), skipped,
        )
    if unidentifiable:
        # Louder than `skipped`: this is a candidate we WOULD have proposed and
        # deliberately did not, so an empty queue stays explainable.
        logger.warning(
            "proposer org=%s withheld=%s (no rfc_message_id, so the reply "
            "obligation has no stable identity and cannot be de-duplicated)",
            org_id, unidentifiable,
        )
    return out
