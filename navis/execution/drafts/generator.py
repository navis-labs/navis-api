"""Grounded email drafts for the three proactive candidate kinds.

DELIBERATELY DETERMINISTIC — no LLM. That is an engineering decision, not an
omission, and it follows from item 11 being absolute:

  A template can be PROVEN not to fabricate. Every variable slot is filled from
  a fact that already exists in Memory or in the Gmail thread, and the tests
  below assert exactly which slots exist. A language model cannot be proven not
  to fabricate — it can only be observed not to, on the examples someone
  thought to try.

The failure this protects against is specific and severe. These emails go to
investors, customers and colleagues under the founder's own name. A model that
invents "as we discussed on Tuesday" or answers "yes, Monday works" on the
founder's behalf does not produce a bad draft — it produces a lie the founder
sends. Phrasing quality is worth far less than that.

Where an LLM WOULD help is rephrasing a template-grounded draft into warmer
prose. That is a strictly-additive future step: generate deterministically,
optionally rephrase, and diff the two for invented entities before showing the
rephrased version. It is not this phase.

THE RULE EVERY TEMPLATE FOLLOWS
--------------------------------
Navis MAY, in the founder's voice:
  * acknowledge receipt;
  * reference what the OTHER person said, quoted or closely paraphrased from
    their own message;
  * say the founder will come back to them.

Navis may NOT:
  * answer a question the founder has not answered;
  * state a decision, date, commitment, opinion or fact not present in the
    thread;
  * APOLOGISE on the founder's behalf — an apology is a concession, and in a
    negotiation or a dispute it has real cost. Whether to make one is the
    founder's call, not a template's;
  * COMMIT TO A TIMEFRAME. "shortly", "by end of day" and "tomorrow" are
    promises the founder never made. Navis's own commitment detector would
    read an inbound "I'll get back to you shortly" as a binding promise —
    manufacturing one for the user is the same error pointed the other way;
  * ASSERT THE FOUNDER'S INTERNAL STATE. "I haven't decided yet" looks harmless
    and is unverifiable: the founder may have decided in a meeting, on a call,
    or in Slack. Navis only knows no decision is recorded IN THIS THREAD, which
    is a different claim entirely;
  * say anything about attachments, meetings or people not in the evidence.

So "Yes, we're still targeting Monday" is never generated, even though it is the
nicer draft — Navis has no way to know it is true. The single promise that IS
kept ("I'll come back to you") is the irreducible content of an acknowledgement:
an email that acknowledges without it says nothing at all. It carries no
timeframe, and the founder sees it before it sends.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from navis.execution.drafts.thread_context import ThreadContext, ThreadMessage

GENERATOR_VERSION = "template-1"

# Candidate kinds this module knows how to draft. Mirrors the `candidate_kind`
# the proposer writes (execution/proposer.py).
WAITING_REPLY = "waiting_reply"
COMMITMENT_FOLLOW_UP = "commitment_follow_up"
DECISION_OWED = "decision_owed"

DRAFTABLE_KINDS = frozenset({WAITING_REPLY, COMMITMENT_FOLLOW_UP, DECISION_OWED})


@dataclass(frozen=True, slots=True)
class GeneratedDraft:
    subject: str
    body: str
    recipients: tuple[str, ...]
    thread_id: str | None
    in_reply_to_message_id: str | None
    reason: str
    evidence: dict
    generator: str = GENERATOR_VERSION


class InsufficientContext(Exception):
    """Not enough grounded material to write anything honest.

    Raised rather than returning a vague draft: item 11 says that when context
    is missing, Navis must ask rather than produce a confident email. The
    service surfaces this to the user instead of storing a draft.
    """


# ── Name handling ──────────────────────────────────────────────────────

# Only a plausible personal first name is used in a greeting. "Hi Support Team,"
# and "Hi noreply," read as machine-written, and a mangled name is worse than
# none — so anything that does not look like a name degrades to a bare "Hi,".
_NAME_TOKEN_RE = re.compile(r"^[A-Z][a-zA-Z'\-]{1,30}$")
_NON_NAME_TOKENS = {
    "The", "Team", "Support", "Sales", "Info", "Admin", "Notifications",
    "No", "Noreply", "Do", "Mailer", "Customer", "Service", "Help",
}


def _greeting_name(msg: ThreadMessage | None, fallback: str | None = None) -> str | None:
    """A first name safe to greet with, or None.

    Never derived from the email local-part: "j.smith92@..." would produce
    "Hi J.smith92," which reads worse than no name at all.
    """
    raw = (msg.from_name if msg else None) or fallback
    if not raw:
        return None
    first = str(raw).strip().split()[0] if str(raw).strip() else ""
    if not first or first in _NON_NAME_TOKENS or not _NAME_TOKEN_RE.match(first):
        return None
    return first


def _greeting(name: str | None) -> str:
    return f"Hi {name}," if name else "Hi,"


def _sign_off(sender_name: str | None) -> str:
    """No invented signature. When the sender's name is unknown the draft ends
    with a bare sign-off and the founder's own client/signature completes it."""
    return f"Best,\n{sender_name}" if sender_name else "Best,"


def _thanks_sign_off(sender_name: str | None) -> str:
    """"Thanks" reads better than "Best" when asking someone for something."""
    return f"Thanks,\n{sender_name}" if sender_name else "Thanks,"


# ── Commitment phrasing ────────────────────────────────────────────────

# Strips the promise scaffolding so "I'll send you the product deck by Friday"
# becomes "the product deck" — usable inside "following up on the product deck".
# Every piece removed is a fixed phrase; nothing is added.
_PROMISE_PREFIX_RE = re.compile(
    r"^\s*(?:hi|hey|hello)?[^a-zA-Z]*"
    r"(?:i|we)\s*(?:'|’)?(?:ll|ll\b|will|am going to|are going to)\s+"
    r"(?:get\s+you\s+|send\s+(?:you\s+|over\s+|across\s+)?|share\s+(?:with\s+you\s+)?"
    r"|deliver\s+|provide\s+|forward\s+|put\s+together\s+)",
    re.IGNORECASE,
)
# The trailing `[.!?]?` matters: without it " by Friday." does not match (the
# final period is excluded from the character class but still has to be
# consumed), the deadline survives the strip, and the body reads
# "…the product deck by Friday by Friday."
_DUE_SUFFIX_RE = re.compile(
    r"\s*(?:\bby\b|\bbefore\b|\bon\b|\bthis\b|\bnext\b|\bin\b)\s+[^.!?]*[.!?]?\s*$",
    re.IGNORECASE,
)
_TRAILING_PUNCT_RE = re.compile(r"[\s.,;:!?]+$")


def describe_commitment(text: str | None) -> tuple[str, bool]:
    """(phrase, is_paraphrase) describing WHAT was promised.

    Returns a short noun phrase when the scaffolding strips cleanly ("the
    product deck"), and otherwise falls back to quoting the sentence verbatim.
    The verbatim fallback is why this cannot fabricate: if the transform does
    not produce something clean, the person's own words are used instead of a
    guess at what they meant.
    """
    raw = (text or "").strip()
    if not raw:
        return "", False
    # The PREFIX match is what turns a sentence into a noun phrase, so it is
    # the condition for paraphrasing at all. Judging by "did the string get
    # shorter" instead let "I'll get back to you tomorrow." through as a
    # paraphrase — it had only lost its full stop — and produced the
    # ungrammatical "Just following up on I'll get back to you tomorrow."
    prefixed = _PROMISE_PREFIX_RE.sub("", raw)
    if prefixed == raw:
        return _TRAILING_PUNCT_RE.sub("", raw), False
    stripped = _TRAILING_PUNCT_RE.sub("", _DUE_SUFFIX_RE.sub("", prefixed)).strip()
    words = stripped.split()
    # Long enough to mean something, short enough to sit mid-sentence.
    if 1 <= len(words) <= 12:
        return stripped, True
    return _TRAILING_PUNCT_RE.sub("", raw), False


# ── Generation ─────────────────────────────────────────────────────────


def generate(
    *,
    candidate_kind: str,
    payload: dict,
    thread: ThreadContext | None,
    sender_name: str | None = None,
) -> GeneratedDraft:
    """Draft a reply for one proactive candidate.

    `thread` is REQUIRED. Refusing to draft without it is deliberate: a draft
    written from the payload alone cannot know the real subject, the real
    recipient, or which message it is answering, and would be exactly the
    "draft from the candidate title" that item 4 forbids.
    """
    if candidate_kind not in DRAFTABLE_KINDS:
        raise InsufficientContext(f"no draft template for candidate kind {candidate_kind!r}")
    if thread is None or not thread.messages:
        raise InsufficientContext(
            "the source Gmail thread is not in Navis's memory yet, so there is "
            "nothing to ground a reply in"
        )

    recipient = (payload.get("person_email") or "").strip().lower()
    if not recipient:
        recipient = (thread.latest_inbound.from_email or "") if thread.latest_inbound else ""
    if not recipient:
        raise InsufficientContext("no recipient could be resolved from the thread")

    their_message = thread.latest_inbound_from(recipient) or thread.latest_inbound
    name = _greeting_name(their_message, payload.get("person_name"))
    subject = thread.reply_subject()

    if candidate_kind == WAITING_REPLY:
        body, reason = _waiting_reply_body(name, sender_name, payload, their_message)
    elif candidate_kind == COMMITMENT_FOLLOW_UP:
        body, reason = _commitment_body(name, sender_name, payload)
    else:
        body, reason = _decision_body(name, sender_name, payload)

    return GeneratedDraft(
        subject=subject,
        body=body,
        recipients=(recipient,),
        thread_id=thread.thread_id,
        # Threading anchor: the RFC Message-ID of the message being answered.
        in_reply_to_message_id=(
            their_message.rfc_message_id if their_message else None
        )
        or payload.get("in_reply_to_message_id"),
        reason=reason,
        evidence=_evidence(candidate_kind, payload, thread, their_message),
    )


def _waiting_reply_body(
    name: str | None, sender_name: str | None, payload: dict, their: ThreadMessage | None
) -> tuple[str, str]:
    """A neutral acknowledgement. NEVER an answer.

    The phase spec's nicer example ("Yes, we're still targeting Monday") is
    exactly what must not be generated: Navis has no way to know the founder
    still targets Monday, and putting that in their mouth to a customer is the
    worst outcome this feature can produce. So every waiting-reply draft
    acknowledges and promises a real answer shortly — which is true, because the
    founder is about to write one.
    """
    days = payload.get("waiting_days")
    # No apology (a concession is the founder's call), no timeframe, and no
    # explanation for a delay Navis cannot know the reason for. Just an
    # acknowledgement the founder can build on.
    body = (
        f"{_greeting(name)}\n\n"
        "Thanks for following up on this — I'll come back to you on it.\n\n"
        f"{_sign_off(sender_name)}"
    )
    waited = f" {days:g} days ago" if isinstance(days, (int, float)) else ""
    subject = (their.subject if their else None) or "this thread"
    reason = (
        f"They wrote about “{subject}”{waited} and there is no reply from you "
        "in the thread."
    )
    return body, reason


def _commitment_body(
    name: str | None, sender_name: str | None, payload: dict
) -> tuple[str, str]:
    """Chase a promise, quoting THEIR words and THEIR deadline.

    No urgency is manufactured and no excuse is offered on their behalf — the
    draft states what they said and asks where it stands.
    """
    phrase, paraphrased = describe_commitment(payload.get("commitment"))
    due_text = (payload.get("due_text") or "").strip()

    if not phrase:
        # Nothing quotable. Rather than invent a subject for the chase, ask.
        raise InsufficientContext(
            "the original commitment text is missing, so there is nothing to "
            "follow up on without inventing it"
        )

    if paraphrased:
        # "…on the product deck — you mentioned you'd send it by Friday" rather
        # than "…on the product deck by Friday", which parses just as readily as
        # "follow up BY Friday" and made the ask ambiguous.
        opener = f"Just following up on {phrase}"
    else:
        # Verbatim fallback — their sentence, in quotes, unaltered.
        opener = f"Just following up on this — you mentioned: “{phrase}”"

    # The deadline is appended only to the PARAPHRASED form. In the verbatim
    # form their own sentence already contains it, and repeating it would read
    # as "…you mentioned: 'I'll send it by Friday' by Friday."
    # Attributed to THEM so the deadline reads as their own words rather
    # than an assertion Navis is making. Paraphrased form only: the verbatim
    # quote already contains it. It also removes a real ambiguity —
    # "following up on the deck by Friday" parses just as readily as
    # "follow up BY Friday".
    when = (
        f" — you mentioned you'd send it by {due_text}"
        if due_text and paraphrased
        else ""
    )
    body = (
        f"{_greeting(name)}\n\n"
        # "where this stands" rather than "had a chance to share it": not
        # every commitment is a thing to SEND. "I'll call the vendor by
        # Friday" takes the verbatim path, where "share it" was just wrong.
        f"{opener}{when}. Wanted to check where this stands.\n\n"
        f"{_thanks_sign_off(sender_name)}"
    )
    overdue = payload.get("overdue_days")
    late = f", {overdue:g} days ago" if isinstance(overdue, (int, float)) else ""
    reason = (
        f"They said they would send this"
        + (f" by {due_text}{late}" if due_text else "")
        + ". No delivery was found in the thread after the deadline."
    )
    return body, reason


def _decision_body(
    name: str | None, sender_name: str | None, payload: dict
) -> tuple[str, str]:
    """Acknowledge a decision request WITHOUT making the decision.

    This is the sharpest edge in the whole feature. "Should we launch Monday or
    Tuesday?" has two plausible answers and Navis knows neither. Picking one —
    even the one that looks likelier — manufactures the founder's decision and
    communicates it to their team as fact.
    """
    # Deliberately does NOT say "I haven't decided yet". Navis knows only
    # that no decision is recorded IN THIS THREAD; the founder may well have
    # decided in a meeting or on a call. Putting a false statement about
    # their own position in front of their team is worse than saying less.
    body = (
        f"{_greeting(name)}\n\n"
        "Thanks for following up — I'll come back to you with an answer "
        "on this.\n\n"
        f"{_sign_off(sender_name)}"
    )
    question = (payload.get("decision_question") or "").strip()
    age = payload.get("age_days")
    waited = f" {age:g} days ago" if isinstance(age, (int, float)) else ""
    reason = (
        (f"They asked “{question}”{waited}" if question else f"They asked for a decision{waited}")
        + " and no decision has been recorded in the thread since."
    )
    return body, reason


def _evidence(
    candidate_kind: str,
    payload: dict,
    thread: ThreadContext,
    their: ThreadMessage | None,
) -> dict:
    """Concise FACTS, never reasoning.

    Item 10: the user must be able to answer "why is Navis asking me to send
    this?" from linkable evidence. Item 11's corollary: no chain-of-thought.
    """
    out: dict = {
        "candidate_kind": candidate_kind,
        "thread_id": thread.thread_id,
        "thread_subject": thread.subject,
        "message_count": len(thread.messages),
    }
    if their is not None:
        out["source_message_id"] = their.message_id
        out["source_message_at"] = their.sent_at.isoformat() if their.sent_at else None
        out["source_from"] = their.from_email
        out["source_excerpt"] = (their.snippet or "")[:300] or None
    if candidate_kind == COMMITMENT_FOLLOW_UP:
        out["commitment"] = payload.get("commitment")
        out["due_at"] = payload.get("due_at")
        out["due_text"] = payload.get("due_text")
        out["overdue_days"] = payload.get("overdue_days")
    elif candidate_kind == DECISION_OWED:
        out["decision_question"] = payload.get("decision_question")
        out["age_days"] = payload.get("age_days")
        out["asked_at"] = payload.get("asked_at")
    else:
        out["waiting_days"] = payload.get("waiting_days")
        out["last_inbound_at"] = payload.get("last_inbound_at")
    return out
