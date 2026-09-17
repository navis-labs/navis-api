"""Stable identity for a reply obligation.

THE BUG THIS EXISTS TO FIX
--------------------------
`idempotency_key` hashes (org, decision_id, action_type, payload). For a reply
that turned out to encode *when Navis looked*, not *what is owed*:

  * `decision_id` = uuid5(org | reasoning_version | state_watermark | signal_id),
    and `state_watermark` moves on EVERY new memory record — so one unrelated
    inbound email anywhere in the org mints a new decision_id for an unchanged
    obligation.
  * the payload carries `waiting_days`, which drifts by the day.

Measured on real data: ONE unanswered Gmail message produced three decision_ids
and three distinct idempotency keys across three briefs — three rows that would
each insert cleanly and each send a separate email into the same thread:

    brief 5a537645  waiting_days=2.3  decision=842732eb  key=63418e25
    brief 62059d13  waiting_days=2.3  decision=53878738  key=4d348a78
    brief f12eab92  waiting_days=5.9  decision=cbb65303  key=e800950b

Note the first two: identical evidence, identical waiting_days, differing ONLY
by state_watermark. The duplicate does not even need a day to pass.

THE IDENTITY
------------
A reply obligation is "reply into THIS thread about THIS specific thing". Both
components are already on the payload and are invariant across briefs:

    thread_id                 where the reply goes
    in_reply_to_message_id    which message is unanswered (RFC 2822 Message-ID)

`candidate_kind` joins them because "reply because they're waiting" and "reply
to chase a commitment" are genuinely different obligations that can co-exist on
one thread.

`commitment_follow_up` is the one kind with no Message-ID — it hangs off
CommitmentState, which has no rfc_message_id — so its subject is the promise
being chased, named by a content hash of the promise text. That is a
deterministic stable fallback, NOT an invented one: it is the same
normalisation Memory already uses to keep one promise from forking into two
facts. See `_commitment_subject`.

Nothing observation-scoped may appear here — not decision_id, state_watermark,
waiting_days, brief id, or any timestamp of when we looked. That exclusion IS
the fix; see test_reply_obligation_identity.py, which pins each one.

SCOPE
-----
reply_to_email only. Calendar and every other action type keep the existing
payload-hash idempotency untouched: Google's `events.insert` takes a
client-generated id and 409s on replay, so scheduling already has an
exactly-once guarantee that does not need this.
"""
from __future__ import annotations

import hashlib
from typing import Any

# The one action type this applies to. Kept as a literal rather than importing
# ActionType so this module stays free of model imports and trivially testable.
REPLY_ACTION_TYPE = "reply_to_email"


def _clean_message_id(raw: Any) -> str | None:
    """Normalise an RFC 2822 Message-ID to its addr-spec.

    The angle brackets are delimiters, not part of the identifier, and the same
    id reaches us both ways depending on the header path it travelled. Folding
    them away means one message cannot present as two obligations.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    text = text.strip()
    return text or None


def _clean(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    return text or None


def _commitment_subject(payload: dict[str, Any]) -> str | None:
    """Stable identity for "chase this promise", where no Message-ID exists.

    `commitment_follow_up` is the one draftable kind with no message-level id:
    it hangs off CommitmentState, which carries no rfc_message_id (only
    ThreadState and DecisionState do — state/v2/models.py). So the obligation is
    not "answer message M" but "chase promise P", and P needs naming.

    The promise text is the right name for it, and Memory already identifies
    commitments exactly this way — `memory/extractors/email.py::_fact_key`
    hashes the same normalisation to keep one promise from forking into two
    facts. Reusing that rule means the execution layer and the memory layer
    agree on when two promises are the same promise.

    Critically this is CONTENT-derived, not observation-derived: it does not
    move with the state watermark, the brief, or `overdue_days`. A different
    promise is a different obligation, which is the behaviour we want.
    """
    text = _clean(payload.get("commitment"))
    if not text:
        return None
    normalized = " ".join(text.strip().lower().split())
    return "commitment:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def reply_obligation_key(action_type: str, payload: dict[str, Any] | None) -> str | None:
    """The stable natural key for a reply obligation, or None if it has none.

    None means "this candidate cannot be identified", and the caller MUST refuse
    to create a sendable execution rather than fall back to something unstable.
    An unstable identity is what caused the duplicate in the first place — so
    the only accepted subjects are the two below, both invariant across briefs:

        the RFC Message-ID being answered   (waiting_reply, decision_owed)
        the promise being chased            (commitment_follow_up)

    Nothing observation-scoped is eligible: not decision_id, state_watermark,
    waiting_days, overdue_days, brief id, or any timestamp of when we looked.
    """
    if action_type != REPLY_ACTION_TYPE:
        return None
    payload = payload or {}

    kind = _clean(payload.get("candidate_kind"))
    thread_id = _clean(payload.get("thread_id")) or _clean(payload.get("reply_thread_id"))
    if not kind or not thread_id:
        return None

    subject = _clean_message_id(payload.get("in_reply_to_message_id"))
    if subject is None and kind == "commitment_follow_up":
        subject = _commitment_subject(payload)
    if subject is None:
        return None

    # Readable on purpose: this lands in a column an operator can eyeball and
    # group by, not just an opaque digest.
    return f"reply|{kind}|{thread_id}|{subject}"
