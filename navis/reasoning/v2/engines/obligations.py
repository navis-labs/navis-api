"""Obligations: promises made TO the founder, and decisions owed BY them.

Sibling of `engines/waiting.py` (a reply the founder owes) and of the decay
engines in `engines/risk.py`. All three obligation engines share one property
that separates them from decay: they measure an unmet DUTY, which does not
become less true with time. That is why they use `MAX_OBLIGATION_AGE_DAYS`
rather than the 30-day decay ceiling.

DETERMINISTIC THROUGHOUT. No model decides whether a promise was kept or a
decision was made — both are read off facts that Memory already records:

  COMMITMENT_FOLLOW_UP
    A COMMITMENT fact carries `due_at` when (and only when) the sentence stated
    a deadline. Past that deadline with no delivery evidence in the thread, the
    promise is overdue.

  DECISION_OWED
    A DECISION fact with `kind == "request"` is somebody asking the founder to
    decide. It stops being owed when the founder replies in that thread, or when
    anyone records an actual decision there.

Both suppressions ride on the STICKY thread markers written by
`extractors/email.py` (`fulfilment_at` / `resolution_at`), which persist under
State's latest-non-null fold. So resolution needs no teardown step that could be
forgotten: the later message simply exists, and the engine stops firing.
"""
from __future__ import annotations

from datetime import datetime

from navis.reasoning.v2.base import ReasoningContext, ReasoningEngine, days_between
from navis.reasoning.v2.contract import SignalDraft, SignalType
from navis.reasoning.v2.engines.risk import MAX_OBLIGATION_AGE_DAYS, _clip, _plain
from navis.state.v2.models import OrganizationalState, ThreadState

# Grace after a deadline before Navis says anything. Somebody who promised
# "by Friday" and delivers Friday evening has not missed anything, and a card
# that appears at 00:01 on Saturday reads as pedantic rather than helpful.
FOLLOW_UP_GRACE_DAYS = 0.5

# How long the founder gets before an unanswered decision request is worth
# raising. Shorter than the reply window in `waiting.py` (2 days) because a
# blocked decision is blocking someone ELSE's work.
DECISION_OWED_MIN_DAYS = 1.0


def _parse(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


def _threads_by_id(state: OrganizationalState) -> dict[str, ThreadState]:
    """Thread lookup keyed by the Gmail thread id the facts carry."""
    out: dict[str, ThreadState] = {}
    for t in state.threads():
        key = t.thread_id
        if key:
            out[key] = t
    return out


def _delivered_after(thread: ThreadState | None, speaker: str | None, since: datetime | None) -> bool:
    """Did `speaker` claim delivery in this thread AFTER `since`?

    The `after` comparison is what stops an old delivery from excusing a NEW
    promise: "sent it!" last week says nothing about the deck promised today.
    """
    if thread is None or thread.fulfilment_at is None:
        return False
    when = _parse(thread.fulfilment_at)
    if when is None:
        return False
    if since is not None and when <= since:
        return False
    # A delivery claim from a DIFFERENT person does not discharge this person's
    # promise. Unknown speaker => accept the evidence rather than nag.
    if speaker and thread.fulfilment_from and thread.fulfilment_from != speaker:
        return False
    return True


class CommitmentFollowUpEngine(ReasoningEngine):
    """Somebody promised the founder something by a date, and that date passed."""

    name = "commitment_follow_up"
    version = "1.0.0"

    def analyze(
        self, state: OrganizationalState, ctx: ReasoningContext
    ) -> list[SignalDraft]:
        if ctx.as_of is None:
            return []
        threads = _threads_by_id(state)
        out: list[SignalDraft] = []

        for c in state.commitments():
            if c.status != "open":
                continue
            # Only DATED promises. An undated one cannot be called overdue
            # without inventing a deadline, so it stays with the existing
            # COMMITMENT_SLIPPING policy (open N days with no newer fact).
            due_at = _parse(c.due_at)
            if due_at is None:
                continue

            overdue_days = (ctx.as_of - due_at).total_seconds() / 86400.0
            if overdue_days < FOLLOW_UP_GRACE_DAYS:
                continue  # not due yet, or due today

            age = days_between(ctx.as_of, c.promised_at or c.last_seen)
            if age is not None and age > MAX_OBLIGATION_AGE_DAYS:
                continue

            speaker = (c.speaker_email or "").strip().lower() or None
            if not speaker:
                continue  # nobody to follow up with

            # Already delivered? The thread carries a sticky delivery claim.
            promised_at = _parse(c.promised_at) or _parse(c.last_seen)
            if _delivered_after(threads.get(c.thread_id or ""), speaker, promised_at):
                continue

            text = _plain(c.text)
            if not text:
                continue

            display = _plain(c.speaker_name) or speaker
            due_phrase = c.due_text or due_at.date().isoformat()
            out.append(
                SignalDraft(
                    signal_type=SignalType.COMMITMENT_FOLLOW_UP,
                    entity_type="commitment",
                    entity_id=c.entity_id,
                    title=f"Follow up with {display} — {_clip(text, 60)}",
                    why=(
                        f'They said "{_clip(text)}" and it was due {due_phrase} '
                        f"({overdue_days:.0f} days ago). No delivery in the thread since."
                    ),
                    # The promise and the deadline are both quoted from the
                    # message, so detection is observed rather than inferred.
                    confidence=0.85,
                    severity=min(0.9, 0.45 + overdue_days / 20),
                    dedup_key=c.entity_id,
                    evidence={
                        "person_email": speaker,
                        "person_name": _plain(c.speaker_name) or None,
                        "commitment": text,
                        "due_at": c.due_at,
                        "due_text": c.due_text,
                        "overdue_days": round(overdue_days, 1),
                        "promised_at": c.promised_at,
                        "thread_id": c.thread_id,
                        "subject": _plain(c.subject) or None,
                    },
                )
            )
        return out


class DecisionOwedEngine(ReasoningEngine):
    """Somebody asked the founder to decide, and no decision has landed."""

    name = "decision_owed"
    version = "1.0.0"

    def analyze(
        self, state: OrganizationalState, ctx: ReasoningContext
    ) -> list[SignalDraft]:
        if ctx.as_of is None:
            return []
        threads = _threads_by_id(state)
        out: list[SignalDraft] = []

        for d in state.decisions():
            if d.kind != "request":
                continue  # a recorded decision is not one that is owed
            asked_at = _parse(d.asked_at) or _parse(d.last_seen)
            if asked_at is None:
                continue

            age = (ctx.as_of - asked_at).total_seconds() / 86400.0
            if age < DECISION_OWED_MIN_DAYS or age > MAX_OBLIGATION_AGE_DAYS:
                continue

            thread = threads.get(d.thread_id or "")

            # RESOLUTION 1 — anyone recorded an actual decision in the thread
            # after it was asked. Covers "another participant resolves it".
            resolved_at = _parse(thread.resolution_at) if thread else None
            if resolved_at is not None and resolved_at > asked_at:
                continue

            # RESOLUTION 2 — the founder replied in the thread after the ask.
            #
            # ANY outbound reply counts, not only one that reads as decisive.
            # Judging decisiveness would need a model, and the asymmetry is
            # clear: continuing to tell a founder they owe an answer they have
            # already given is the trust-destroying failure, while a genuinely
            # unresolved question comes back the moment the other party asks
            # again — which creates a NEW request fact and a new candidate.
            if thread is not None and thread.direction == "outbound":
                replied_at = _parse(thread.last_seen)
                if replied_at is not None and replied_at > asked_at:
                    continue

            asker = (d.asked_by or "").strip().lower() or None
            if not asker:
                continue
            text = _plain(d.text)
            if not text:
                continue

            display = _plain(d.asked_by_name) or asker
            subject = _plain(d.subject)
            out.append(
                SignalDraft(
                    signal_type=SignalType.DECISION_OWED,
                    entity_type="decision",
                    entity_id=d.entity_id,
                    title=(
                        f"Decide: {_clip(subject or text, 60)}"
                        if subject
                        else f"Decide: {_clip(text, 60)}"
                    ),
                    why=(
                        f'{display} asked "{_clip(text)}" {age:.0f} days ago '
                        "and no decision has been recorded since."
                    ),
                    confidence=0.8,
                    severity=min(0.9, 0.5 + age / 20),
                    dedup_key=d.entity_id,
                    evidence={
                        "person_email": asker,
                        "person_name": _plain(d.asked_by_name) or None,
                        "decision": text,
                        "age_days": round(age, 1),
                        "asked_at": d.asked_at,
                        "thread_id": d.thread_id,
                        "subject": subject or None,
                        "rfc_message_id": d.rfc_message_id,
                    },
                )
            )
        return out
