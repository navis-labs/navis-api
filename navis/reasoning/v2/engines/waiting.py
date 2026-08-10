"""Who is waiting on the founder to reply.

The first genuinely PROACTIVE signal in Navis, and the one a Chief of Staff is
most expected to catch: "John emailed you on Monday and you never answered."

DETERMINISTIC BY CONSTRUCTION — no LLM decides whether the user replied.
"Did you reply?" is a fact, and getting it wrong is expensive in both
directions: nagging someone who already replied destroys trust in the product,
and staying silent on a waiting customer is the exact failure the feature
exists to prevent. So the judgement is reduced to a lookup:

  * The Gmail connector captures ALL messages, sent included, and Google stamps
    outbound mail with the `SENT` label.
  * `ThreadExtractor` records that as `direction` on every THREAD memory record.
  * State V2 folds thread records in ascending (event_timestamp, memory_id)
    order with latest-non-null-wins.

=> `ThreadState.direction` IS the direction of the newest message in the thread,
   and `last_seen` is when it arrived.

   direction == "inbound"  ->  they spoke last  ->  nobody has replied
   direction == "outbound" ->  the founder spoke last -> not waiting

No new table, no per-thread scan, no inference. A reply lands as an outbound
record whose timestamp is newer, the fold flips `direction`, and the signal stops
firing on the next pass — which is also why "user replies after 6 days" needs no
explicit teardown path.
"""
from __future__ import annotations

from navis.reasoning.v2.base import ReasoningContext, ReasoningEngine, days_between
from navis.reasoning.v2.contract import SignalDraft, SignalType
from navis.reasoning.v2.engines.risk import MAX_SURFACE_AGE_DAYS, _clip, _plain
from navis.state.v2.models import OrganizationalState

# How long silence has to last before it is worth surfacing. Under a day is not
# a lapse — it is a normal working rhythm, and a Chief of Staff that pings you
# about this morning's email is noise. Versioned with REASONING_VERSION.
WAITING_MIN_DAYS = 2.0

# Thread categories that are never "waiting on a reply". Gmail's own tabs plus
# our `is_automated` classifier already separate these; we reuse that rather
# than inventing a second, divergent notion of "is this a real person".
_NON_CONVERSATIONAL_CATEGORIES = frozenset(
    {"promotional", "social", "forum", "update", "automated", "spam"}
)

# Sender local-parts that never expect a human reply, for the residue that slips
# past category + is_automated (a personal-looking transactional sender).
_NO_REPLY_LOCALPARTS = (
    "no-reply", "noreply", "do-not-reply", "donotreply", "notifications",
    "notification", "mailer-daemon", "postmaster", "bounce", "bounces",
    "support@", "alerts", "alert", "news", "newsletter", "digest",
    "automated", "auto-confirm", "receipts", "billing",
)


def _is_no_reply_sender(email: str | None) -> bool:
    if not email:
        return True  # cannot address a reply to nobody
    local = email.split("@", 1)[0].lower()
    return any(marker.rstrip("@") in local for marker in _NO_REPLY_LOCALPARTS)


class WaitingReplyReasoningEngine(ReasoningEngine):
    """Emit WAITING_ON_USER_REPLY for threads the founder has left unanswered."""

    name = "waiting_reply_reasoning"
    version = "1.0.0"

    def analyze(
        self, state: OrganizationalState, ctx: ReasoningContext
    ) -> list[SignalDraft]:
        out: list[SignalDraft] = []
        for t in state.threads():
            # 1. Someone else has to have spoken last.
            if t.direction != "inbound":
                continue
            # 2. It has to be a conversation, not a broadcast.
            if t.is_automated:
                continue
            if (t.category or "").lower() in _NON_CONVERSATIONAL_CATEGORIES:
                continue
            sender = (t.last_message_from or "").strip().lower()
            if _is_no_reply_sender(sender):
                continue
            # 3. Long enough to matter, recent enough to still be actionable.
            #    The upper bound mirrors the other risk engines: past a month
            #    this is history, and a nudge the founder can neither salvage
            #    nor act on quickly is clutter.
            waiting_days = days_between(ctx.as_of, t.last_seen)
            if waiting_days is None:
                continue
            if waiting_days < WAITING_MIN_DAYS or waiting_days > MAX_SURFACE_AGE_DAYS:
                continue
            # 4. Enough substance to render a card the founder can act on.
            subject = _plain(t.subject)
            if not subject:
                continue

            display = _plain(t.last_message_from_name) or sender
            out.append(
                SignalDraft(
                    signal_type=SignalType.WAITING_ON_USER_REPLY,
                    entity_type="thread",
                    entity_id=t.entity_id,
                    title=f"{display} is waiting for your reply",
                    why=(
                        f'They emailed "{_clip(subject)}" '
                        f"{waiting_days:.0f} days ago and there is no reply from "
                        "you in that thread."
                    ),
                    # Confidence is about the DETECTION, not the importance:
                    # the underlying fact (no outbound message in the thread) is
                    # observed, not inferred, so it starts high and does not
                    # climb with age.
                    confidence=0.9,
                    # Severity DOES climb — the longer someone waits, the worse
                    # it is — but stays bounded so one ancient thread can never
                    # outrank a live commitment risk.
                    severity=min(0.85, 0.35 + waiting_days / 20),
                    dedup_key=t.entity_id,
                    evidence={
                        "person_email": sender,
                        "person_name": _plain(t.last_message_from_name) or None,
                        "thread_id": t.thread_id or t.entity_id,
                        "rfc_message_id": t.rfc_message_id,
                        "subject": subject,
                        "waiting_days": round(waiting_days, 1),
                        "last_inbound_at": t.last_seen,
                        "category": t.category,
                    },
                )
            )
        return out
