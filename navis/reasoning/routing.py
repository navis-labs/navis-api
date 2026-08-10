"""ONE ordered decision about what a chat turn is asking for.

Before this module the chat endpoint asked three independent yes/no questions
(`is_scheduling_intent`, `is_probable_followup`, `is_calendar_query_intent`) and
sent everything that answered "no" to all three into RAG-over-memory. Whole
classes of operational request had no detector at all, so they fell into the
memory path and came back as "Navis doesn't have enough memory to answer this
yet" — which is not just unhelpful, it is FALSE: the answer was sitting in
Google Calendar or Gmail the whole time.

Every one of the reported failures traces to a missing rule, not a broken one:

  "Reschedule the meeting to 5-6 PM"   no reschedule detector existed; and
                                       `\bschedule\b` does not match inside
                                       "reschedule", so even the schedule rule
                                       could not fire.
  "Cancel the meeting"                 no cancel-a-meeting detector; "cancel"
                                       only reached a chip-rendering branch that
                                       prefixed the memory-fallback prose.
  "Send email to x@y.com"              no email detector at all.
  "What meetings do I have this week?" routed correctly, but a calendar error
                                       silently fell THROUGH to memory.
  "What should I do first?"            memory-only by construction.
  "Who am I keeping waiting?"          memory-only by construction.

So: one function, one ordered rule list, one enum. Adding an operational verb
now means adding a rule here — it cannot be forgotten by a caller.

ORDER MATTERS and is asserted by tests. Most-specific verbs first, because the
vocabularies genuinely overlap: "reschedule my meeting" contains a meeting noun
(schedule), a query verb (my), and a mutation verb (reschedule). Only the last
reading is what the founder means.
"""
from __future__ import annotations

import enum
import re

from navis.reasoning.calendar_query.intent import is_calendar_query_intent
from navis.reasoning.scheduling.intent import (
    is_probable_followup,
    is_scheduling_intent,
)


class ChatRoute(str, enum.Enum):
    """Where a chat turn is handled. Exhaustive — every turn gets exactly one."""

    SCHEDULE_MEETING = "schedule_meeting"
    RESCHEDULE_MEETING = "reschedule_meeting"
    CANCEL_MEETING = "cancel_meeting"
    SEND_EMAIL = "send_email"
    CALENDAR_QUERY = "calendar_query"
    # Grounded answer over memory — WITH a Calendar/Gmail fallback so an empty
    # memory no longer means an empty answer. See `reasoning.workspace_context`.
    WORKSPACE_QA = "workspace_qa"


# ── Normalisation ──────────────────────────────────────────────────────
#
# Chat input arrives with the typography a human (or their phone keyboard, or a
# copy-paste out of a calendar invite) produced. "5–6 PM" uses an EN DASH, which
# no `-`-based range regex matches — the reported "Reschedule the meeting to
# 5–6 PM" failure is partly this. Normalise once, here, so every downstream
# extractor sees ASCII.
# Written as escapes, not literals: these glyphs are visually
# indistinguishable from ASCII '-' in most editors, so a literal table is
# unreviewable and one stray paste silently changes behaviour.
_DASHES = str.maketrans(
    {
        "‐": "-",  # hyphen
        "‑": "-",  # non-breaking hyphen
        "‒": "-",  # figure dash
        "–": "-",  # en dash   <- "5–6 PM"
        "—": "-",  # em dash
        "―": "-",  # horizontal bar
        "−": "-",  # minus sign
    }
)
_QUOTES = str.maketrans(
    {
        "‘": "'",  # left single quote
        "’": "'",  # right single quote / apostrophe
        "“": '"',  # left double quote
        "”": '"',  # right double quote
    }
)


def normalise_utterance(utterance: str) -> str:
    """Fold Unicode dashes/quotes to ASCII and collapse whitespace.

    Idempotent and lossless for meaning — it only touches characters whose ASCII
    counterpart is what every parser in the codebase already expects.
    """
    if not utterance:
        return ""
    return " ".join(utterance.translate(_DASHES).translate(_QUOTES).split())


# ── Meeting mutation verbs ─────────────────────────────────────────────

# A meeting-ish noun. Shared by the reschedule + cancel rules; both need to
# distinguish "cancel the meeting" from "cancel my subscription".
_MEETING_NOUN = (
    r"(?:meet(?:ing)?s?|call|sync|chat|1:1|one[- ]?on[- ]?one|invite|"
    r"appointment|standup|kickoff|kick[- ]?off|check[- ]?in|catch\s*up|"
    r"debrief|interview|event|slot|booking)"
)

# "reschedule", "move", "push", "shift", "postpone", "bump", "change the time".
# `reschedule` is listed explicitly because `\bschedule\b` does NOT match inside
# it — the word boundary falls between `e` and `s`, both word characters. That
# single fact is why "Reschedule the meeting to 5-6 PM" reached the memory
# fallback instead of the calendar.
_RESCHEDULE_RE = re.compile(
    rf"\b(?:re-?schedule|rebook|move|push(?:\s+back|\s+out)?|shift|postpone|"
    rf"bring\s+forward|bump|pull\s+forward|change\s+the\s+time\s+of)\b"
    rf"[^.?!]*?\b{_MEETING_NOUN}\b"
    rf"|\b{_MEETING_NOUN}\b[^.?!]*?\b(?:re-?schedule|rebook|moved?|pushed?|"
    rf"shifted?|postponed?|bumped?)\b",
    re.IGNORECASE,
)
# Bare "reschedule to 4 PM" / "move it to Friday" — no explicit noun, but the
# verb is unambiguous and a time/pronoun follows.
_RESCHEDULE_BARE_RE = re.compile(
    r"\b(?:re-?schedule|postpone|rebook)\b|"
    r"\b(?:move|push|shift|bump)\s+(?:it|that|this|them)\b",
    re.IGNORECASE,
)

_CANCEL_RE = re.compile(
    rf"\b(?:cancel|call\s+off|drop|delete|remove|scrap|kill|decline)\b"
    rf"[^.?!]*?\b{_MEETING_NOUN}\b"
    rf"|\b{_MEETING_NOUN}\b[^.?!]*?\b(?:cancel(?:led|led)?|called\s+off)\b",
    re.IGNORECASE,
)
# Bare "cancel it" / "cancel that" while a meeting is the obvious referent.
_CANCEL_BARE_RE = re.compile(
    r"\bcancel\s+(?:it|that|this|them)\b", re.IGNORECASE
)

# Phrases that look like a mutation but are QUESTIONS about one. "Was the
# meeting cancelled?" must not delete anything.
_MUTATION_QUESTION_RE = re.compile(
    r"^\s*(?:was|were|is|are|has|have|did|do|does|why|when|who|which)\b|"
    r"\b(?:did\s+(?:i|we|you)|has\s+(?:it|the)|tell\s+me\s+(?:if|whether))\b",
    re.IGNORECASE,
)


# ── Email ──────────────────────────────────────────────────────────────

_EMAIL_ADDRESS_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)
# "send an email to X", "email X", "write to X", "reply to X", "drop X a note".
_EMAIL_VERB_RE = re.compile(
    r"\b(?:send|shoot|fire\s+off|drop|write|compose|draft|reply|respond|"
    r"follow\s+up)\b",
    re.IGNORECASE,
)
_EMAIL_NOUN_RE = re.compile(
    r"\b(?:e-?mail|mail|note|message|reply|response)\b", re.IGNORECASE
)
# Questions ABOUT email ("what emails came in?") are not send requests.
_EMAIL_QUESTION_RE = re.compile(
    r"^\s*(?:what|which|who|when|how\s+many|any|show|list|summar)", re.IGNORECASE
)
# Statements that merely CONTAIN an address ("my email is x@y.com", "the invoice
# from billing@x.com"). An address is strong evidence of intent to send, but not
# when it is being reported rather than addressed.
_EMAIL_SELF_REFERENCE_RE = re.compile(
    r"\b(?:my|his|her|their|the)\s+(?:e-?mail|address)\s+is\b|"
    r"\b(?:from|by|cc'?d|cced)\s+\S+@",
    re.IGNORECASE,
)


def _is_mutation_question(u: str) -> bool:
    return bool(_MUTATION_QUESTION_RE.search(u))


def is_reschedule_intent(utterance: str) -> bool:
    """True iff the user is asking to MOVE an existing meeting."""
    u = normalise_utterance(utterance)
    if not u or _is_mutation_question(u):
        return False
    return bool(_RESCHEDULE_RE.search(u) or _RESCHEDULE_BARE_RE.search(u))


def is_cancel_meeting_intent(utterance: str) -> bool:
    """True iff the user is asking to CANCEL an existing meeting."""
    u = normalise_utterance(utterance)
    if not u or _is_mutation_question(u):
        return False
    return bool(_CANCEL_RE.search(u) or _CANCEL_BARE_RE.search(u))


def is_send_email_intent(utterance: str) -> bool:
    """True iff the user is asking Navis to SEND an email.

    Requires an addressee signal (a literal address, or "to <someone>") on top
    of the verb+noun pair, so "draft the roadmap" and "what should I reply to
    Priya?" stay out of the execution path.
    """
    u = normalise_utterance(utterance)
    if not u or _EMAIL_QUESTION_RE.search(u) or _EMAIL_SELF_REFERENCE_RE.search(u):
        return False
    has_address = bool(_EMAIL_ADDRESS_RE.search(u))
    has_recipient_phrase = bool(re.search(r"\bto\s+\S", u, re.IGNORECASE))
    if not (has_address or has_recipient_phrase):
        return False
    # A literal address is strong evidence of intent, so either the verb OR the
    # noun suffices — "email priya@x.com about the term sheet" uses "email" as
    # the verb, which no verb list would catch without also swallowing every
    # question about email.
    if has_address:
        return bool(_EMAIL_VERB_RE.search(u) or _EMAIL_NOUN_RE.search(u))
    # No address: require both, so "move it to Friday" and "talk to Priya" stay
    # out of the send path.
    return bool(_EMAIL_VERB_RE.search(u) and _EMAIL_NOUN_RE.search(u))


def classify(utterance: str, *, has_pending_scheduling: bool = False) -> ChatRoute:
    """The single routing decision for a chat turn.

    `has_pending_scheduling` is the frontend echoing back an open scheduling
    clarification. When one is open, a bare fragment ("tomorrow at 3 PM",
    "Rahul") belongs to that flow and nothing else — so it is checked FIRST.
    """
    u = normalise_utterance(utterance)
    if not u:
        return ChatRoute.WORKSPACE_QA

    # 0. An open scheduling clarification owns plausible follow-ups.
    if has_pending_scheduling and is_probable_followup(u):
        return ChatRoute.SCHEDULE_MEETING

    # 1. Mutations of an EXISTING meeting, before creation of a new one:
    #    "reschedule the meeting" contains a meeting noun and would otherwise be
    #    read as a fresh booking (or, historically, as nothing at all).
    if is_cancel_meeting_intent(u):
        return ChatRoute.CANCEL_MEETING
    if is_reschedule_intent(u):
        return ChatRoute.RESCHEDULE_MEETING

    # 2. Creating a new meeting.
    if is_scheduling_intent(u):
        return ChatRoute.SCHEDULE_MEETING

    # 3. Outbound email.
    if is_send_email_intent(u):
        return ChatRoute.SEND_EMAIL

    # 4. Reading the calendar.
    if is_calendar_query_intent(u):
        return ChatRoute.CALENDAR_QUERY

    # 5. Everything else is a question about the workspace. NOT "memory only" —
    #    the handler falls back to Calendar + Gmail before ever claiming it has
    #    nothing to say.
    return ChatRoute.WORKSPACE_QA
