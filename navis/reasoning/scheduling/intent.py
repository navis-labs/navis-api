"""Deterministic scheduling-intent detector.

Kept separate from the LLM slot-filler so:
  * false positives are cheap (regex miss, no LLM call);
  * unit tests can pin the exact utterance shapes we route through
    scheduling vs. the general chat pipeline.

Any change to the surface phrases here must round-trip through the
tests in `tests/reasoning/test_scheduling_agent.py::test_intent_*`.
"""
from __future__ import annotations

import re

# Scheduling verbs. Requires a paired meeting-noun via `_NOUN_RE` (or a
# composite phrase via `_COMPOSITE_RE`) so plain requests like
# "book the flight" or "plan the roadmap" don't misclassify.
_VERB_RE = re.compile(
    r"\b(schedule|book|invite|set\s+up|set-up|arrange|organi[sz]e|"
    r"put\s+on\s+the\s+calendar|find\s+time|"
    # Indian English / informal — "fix a meeting", "lock in a call",
    # "plan a sync", "put together a review". All safe because they
    # ONLY trigger when a meeting-noun follows.
    r"fix|lock\s+in|plan|put\s+together)\b",
    re.IGNORECASE,
)

# Meeting nouns. A single word from this set combined with a scheduling
# verb triggers routing. Kept narrow so "meeting notes" or "call the
# API" don't drag prose into scheduling.
_NOUN_RE = re.compile(
    r"\b(meet(?:ing)?s?|call|sync|chat|catch\s*up|1:1|one[- ]?on[- ]?one|"
    r"interview|coffee|standup|kickoff|kick[- ]?off|check[- ]?in|"
    r"debrief|touch[- ]?base|huddle)\b",
    re.IGNORECASE,
)

# Standalone "meet with X" / "catch up with X" / "grab coffee with X" —
# verb + noun collapsed into one phrase.
_COMPOSITE_RE = re.compile(
    r"\b(meet\s+(?:with\s+)?|catch\s*up\s+with\s+|grab\s+coffee\s+with\s+|"
    r"get\s+on\s+a\s+call\s+with\s+|hop\s+on\s+a\s+call\s+with\s+)",
    re.IGNORECASE,
)

# Anti-noun sentinels that DEFEAT verb+noun matches when they appear.
# "meeting notes" isn't scheduling; "call the API" isn't scheduling;
# "meeting schedule" (noun phrase, e.g. "tell me my meeting schedule")
# isn't creating a meeting — it's asking about EXISTING ones.
_ANTI_PATTERNS = (
    # Non-scheduling uses of "meeting" / "call" / "sync"
    re.compile(r"\bmeeting\s+notes?\b", re.IGNORECASE),
    re.compile(r"\bmeeting\s+minutes\b", re.IGNORECASE),
    re.compile(r"\bmeeting\s+room\b", re.IGNORECASE),
    re.compile(r"\bmeeting\s+schedule\b", re.IGNORECASE),  # noun phrase
    re.compile(r"\bcall\s+the\s+api\b", re.IGNORECASE),
    re.compile(r"\bapi\s+call\b", re.IGNORECASE),
    re.compile(r"\bsync\s+(?:the\s+)?(?:branch|repo|repository|code|state|data)\b", re.IGNORECASE),
    re.compile(r"\bsync\s+up\s+on\b", re.IGNORECASE),  # "sync up on X" = discuss
    # Query intent — the user is ASKING ABOUT meetings, not creating one.
    # These utterances contain scheduling nouns/verbs but the semantic is
    # "list / count / show / prepare for" existing ones. Routing them to
    # the scheduling agent produces confusing attendee clarifications.
    re.compile(r"^\s*(tell|show|list|count|prepare|remind|explain)\b", re.IGNORECASE),
    re.compile(r"\btell\s+(?:me\s+)?(?:about|how|what|when|which|any)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+many\b", re.IGNORECASE),
    re.compile(r"\bshow\s+(?:me\s+)?", re.IGNORECASE),
    re.compile(r"\bwhat\s+meetings?\b", re.IGNORECASE),
    re.compile(r"\bwhat.?s\s+on\s+my\s+(?:calendar|schedule)\b", re.IGNORECASE),
    re.compile(r"\bwhen\s+is\s+my\s+(?:next|upcoming)?\s*(?:meeting|call|sync)\b", re.IGNORECASE),
    re.compile(r"\bprepare\s+(?:me\s+)?for\b", re.IGNORECASE),
    # Explicit "cancel scheduling" phrases
    re.compile(r"\bnot\s+schedul", re.IGNORECASE),
    re.compile(r"\bnevermind\b", re.IGNORECASE),
    re.compile(r"\bforget\s+it\b", re.IGNORECASE),
)


def is_scheduling_intent(utterance: str) -> bool:
    """True iff the utterance is asking Navis to schedule a meeting.

    Requires (verb + meeting-noun) OR a composite phrase, AND that none
    of the anti-patterns fire. Conservative on purpose — false positives
    hijack the LLM slot-filler and burn inference budget on non-scheduling
    prose.
    """
    if not utterance or not utterance.strip():
        return False
    # Anti-patterns short-circuit even a strong verb+noun match.
    if any(p.search(utterance) for p in _ANTI_PATTERNS):
        return False
    if _COMPOSITE_RE.search(utterance):
        return True
    return bool(_VERB_RE.search(utterance) and _NOUN_RE.search(utterance))


# ── Follow-up detection ────────────────────────────────────────────────
# When a Clarification is pending, the user's next reply is usually a
# bare fragment ("tomorrow at 3 PM", "3 PM", "Friday", "Rahul"). Those
# don't match `is_scheduling_intent` on their own. `is_probable_followup`
# is a permissive check the chat router uses to route SHORT utterances
# through the scheduling agent WHEN a pending state exists.
#
# The router logic is:
#   is_scheduling_intent(msg) OR (pending is not None AND is_probable_followup(msg))
# So this only fires when there's already pending scheduling context.

# Disqualifiers — patterns that mean the utterance is NOT a scheduling
# follow-up even if pending state is present. Split into two groups:
#   * STARTS_WITH — a query verb at position 0 ("show me the calendar")
#   * ANYWHERE    — a query phrase or explicit cancel signal anywhere
#     in the utterance ("not schedule just tell me how many meetings")
_FOLLOWUP_DISQUALIFY_START = re.compile(
    r"^\s*(what|who|when|where|why|how|show|find|open|search|list|browse|"
    r"tell|explain|count|prepare|remind|can\s+you)\b",
    re.IGNORECASE,
)
_FOLLOWUP_DISQUALIFY_ANYWHERE = (
    re.compile(r"\bhow\s+many\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\b", re.IGNORECASE),
    re.compile(r"\bshow\s+me\b", re.IGNORECASE),
    re.compile(r"\bnot\s+schedul", re.IGNORECASE),
    re.compile(r"\bnevermind\b", re.IGNORECASE),
    re.compile(r"\bforget\s+it\b", re.IGNORECASE),
    re.compile(r"\bprepare\s+(?:me\s+)?for\b", re.IGNORECASE),
)


def is_probable_followup(utterance: str) -> bool:
    """True when the utterance plausibly answers a scheduling clarification.

    Used ONLY when `pending` state already exists — the router combines
    this with pending presence to route follow-ups through the agent.
    Rejects utterances that START with a query verb OR contain any of
    the explicit disqualifier phrases anywhere. Everything else — dates,
    times, names, emails, short fragments — is treated as a follow-up.
    """
    if not utterance or not utterance.strip():
        return False
    if _FOLLOWUP_DISQUALIFY_START.search(utterance):
        return False
    if any(p.search(utterance) for p in _FOLLOWUP_DISQUALIFY_ANYWHERE):
        return False
    return True
