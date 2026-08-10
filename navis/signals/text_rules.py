"""Shared linguistic rules for commitment / decision detection.

These predicates are used by TWO consumers that must never disagree:

  * Memory extraction (`navis.memory.extractors.email` / `.slack`) — deciding
    whether a sentence becomes a COMMITMENT or DECISION fact;
  * Signal importance (`navis.signals.importance`) — deciding how much a message
    matters to a founder.

They live in the Signals layer because Memory already imports Signals and never
the reverse; putting them here lets both consumers share one definition instead
of drifting apart. Nothing here touches the database, the clock, or a model —
pure functions of a string, so the same text always yields the same verdict and
a replay is exact.
"""
from __future__ import annotations

import re

# Straight AND typographic apostrophe — mail clients emit both.
_APOS = "['’]"

# The apostrophe is REQUIRED on contractions. An earlier `\bwe'?ll\b` made it
# optional, so the pattern also matched the ordinary word "well" — every
# "I hope you are doing well" / "as well" / "well-oiled machine" in the corpus
# became a "commitment". Same defect in `\bi'?ll\b`, which matched "ill".
_COMMITMENT_RE = re.compile(
    "|".join(
        [
            rf"\bi{_APOS}ll\b", rf"\bwe{_APOS}ll\b", r"\bi will\b", r"\bwe will\b",
            rf"\bwe{_APOS}re going to\b", r"\bwe are going to\b", r"\bi am going to\b",
            r"\bfollow(?:ing)? up\b", r"\bget back to you\b",
            r"\bsend (?:you|over|across)\b", r"\bcircle back\b",
            r"\bby (?:eod|end of day|tomorrow|monday|tuesday|wednesday|thursday|friday|next week)\b",
        ]
    ),
    re.IGNORECASE,
)

# Bare "approved" matched newsletter prose like "The government approved
# construction of a 14.52 km line" and "the tool catching AI agents making
# product decisions nobody approved". A decision is something THIS org made, so
# every alternative needs a first-person subject or an explicit marker.
_DECISION_RE = re.compile(
    "|".join(
        [
            r"\bwe (?:have )?decided\b", rf"\bwe{_APOS}ve decided\b",
            r"\bdecision:", r"\bwe (?:have )?agreed\b", rf"\bwe{_APOS}ve agreed\b",
            r"\bwe (?:have )?approved\b", rf"\bwe{_APOS}ve approved\b",
            r"\bwe will go with\b", rf"\bwe{_APOS}re going with\b",
            rf"\blet{_APOS}?s go with\b",
        ]
    ),
    re.IGNORECASE,
)

# Complete social pleasantries. These are never commitments, but they sit right
# next to commitment vocabulary, so they are rejected unless the SAME sentence
# also carries an explicit first-person future ("...doing well. I'll send it").
_YOU_ARE = rf"you(?:{_APOS}re| are)?"
_PLEASANTRY_RE = re.compile(
    "|".join(
        [
            r"\bhope (?:this|the) (?:e-?mail|message|note) finds you well\b",
            r"\bhope (?:" + _YOU_ARE + r"|all is|everything is|things are) [^.!?]{0,20}\bwell\b",
            r"\bhope " + _YOU_ARE + r" doing well\b",
            r"\btrust (?:this|" + _YOU_ARE + r") [^.!?]{0,20}\bwell\b",
        ]
    ),
    re.IGNORECASE,
)
_EXPLICIT_PROMISE_RE = re.compile(
    rf"\bi{_APOS}ll\b|\bwe{_APOS}ll\b|\bi will\b|\bwe will\b", re.IGNORECASE
)

# A character-length floor alone lets through degenerate matches: the literal
# heading "*Decision:*" is 11 characters and became a decision fact carrying no
# information. Require enough real words to constitute a statement.
_MIN_WORDS = 4
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def sentences(body: str) -> list[str]:
    """Split a message body into candidate sentences (deterministic)."""
    return [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", body) if s.strip()]


def _has_substance(sentence: str) -> bool:
    return len(_WORD_RE.findall(sentence)) >= _MIN_WORDS


def is_commitment_sentence(sentence: str) -> bool:
    """True if this sentence states a promise someone made."""
    if not (8 <= len(sentence) <= 400) or not _has_substance(sentence):
        return False
    if not _COMMITMENT_RE.search(sentence):
        return False
    # A pure social pleasantry is not a promise; keep it only when the same
    # sentence also states an explicit first-person future.
    if _PLEASANTRY_RE.search(sentence) and not _EXPLICIT_PROMISE_RE.search(sentence):
        return False
    return True


def is_decision_sentence(sentence: str) -> bool:
    """True if this sentence records a decision the organization made."""
    return bool(
        8 <= len(sentence) <= 400
        and _has_substance(sentence)
        and _DECISION_RE.search(sentence)
    )


# ── Hedged / non-binding language ──────────────────────────────────────
#
# `is_commitment_sentence` matches on promise VOCABULARY, which is necessary but
# not sufficient: "Maybe I'll send it Friday" contains "I'll" and is not a
# promise. Chasing someone over something they never committed to is exactly the
# behaviour that makes a Chief of Staff feel wrong, so a hedge disqualifies the
# sentence outright rather than lowering a score.
_HEDGE_RE = re.compile(
    "|".join(
        [
            r"\bmaybe\b", r"\bperhaps\b", r"\bpossibly\b", r"\bprobably\b",
            r"\bmight\b", r"\bcould\b", r"\bmay\b(?!\s+\d)",  # "may 5" is a date
            r"\bif (?:i|we) (?:can|get|have|manage)\b",
            r"\bhopefully\b", r"\bno promises\b", r"\bnot sure (?:if|whether)\b",
            r"\btry(?:ing)? to\b", r"\bshould be able to\b",
            r"\bwould (?:like to|try)\b", r"\bthinking of\b", r"\bplanning to\b",
        ]
    ),
    re.IGNORECASE,
)

# Suggestions, not promises: "We should send the deck" proposes work, it does
# not commit anyone to it. Distinct from _HEDGE_RE because these are about the
# MODAL VERB rather than uncertainty.
_SUGGESTION_RE = re.compile(
    r"\b(?:we|you|i) should\b|\blet{a}?s (?:think|discuss|consider)\b".format(a=_APOS),
    re.IGNORECASE,
)


# A request ASKS the reader to do something; a commitment promises the reader
# that the writer will. `_COMMITMENT_RE` cannot tell them apart on its own,
# because several of its alternatives carry no subject at all — the bare
# "by friday" branch matches "Can you send me the deck by Friday?" exactly as
# well as "I'll send the deck by Friday". Treating that as a promise would have
# Navis chase somebody over work they ASKED FOR.
_REQUEST_AT_READER_RE = re.compile(
    "|".join(
        [
            r"\b(?:can|could|would|will) you\b",
            r"\bplease (?:send|share|provide|confirm|get|forward|let)\b",
            r"\bare you able to\b", r"\bdo you mind\b",
            r"\bwe need you to\b", r"\bcould we get\b",
        ]
    ),
    re.IGNORECASE,
)


def is_binding_commitment(sentence: str) -> bool:
    """True iff the sentence is a promise someone can actually be held to.

    `is_commitment_sentence` decides "does this look like promise language";
    this decides "is it BINDING". Both must pass before Navis will chase anyone,
    because a follow-up on a hedge ("maybe I'll send it") — or worse, on a
    request the founder themselves made — is far more damaging than silence.
    """
    if not is_commitment_sentence(sentence):
        return False
    if _HEDGE_RE.search(sentence):
        return False
    if _SUGGESTION_RE.search(sentence):
        return False
    # Requests and bare questions are only promises when the SAME sentence also
    # states an explicit first-person future ("Can you confirm? I'll send it
    # Friday" is both an ask and a promise).
    explicit_promise = bool(_EXPLICIT_PROMISE_RE.search(sentence))
    if _REQUEST_AT_READER_RE.search(sentence) and not explicit_promise:
        return False
    if sentence.rstrip().endswith("?") and not explicit_promise:
        return False
    return True


# ── Decision REQUESTS ──────────────────────────────────────────────────
#
# The mirror image of `is_decision_sentence`, which records a decision the org
# already MADE ("we decided to raise"). This detects someone ASKING the founder
# to decide — the thing that then sits unanswered.
#
# Requires an explicit ask directed at the reader. Bare questions do not count:
# "how are you?" and "did you see the news?" are questions, not decisions owed.
_DECISION_REQUEST_RE = re.compile(
    "|".join(
        [
            # Direct approval / confirmation asks.
            r"\b(?:can|could|would) you (?:please )?(?:approve|confirm|sign off|decide|choose|pick)\b",
            r"\bplease (?:approve|confirm|decide|advise|sign off|let us know whether)\b",
            r"\b(?:need|needs|awaiting|waiting for) your (?:approval|sign[- ]?off|decision|go[- ]?ahead|call)\b",
            r"\byour (?:approval|sign[- ]?off|decision|go[- ]?ahead) (?:is )?(?:needed|required)\b",
            r"\bapproval needed\b", r"\bawaiting approval\b",
            # Choice between options / whether to proceed.
            r"\bshould we\b", r"\bshall we\b", r"\bdo you want us to\b",
            r"\bare we going (?:with|ahead)\b",
            r"\bwhich (?:option|one|approach|way) (?:do you|should we|would you)\b",
            r"\blet (?:me|us) know (?:if|whether) we should\b",
            r"\bconfirm whether\b", r"\bdecide (?:on|between|whether)\b",
            r"\bok(?:ay)? to proceed\b", r"\bgo/no[- ]?go\b",
        ]
    ),
    re.IGNORECASE,
)

# Statements that CONTAIN request vocabulary but are informational. "Here's the
# decision we made" and "FYI the launch is Monday" must never become a card
# telling the founder they owe an answer.
_INFORMATIONAL_RE = re.compile(
    "|".join(
        [
            r"^\s*fyi\b", r"\bjust so you know\b", r"\bfor your (?:information|awareness)\b",
            r"\bno (?:action|response|reply) (?:is )?(?:needed|required)\b",
            r"\bhere{a}?s the decision\b".format(a=_APOS),
            r"\bwe (?:have )?(?:already )?decided\b",
            r"\bthis (?:is|was) (?:just )?an? fyi\b",
        ]
    ),
    re.IGNORECASE,
)


def is_decision_request_sentence(sentence: str) -> bool:
    """True iff the sentence asks the reader to make or confirm a decision."""
    if not (8 <= len(sentence) <= 400) or not _has_substance(sentence):
        return False
    if _INFORMATIONAL_RE.search(sentence):
        return False
    # A recorded decision is the OPPOSITE of a request for one; when a sentence
    # reads as both, the record wins ("we decided we should go with A").
    if _DECISION_RE.search(sentence):
        return False
    return bool(_DECISION_REQUEST_RE.search(sentence))


# ── Fulfilment ─────────────────────────────────────────────────────────
#
# "I've attached it" / "sent it over" / "here's the deck" — evidence that
# somebody DID the thing they promised. Used to suppress a follow-up, so it is
# deliberately generous: silently dropping a nudge because someone said they
# delivered is a far cheaper error than chasing a person who already did.
_FULFILMENT_RE = re.compile(
    "|".join(
        [
            r"\battach(?:ed|ing)\b", r"\bplease find\b", r"\benclosed\b",
            rf"\bhere{_APOS}?s the\b", r"\bhere is the\b",
            rf"\b(?:i|we){_APOS}ve (?:sent|shared|attached|uploaded|added)\b",
            r"\b(?:i|we) (?:have )?(?:just )?(?:sent|shared|attached|uploaded)\b",
            r"\bsent (?:it|this|that|them|the) (?:over|across|to you|your way)\b",
            r"\bsharing (?:the|it|this)\b", r"\bas promised\b",
            r"\bthis is (?:now )?done\b", r"\ball done\b", r"\bcompleted\b",
        ]
    ),
    re.IGNORECASE,
)


def is_fulfilment_sentence(sentence: str) -> bool:
    """True iff the sentence claims the promised thing was delivered."""
    if not (5 <= len(sentence) <= 400):
        return False
    return bool(_FULFILMENT_RE.search(sentence))


def body_shows_fulfilment(body: str) -> bool:
    """True iff ANY sentence in the body claims delivery."""
    return any(is_fulfilment_sentence(s) for s in sentences(body))


def body_records_decision(body: str) -> bool:
    """True iff ANY sentence records a decision the org made."""
    return any(is_decision_sentence(s) for s in sentences(body))
