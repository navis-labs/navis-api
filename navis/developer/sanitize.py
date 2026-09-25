"""Bounding and scrubbing the metadata an external agent sends.

THE THREAT THIS MODULE EXISTS FOR. A run ledger is written by an agent, from
somebody else's process, with arguments assembled by a model. Two things will
therefore arrive that must never be stored:

  1. SECRETS. An agent that calls an authenticated REST tool will, sooner or
     later, put the Authorization header in the arguments it reports. Nobody
     decided to do that; it is what happens when a model serializes a request
     object. If Navis persists it, a credential for a THIRD-PARTY system is now
     in Navis's database, in Navis's backups, and on the screen of anyone who can
     read the run — and the customer has no idea.

  2. UNBOUNDED PAYLOADS. A loop that reports a large prompt on every step fills a
     shared table. The first symptom is another tenant's list endpoint slowing
     down.

So nothing reaches a column without passing through here.

WHY REDACT RATHER THAN REFUSE. A refusal loses the whole step, including the
parts that were fine, and an agent author debugging a 400 will most likely stop
sending metadata at all — which costs the observability this domain exists to
provide. A redacted value keeps the shape of what happened ("there was an
`authorization` argument") while destroying the secret. Size violations DO refuse,
because truncating a payload makes the record lie about what was sent.

WHY A DENY-LIST, WITH ITS EYES OPEN. A deny-list cannot be complete, and this one
is not claimed to be. It is the second line, not the first: the first is that
this domain never asks for credentials, so anything matching below is already a
mistake. It is paired with a VALUE-shaped check, because the field a secret
arrives under is frequently not named like one (`h`, `opts`, `arg2`).
"""
from __future__ import annotations

import json
import re
from typing import Any

from navis.developer.lifecycle import (
    MAX_METADATA_BYTES,
    MAX_METADATA_KEYS,
)

#: What a redacted value becomes. A fixed marker, never a hash and never a
#: prefix: a hash of a secret is still a credential oracle, and a prefix is what
#: makes a leaked key findable.
REDACTED = "[redacted]"


class PayloadTooLarge(ValueError):
    """A metadata object exceeded its bound. Refused, never truncated."""


#: Key stems whose VALUE is dropped, matched as a SUBSTRING of the normalized
#: key.
#:
#: The key is normalized before matching — lower-cased with `-`, `_`, `.` and
#: spaces removed — so one stem covers every spelling a caller might use.
#: `x-api-key`, `apiKey`, `API_KEY` and `provider.api.key.2` all normalize to
#: something containing `apikey`. Writing the stems with separators is how the
#: first version of this list missed `x-api-key` entirely.
#:
#: ── A SUBSTRING STEM MUST BE A WORD NO INNOCENT KEY CONTAINS ──────────
#: Normalization destroys word boundaries, so a short stem matches anywhere,
#: and a stem shorter than the words it can hide inside redacts them too.
#: `pin` was destroying the values of `mapping`, `shipping`, `spinner`,
#: `pinned` and `opinion` — none of which is a credential, and one of which
#: (`mapping`) is an ordinary key an agent would plausibly report.
#:
#: So a stem belongs HERE only when it cannot occur inside an ordinary English
#: word. Anything shorter goes in `_EXACT_KEY_STEMS` below.
#:
#: `session` deliberately STAYS a substring stem, and the reasoning is worth
#: recording because the opposite change looked obviously right. It redacts
#: `sessionId`, which in THIS domain is an agent harness correlation id and not
#: a credential — but in a great many systems a session identifier IS the
#: bearer credential, and this list is matched against keys written by agents
#: reporting on arbitrary third-party tools, not against Navis's own vocabulary.
#: Nothing is lost by keeping it: a correlation id already has a first-class
#: home in `agent_runs.correlation_id`, which is indexed, unredacted, and
#: documented as not a secret. A caller that needs the value readable should
#: write it there rather than ask this list to get narrower.
_SENSITIVE_KEY_STEMS: tuple[str, ...] = (
    "authorization",
    # Deliberately broad. It also matches an innocent `author` field, which is
    # the trade this module accepts on purpose: a redacted author name is an
    # annoyance, a stored credential is an incident. Anything narrower misses
    # the very common bare `auth` field.
    "auth",
    "apikey",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "privatekey",
    "session",
    "cookie",
    "signature",
    "bearer",
)

#: Key stems matched against the WHOLE normalized key, never inside it.
#:
#: These are real credential names that are also common English fragments. An
#: exact match keeps `pin` and `otp` covered while leaving `mapping` and
#: `shipping` alone. Compound spellings do not need to be listed: `pin_code`
#: normalizes to `pincode` and is caught here, and `one_time_password` is
#: already caught by the `password` stem above.
_EXACT_KEY_STEMS: frozenset[str] = frozenset({"pin", "pincode", "userpin", "otp", "otpcode"})

#: Characters removed before a key is matched against the stems above.
_KEY_SEPARATORS = str.maketrans({"-": None, "_": None, ".": None, " ": None})

#: Value shapes that are credentials whatever they are called. This is what
#: catches the argument named `h` that happens to hold a bearer token.
_SENSITIVE_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # ── EVERY PATTERN HERE MUST CONSUME THE WHOLE SECRET ──────────────
    # Not a style rule — a correctness condition of replacing spans instead
    # of whole values. While a match blanked the entire field, a pattern that
    # matched only a credential's PREFIX was still safe, because everything
    # after it went too. Replacing the span alone, a prefix-only pattern
    # deletes the recognisable part and PUBLISHES THE SECRET:
    #
    #     navis_[0-9a-f]{8,}_   on  navis_f364260edba0f931_rdixdSnD…
    #                          ->  [redacted]rdixdSnD…
    #
    # So this pattern now runs to the end of the key, and the PEM rule below
    # runs to the end of the block. Any pattern added here must do the same.
    #
    # Navis's own credential format. Leaking one of these into a run record
    # would be the platform disclosing its own keys.
    re.compile(r"navis_[0-9a-f]{8,}_[A-Za-z0-9_\-]*", re.IGNORECASE),
    # ── Shape, not vocabulary ─────────────────────────────────────────
    # This was `\bbearer\s+\S+`, which matches the keyword plus whatever
    # follows it — including English. Combined with whole-value replacement
    # in `sanitize_text` (see below), one such phrase anywhere in a field
    # destroyed the WHOLE field: a 2,637-character agent checkpoint
    # describing a day's work was stored as the ten characters
    # "[redacted]", and nothing reported the loss.
    #
    # It also matched its own output. "Bearer [redacted]" — a correctly
    # redacted header — still satisfied `\S+`, so sanitizing an
    # already-sanitized value destroyed it a second time.
    #
    # The discriminator is the one production scanners use and the one the
    # CLI's own redactor already uses: a credential does not look like a
    # word. Not a short all-lower-case word, and at least 16 characters —
    # which "tokens", "of", and "[redacted]" all fail and no real bearer
    # token does.
    re.compile(r"\bbearer\s+(?![a-z]{1,15}\b)[A-Za-z0-9._~+/\-]{16,}", re.IGNORECASE),
    re.compile(r"\bbasic\s+[A-Za-z0-9+/=]{16,}", re.IGNORECASE),
    # Common vendor prefixes.
    re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),
    # A JWT, whatever it is called.
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+"),
    # PEM material — the WHOLE block, header through footer. Matching only
    # the `-----BEGIN …-----` line would, under span replacement, remove the
    # marker and leave the key material itself in the record.
    #
    # The alternation ends the match at the end of the text when the footer
    # is missing, so a truncated block is redacted rather than skipped.
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?"
        r"(?:-----END [A-Z ]*PRIVATE KEY-----|$)"
    ),
)

#: How deep a nested object may go before it is flattened away. Depth is a
#: denial-of-service axis of its own: a deeply nested structure costs recursion
#: on every read as well as every write.
_MAX_DEPTH = 4


def _key_is_sensitive(key: str) -> bool:
    """Normalize away separators, then look for a stem.

    `X-Api-Key` -> `xapikey`, which contains `apikey`. Matching the raw string
    against separator-bearing stems is what let `x-api-key` through before.

    Two passes, because normalization removes the word boundaries a short stem
    would otherwise need: a substring pass for stems that cannot occur inside
    an ordinary word, and an exact pass for the ones that can. See
    `_EXACT_KEY_STEMS`.
    """
    normalized = key.lower().translate(_KEY_SEPARATORS)
    if normalized in _EXACT_KEY_STEMS:
        return True
    return any(stem in normalized for stem in _SENSITIVE_KEY_STEMS)


def _value_is_sensitive(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SENSITIVE_VALUE_PATTERNS)


def _scrub(value: Any, depth: int = 0) -> Any:
    """Recursively redact, with depth and type bounds.

    Types are narrowed deliberately: the output is JSON-safe scalars, lists and
    dicts only. Anything else becomes its `repr`, scrubbed — an agent sending a
    datetime or a custom object should not fail the write, but neither should an
    arbitrary object graph reach JSONB.
    """
    if depth > _MAX_DEPTH:
        return REDACTED

    if isinstance(value, dict):
        scrubbed: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if _key_is_sensitive(key):
                scrubbed[key] = REDACTED
            else:
                scrubbed[key] = _scrub(raw_value, depth + 1)
        return scrubbed

    if isinstance(value, (list, tuple)):
        return [_scrub(item, depth + 1) for item in value]

    if isinstance(value, str):
        return REDACTED if _value_is_sensitive(value) else value

    if isinstance(value, bool) or value is None:
        return value

    if isinstance(value, (int, float)):
        return value

    text = repr(value)
    return REDACTED if _value_is_sensitive(text) else text


def sanitize_metadata(value: Any, *, field: str) -> dict:
    """Scrub, bound and return a metadata object ready for JSONB.

    `field` names the column in the error, so an agent author is told which of
    several metadata objects on a step was rejected.

    Raises `PayloadTooLarge` for a violated bound. Size and breadth are refused
    rather than trimmed: dropping keys to fit would produce a record that looks
    complete and is not.
    """
    if value is None:
        return {}

    if not isinstance(value, dict):
        raise PayloadTooLarge(f"{field} must be an object")

    if len(value) > MAX_METADATA_KEYS:
        raise PayloadTooLarge(
            f"{field} has {len(value)} keys; the maximum is {MAX_METADATA_KEYS}"
        )

    scrubbed = _scrub(value)

    # Measured AFTER scrubbing, on the bytes that would actually be stored.
    # `default=str` so an unserializable leftover cannot raise here — `_scrub`
    # has already reduced those to strings, and this is belt and braces on the
    # path where a size error is the correct answer.
    encoded = json.dumps(scrubbed, separators=(",", ":"), default=str)
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise PayloadTooLarge(
            f"{field} is larger than {MAX_METADATA_BYTES} bytes once encoded"
        )

    return scrubbed


def sanitize_text(value: str | None, *, field: str, limit: int) -> str:
    """Bound a free-text field, and scrub a secret out of it.

    Text is scrubbed as well as bounded because a task description is written by
    a human or a model and is exactly the place someone pastes a curl command
    with a header in it.

    ── THE MATCH IS REPLACED, NOT THE DOCUMENT ────────────────────────
    This used to return `REDACTED` for the whole value if any pattern matched
    anywhere in it. On a metadata VALUE that is defensible — the value is the
    credential. On free text it is destruction: these fields carry a task
    description and an agent's outcome summary, which are paragraphs, and one
    credential-shaped substring in one of them deleted everything else.

    Observed in production, on this platform's own ledger: a 2,637-character
    checkpoint recording a day of engineering work was stored as the ten
    characters "[redacted]". The write returned 200, the client reported
    success, and the next session restored an empty checkpoint — the loss was
    total and silent, and it hid an older, intact checkpoint underneath it
    because the blanked row was newer.

    Replacing the matched SPANS keeps the credential out and the sentence in,
    which is what every production secret scanner does. The result is also
    STABLE under re-sanitizing: no pattern here matches `[redacted]`, so
    passing a sanitized value through again returns it unchanged.
    """
    if value is None:
        return ""

    if not isinstance(value, str):
        raise PayloadTooLarge(f"{field} must be a string")

    if len(value) > limit:
        raise PayloadTooLarge(f"{field} is longer than {limit} characters")

    return redact_spans(value)


def redact_spans(value: str) -> str:
    """Replace every credential-shaped SPAN, leaving the rest of the text."""
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        value = pattern.sub(REDACTED, value)
    return value
