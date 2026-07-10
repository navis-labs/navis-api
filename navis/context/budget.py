"""Deterministic, provider-independent token estimation for budgeting.

This is NOT a vendor tokenizer. It estimates a STABLE, relative cost so the same entity
always costs the same number of "tokens" across replays, Python versions, and regions. It
never imports tiktoken / SentencePiece / any model SDK, makes no network calls, and never
tries to predict OpenAI/Anthropic/Gemini/Llama counts — Reasoning may switch models freely
without touching this module. It is a pure Value-Object transform: no State/Context
mutation, no randomness, no clock, no locale dependence.

WORDS_PER_TOKEN (0.75) and the whitespace-run word count are a deliberate, deterministic
heuristic — kept CONSTANT on purpose: making it configurable or model-specific would break
cross-deployment replay (the same entity must cost the same everywhere), and any change to
the estimate's MAGNITUDE is an assembly-logic change that must bump CONTEXT_VERSION. It is a
relative budget proxy, not a vendor-accurate count, and is intentionally language-agnostic
(it counts non-whitespace runs, not English words). A size/character-proportional refinement
is possible but deferred — it would change every estimate and therefore requires a
CONTEXT_VERSION bump; it is NOT a correctness blocker for budgeting.

Fail-safe: canonicalization can NEVER raise (see _canonical) — a malformed entity yields a
deterministic estimate, never an exception, so bad data can never crash ContextService.
"""
from __future__ import annotations

import json
import re
from typing import Any

_WORD = re.compile(r"\S+")
# Provider-independent constant — see module docstring (changing it requires a CONTEXT_VERSION bump).
_WORDS_PER_TOKEN = 0.75
# Bounds recursion so a circular or pathologically deep structure degrades to a marker
# instead of RecursionError / a serialization explosion (well under Python's recursion limit).
_MAX_DEPTH = 64
_TRUNCATED = "…"


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate from a canonical string. Pure arithmetic over a whitespace
    -run count — never raises on a str, minimum 1 token."""
    return max(1, int(len(_WORD.findall(text)) / _WORDS_PER_TOKEN))


def _canonicalize(obj: Any, depth: int = 0) -> Any:
    """Coerce arbitrary content into a deterministic, JSON-safe structure that
    json.dumps(sort_keys=True) can serialize without raising and without order ambiguity:

      * dict        -> keys stringified (handles mixed/non-str keys → no sort_keys TypeError),
                       values recursed; json.dumps then sorts the now-uniform str keys.
      * list/tuple  -> recursed list (tuples already serialize like lists, made explicit).
      * set/frozenset -> SORTED list (str-key) — a set's iteration order is hash-seed
                         dependent; sorting makes it deterministic across runs.
      * scalars     -> as-is (str/bool/int/float/None).
      * anything else (bytes, Decimal, UUID, datetime, dataclass, arbitrary objects) -> str().
    Recursion is depth-bounded (circular / over-deep → _TRUNCATED). Every coercion that could
    raise (str(), sorted()) is guarded, so this function never throws."""
    if depth >= _MAX_DEPTH:
        return _TRUNCATED
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            try:
                skey = key if isinstance(key, str) else str(key)
            except Exception:
                skey = _TRUNCATED
            out[skey] = _canonicalize(value, depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v, depth + 1) for v in obj]
    if isinstance(obj, (set, frozenset)):
        try:
            ordered = sorted(obj, key=str)
        except Exception:
            ordered = list(obj)
        return [_canonicalize(v, depth + 1) for v in ordered]
    try:
        return str(obj)
    except Exception:
        return _TRUNCATED


def _safe_get(raw: dict[str, Any], key: str) -> Any:
    try:
        return raw.get(key, {})
    except Exception:
        return {}


def _canonical(raw: dict[str, Any]) -> str:
    """Canonical, deterministic serialization of an entity's reasoning-relevant content
    (fields + lists + links). NEVER raises: content is canonicalized to a JSON-safe,
    order-stable structure first, so json.dumps(sort_keys=True) cannot throw on mixed keys,
    sets, depth, or exotic values. Sorted keys ⇒ identical bytes for the same entity across
    replays, Python versions, and regions."""
    try:
        payload = {
            "fields": _canonicalize(_safe_get(raw, "fields"), 1),
            "lists": _canonicalize(_safe_get(raw, "lists"), 1),
            "links": _canonicalize(_safe_get(raw, "links"), 1),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except Exception:
        return "{}"  # ultimate deterministic fail-safe (estimate_tokens("{}") == 1)


def entity_tokens(raw: dict[str, Any]) -> int:
    """Deterministic, fail-safe token cost of including one entity in a bundle. NEVER raises —
    malformed content yields a deterministic estimate, so a bad entity cannot crash
    ContextService."""
    return estimate_tokens(_canonical(raw))
