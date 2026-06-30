"""Deterministic token estimation for budgeting.

No tokenizer dependency and no model calls — a stable word-based estimate so the
same entity always costs the same number of tokens across replays and regions.
"""
from __future__ import annotations

import json
import re
from typing import Any

_WORD = re.compile(r"\S+")
# ~0.75 words per token (conservative). Kept local to Context — never imports ml/rag.
_WORDS_PER_TOKEN = 0.75


def estimate_tokens(text: str) -> int:
    return max(1, int(len(_WORD.findall(text)) / _WORDS_PER_TOKEN))


def _canonical(raw: dict[str, Any]) -> str:
    """Stable serialization of an entity's reasoning-relevant content (fields + lists +
    links). Sorted keys → identical bytes across replays, so token cost is deterministic."""
    payload = {
        "fields": raw.get("fields", {}),
        "lists": raw.get("lists", {}),
        "links": raw.get("links", {}),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def entity_tokens(raw: dict[str, Any]) -> int:
    """Deterministic token cost of including one entity in a bundle."""
    return estimate_tokens(_canonical(raw))
