"""Prompt regression: a stable fingerprint of a prompt revision + its rendered output.

Pin a fingerprint in a test; if a prompt or its rendering changes, the fingerprint
changes and the regression test fails loudly — no silent prompt drift.
"""
from __future__ import annotations

import hashlib


def prompt_fingerprint(prompt_id: str, version: int, system: str, rendered_user: str) -> str:
    """Stable, INJECTIVE fingerprint of a prompt revision + its rendered output.

    Each part is length-prefixed (netstring `<len>:<part>`) before hashing, so no delimiter
    inside a part can ever alias two different revisions to the same fingerprint. A plain
    `|`-join could: ("b|c","d") and ("b","c|d") collide — and a collision would let a real
    prompt change slip past drift detection, the one thing this function exists to catch.
    sha256 is stable across Python versions/platforms (unlike the salted builtin hash)."""
    parts = (prompt_id, str(version), system, rendered_user)
    canonical = "".join(f"{len(p)}:{p}" for p in parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
