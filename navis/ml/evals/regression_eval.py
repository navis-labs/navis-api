"""Prompt regression: a stable fingerprint of a prompt revision + its rendered output.

Pin a fingerprint in a test; if a prompt or its rendering changes, the fingerprint
changes and the regression test fails loudly — no silent prompt drift.
"""
from __future__ import annotations

import hashlib


def prompt_fingerprint(prompt_id: str, version: int, system: str, rendered_user: str) -> str:
    h = hashlib.sha256(f"{prompt_id}|{version}|{system}|{rendered_user}".encode())
    return h.hexdigest()[:16]
