"""Deterministic idempotency for execution requests.

Same (org_id, decision_id, action_type, payload) -> same idempotency_key -> same
execution_id (UUIDv5). Combined with a DB unique constraint and an
INSERT ... ON CONFLICT guard, this gives at-least-once delivery + idempotent
execution: retried API calls, double approvals, and worker re-runs all converge
to ONE request and ONE side effect.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

_EXECUTION_NAMESPACE = uuid.UUID("e9c1a7b2-4d36-5e80-9a1f-2b3c4d5e6f70")


def _canonical(payload: dict[str, Any] | None) -> str:
    # Determinism constraint: payloads must be JSON-pure (str keys; no set/frozenset).
    # Both producers guarantee this (proposer builds literals; the API parses JSON), and
    # `default=str` stays deterministic for UUID/datetime — but a set would stringify in
    # PYTHONHASHSEED-dependent order, splitting one logical payload into DIFFERENT
    # idempotency keys across processes (= duplicate side effects). Mixed-type dict keys
    # fail fast in sort_keys. Keep payloads JSON-pure.
    return json.dumps(payload or {}, sort_keys=True, separators=(",", ":"), default=str)


def idempotency_key(
    org_id: uuid.UUID, decision_id: uuid.UUID, action_type: str, payload: dict[str, Any] | None
) -> str:
    raw = f"{org_id}|{decision_id}|{action_type}|{_canonical(payload)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def obligation_idempotency_key(org_id: uuid.UUID, obligation_key: str) -> str:
    """Idempotency for an action whose identity is a stable natural key.

    Deliberately takes NEITHER decision_id NOR the payload. Both are
    observation-scoped for a reply — decision_id moves with the state watermark
    and the payload carries a drifting `waiting_days` — so including either
    would reintroduce the duplicate this exists to prevent. See
    execution/obligation.py for the measured evidence.

    Disjoint from `idempotency_key` by the "obligation" tag, so an obligation
    key can never collide with a payload hash.
    """
    raw = f"{org_id}|obligation|{obligation_key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def execution_id(org_id: uuid.UUID, key: str) -> uuid.UUID:
    return uuid.uuid5(_EXECUTION_NAMESPACE, f"{org_id}|{key}")
