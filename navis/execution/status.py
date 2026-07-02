"""Execution status machine — enforce valid transitions, forbid invalid jumps.

    REQUESTED ──▶ APPROVED ──▶ EXECUTING ──▶ EXECUTED   (terminal)
        │            │              └──────▶ FAILED ──▶ EXECUTING (retry)
        │            └──▶ CANCELLED (terminal)            └──▶ CANCELLED
        ├──▶ REJECTED (terminal)
        └──▶ CANCELLED (terminal)

A request can only move along a declared edge; anything else raises
InvalidTransition. This is the safety rail that makes double-approval and
out-of-order worker callbacks no-ops instead of double side effects.
"""
from __future__ import annotations

import enum
from collections.abc import Mapping
from types import MappingProxyType


class ExecutionStatus(str, enum.Enum):
    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# The transition LAW is immutable (read-only mapping of frozen edge sets): nothing can
# widen or narrow it at runtime — the same guarantee the provider registry makes.
_ALLOWED: Mapping[ExecutionStatus, frozenset[ExecutionStatus]] = MappingProxyType({
    ExecutionStatus.REQUESTED: frozenset({ExecutionStatus.APPROVED, ExecutionStatus.REJECTED, ExecutionStatus.CANCELLED}),
    ExecutionStatus.APPROVED: frozenset({ExecutionStatus.EXECUTING, ExecutionStatus.CANCELLED}),
    ExecutionStatus.EXECUTING: frozenset({ExecutionStatus.EXECUTED, ExecutionStatus.FAILED}),
    ExecutionStatus.FAILED: frozenset({ExecutionStatus.EXECUTING, ExecutionStatus.CANCELLED}),
    ExecutionStatus.REJECTED: frozenset(),
    ExecutionStatus.EXECUTED: frozenset(),
    ExecutionStatus.CANCELLED: frozenset(),
})

# Fail FAST at import: a new ExecutionStatus member with no declared edges would otherwise
# surface as every-transition-denied at runtime. The edge map must be total over the enum.
_UNDECLARED = set(ExecutionStatus) - set(_ALLOWED)
if _UNDECLARED:  # pragma: no cover — import-time deploy guard
    raise RuntimeError(f"ExecutionStatus member(s) without declared edges: {sorted(s.value for s in _UNDECLARED)}")

TERMINAL = frozenset({ExecutionStatus.REJECTED, ExecutionStatus.EXECUTED, ExecutionStatus.CANCELLED})


class InvalidTransition(Exception):
    def __init__(self, frm: ExecutionStatus, to: ExecutionStatus) -> None:
        super().__init__(f"invalid execution transition {frm.value} -> {to.value}")
        self.frm = frm
        self.to = to


def can_transition(frm: ExecutionStatus, to: ExecutionStatus) -> bool:
    return to in _ALLOWED.get(frm, frozenset())  # unknown frm -> fail closed


def assert_transition(frm: ExecutionStatus, to: ExecutionStatus) -> None:
    if not can_transition(frm, to):
        raise InvalidTransition(frm, to)


def is_terminal(status: ExecutionStatus) -> bool:
    return status in TERMINAL
