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


class ExecutionStatus(str, enum.Enum):
    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED: dict[ExecutionStatus, set[ExecutionStatus]] = {
    ExecutionStatus.REQUESTED: {ExecutionStatus.APPROVED, ExecutionStatus.REJECTED, ExecutionStatus.CANCELLED},
    ExecutionStatus.APPROVED: {ExecutionStatus.EXECUTING, ExecutionStatus.CANCELLED},
    ExecutionStatus.EXECUTING: {ExecutionStatus.EXECUTED, ExecutionStatus.FAILED},
    ExecutionStatus.FAILED: {ExecutionStatus.EXECUTING, ExecutionStatus.CANCELLED},
    ExecutionStatus.REJECTED: set(),
    ExecutionStatus.EXECUTED: set(),
    ExecutionStatus.CANCELLED: set(),
}

TERMINAL = {ExecutionStatus.REJECTED, ExecutionStatus.EXECUTED, ExecutionStatus.CANCELLED}


class InvalidTransition(Exception):
    def __init__(self, frm: ExecutionStatus, to: ExecutionStatus) -> None:
        super().__init__(f"invalid execution transition {frm.value} -> {to.value}")
        self.frm = frm
        self.to = to


def can_transition(frm: ExecutionStatus, to: ExecutionStatus) -> bool:
    return to in _ALLOWED.get(frm, set())


def assert_transition(frm: ExecutionStatus, to: ExecutionStatus) -> None:
    if not can_transition(frm, to):
        raise InvalidTransition(frm, to)


def is_terminal(status: ExecutionStatus) -> bool:
    return status in TERMINAL
