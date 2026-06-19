"""Execution — Layer 6. The ONLY layer that creates external side effects.

Execution consumes Decision (via the proposer) and produces ExecutionRequest +
ExecutionEvent (outbox). It never calculates risk/priority/confidence and never
reads Memory/State/Signals. Every action is human-approved, idempotent, fully
traceable to its Decision -> ReasoningSignals -> State watermark, and executed
out-of-band by a worker behind a provider abstraction.

    Decision -> request(REQUESTED) -> [human approve] -> APPROVED
        -> worker: EXECUTING -> provider -> EXECUTED/FAILED  (+ outbox events)
"""
from navis.execution.models import ActionType, ExecutionOutbox, ExecutionRequest
from navis.execution.service import ExecutionService
from navis.execution.status import ExecutionStatus

__all__ = [
    "ActionType",
    "ExecutionRequest",
    "ExecutionOutbox",
    "ExecutionService",
    "ExecutionStatus",
]
