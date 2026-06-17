"""Signals — Layer 1. Reality capture.

Connectors (gmail, calendar, drive, ...) pull from Google Workspace and emit
exactly one thing: an immutable, version-aware, multi-tenant `SignalEvent`,
persisted to an append-only log and announced as a pointer on the event bus.

HARD BOUNDARY: signals contain NO memory, NO state, NO reasoning, NO LLM calls,
NO business logic. A connector that decides "this is a commitment" is wrong —
that is a downstream consumer's job. Signals stop at the event bus and do not
know what happens after.
"""
from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalEventType,
    SignalSource,
    SignalValidationError,
    SyncResult,
    WorkspaceConnector,
)

__all__ = [
    "SignalContext",
    "SignalEvent",
    "SignalEventType",
    "SignalSource",
    "SignalValidationError",
    "SyncResult",
    "WorkspaceConnector",
]
