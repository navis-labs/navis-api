"""Context — Layer 5. The deterministic boundary between State and Reasoning.

Context answers exactly one question: "What information does THIS reasoning task need?"
It reads ONLY the authoritative OrganizationalState, never Memory/Signals/DB/RAG, and
emits an immutable, content-addressed `ContextBundle` (which embeds a SCOPED
OrganizationalState so Reasoning consumes it unchanged). It computes NO intelligence —
no risk/priority/score/recommendation. Pure, deterministic, replayable, multi-tenant.

Public surface (everything else is internal):

    from navis.context import ContextService, ContextRequest, ContextTask, ContextBundle

    bundle = await ContextService(db).build(ContextRequest.full_org(org_id))
    signals = run_reasoning_v2(bundle.state)   # Reasoning is unchanged

`build()` raises a `ContextError` subclass on a bad request / misconfiguration, so a caller
can catch the whole layer with one type (`ContextError`) — it is part of this public surface.
"""
from __future__ import annotations

from navis.context.contracts import (
    ActorRef,
    ContextBundle,
    ContextRequest,
    PermissionScope,
    TokenBudget,
)
from navis.context.exceptions import ContextError
from navis.context.service import ContextService
from navis.context.tasks import ContextTask

__all__ = [
    "ContextService",
    "ContextRequest",
    "ContextTask",
    "ContextBundle",
    "ActorRef",
    "PermissionScope",
    "TokenBudget",
    "ContextError",
]
