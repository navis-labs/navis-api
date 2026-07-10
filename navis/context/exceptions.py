"""Typed failure surface for the Context layer.

Every error this layer RAISES is a subclass of ``ContextError`` carrying a stable, machine-
readable ``code`` — so an orchestrator can catch the whole layer with one type AND classify
the cause without parsing a free-text message (the message wording is allowed to drift; the
code is a frozen contract).

Scope of this surface: these are the layer's OWN raised errors (a bad request / a
misconfiguration). They are surfaced IN-PROCESS to the caller of ``ContextService.build`` —
they are never serialized across a wire, so ``str(exc)`` plus ``.code`` is the whole contract;
there is deliberately no ``to_dict``/payload framework (it would be dead code with no consumer).

Determinism / replay: an exception holds only the immutable facts the raise site already had
(an enum, a string) — no clock, no uuid, no randomness, no I/O — so raising is pure and a
replay produces a byte-identical error.

NOT covered here: infrastructure faults from the cache backend. ``repository.get`` degrades a
corrupt/stale row to a rebuild (returns ``None``); ``repository.put`` currently lets the
driver's exception (e.g. SQLAlchemy) propagate. Wrapping write failures in a Context-typed
error is a repository.py change (out of this module's scope) — see that module's notes.
"""
from __future__ import annotations

from typing import ClassVar


class ContextError(Exception):
    """Base for every error the Context layer raises.

    ``code`` is a stable identifier for programmatic classification across boundaries; every
    subclass overrides it. The base keeps the builtin ``Exception`` constructor, so subclasses
    that only need a message work unchanged.
    """

    code: ClassVar[str] = "context.error"


class UnknownContextTask(ContextError):
    """Requested a ContextTask with no registered selection spec.

    This is a programming/registry error (the task registry is asserted complete at import),
    kept as a defensive guard. The offending task is carried as a structured ``.task``
    attribute — not only inside the message — so a caller/log can read it without string-
    parsing. (The task registry completeness check normally makes this unreachable.)
    """

    code: ClassVar[str] = "context.unknown_task"

    def __init__(self, task: object) -> None:
        self.task = task
        super().__init__(f"no selection spec for context task {task!r}")


class PermissionConfigurationError(ContextError):
    """A PermissionScope / role policy is internally inconsistent (e.g. ``allow_all`` combined
    with type/predicate constraints, or a role granting an unknown entity type). A fail-closed
    configuration error, not an authorization denial."""

    code: ClassVar[str] = "context.permission_misconfigured"
