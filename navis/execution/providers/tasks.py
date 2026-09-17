"""Task provider (create_task) — EXPLICITLY UNSUPPORTED.

There is no task system behind this action type. There never was.

The previous implementation hashed the title into a synthetic id and returned
`ok=True`, which the worker recorded as EXECUTED. So every delegation decision
told the founder their work had been filed, with an id, in a system that does
not exist. That is worse than an error: an error gets fixed, a confident lie
gets trusted and built on.

Until a real task backend is wired in, execution FAILS LOUDLY. The action type
and provider are retained (rather than deleted) so that:

  * historical `create_task` rows still resolve to a provider and can be
    explained, rather than dead-ending on "no provider for create_task";
  * the seam for a real integration stays exactly where it was.

`ProviderValidationError` is deliberate: it is non-retryable by contract, so an
unsupported action fails ONCE with a clear reason and dead-letters, instead of
burning the retry budget rediscovering that tasks still don't exist.

To make this real: implement a sender against the chosen backend (Google Tasks,
Linear, Asana), inject it here the way `CalendarProvider`/`GmailProvider` take
theirs, and delete `_UNSUPPORTED`.
"""
from __future__ import annotations

from typing import Any

from navis.execution.contract import ProviderResult
from navis.execution.providers.base import Provider, ProviderValidationError

_UNSUPPORTED = (
    "create_task is not supported: Navis has no task system connected, so this "
    "action cannot be completed. Nothing was created."
)


class TasksProvider(Provider):
    name = "tasks"
    action_type = "create_task"
    # Deliberately unsupported, and honest about it: validate() below refuses
    # every payload, so this provider can never report work it did not do. That
    # is what makes it safe to keep in the production registry (which withholds
    # providers that would silently simulate) and is why historical create_task
    # rows still resolve to a provider that can explain itself.
    unsupported = True

    async def validate(self, payload: dict[str, Any]) -> None:
        # Rejected at VALIDATE, not execute: validation runs before any
        # side-effect path and is the non-retryable gate, so an unsupported
        # action can never be dispatched, retried, or reported as done.
        raise ProviderValidationError(_UNSUPPORTED)

    async def execute(self, payload: dict[str, Any]) -> ProviderResult:  # pragma: no cover
        # Unreachable via the worker (validate raises first). Kept honest rather
        # than absent so a direct caller cannot get a fake success either.
        raise ProviderValidationError(_UNSUPPORTED)

    async def status(self, reference: dict[str, Any]) -> str:
        return "unsupported"
