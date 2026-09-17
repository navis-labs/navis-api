"""Provider interface. ALL external-side-effect logic lives behind this.

No service calls a Google/Slack SDK directly — services call a Provider. A provider
does three things:
    validate(payload)  -> raise if the command is malformed (cheap, pre-execute)
    execute(payload)   -> perform the side effect, return a ProviderResult
    status(reference)  -> later, poll the external object's status

SAFETY: providers default to DRY-RUN (no real send) and return a synthetic
ProviderReference flagged `dry_run=True`. Wiring a real sender (with write OAuth
scope) is a single injection point per provider — accidental sends are impossible
until that is done deliberately.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from navis.execution.contract import ProviderResult


class ProviderError(Exception):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ProviderValidationError(ProviderError):
    """Payload is invalid — never retryable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class Provider(ABC):
    name: str
    action_type: str

    # Set True by a provider whose action CANNOT be performed in any deployment
    # because no backend exists for it, and which therefore refuses at
    # validate() rather than simulating (see providers/tasks.py).
    #
    # Such a provider is SAFE in the production registry: it fails loudly and
    # non-retryably instead of reporting a side effect that never happened. The
    # distinction matters because `production_providers()` withholds every
    # provider that would silently simulate, and without this flag it could not
    # tell "deliberately unsupported, fails honestly" from "accidentally
    # unwired, reports a fake success".
    unsupported: bool = False

    @property
    def is_live(self) -> bool:
        """True when a real sender is injected — i.e. `execute()` will reach the
        external system rather than returning a synthetic dry-run reference.

        Reads the `_sender` attribute every concrete provider already stores, so
        a new connector inherits the answer for free. A provider that resolves
        its client some other way should override this; returning True while
        being unable to act is the one thing that reintroduces the false-success
        bug this exists to prevent.
        """
        return getattr(self, "_sender", None) is not None

    @abstractmethod
    async def validate(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, payload: dict[str, Any]) -> ProviderResult:
        raise NotImplementedError

    @abstractmethod
    async def status(self, reference: dict[str, Any]) -> str:
        raise NotImplementedError
