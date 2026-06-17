"""Command handler contract. Each ActionType maps to one handler.

Handlers are where the real side effects live (calling Gmail to send, creating a
calendar event...). Phase 1 ships safe no-op handlers so the approval flow and
observation loop can be exercised before wiring real providers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ActionHandler(ABC):
    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Perform the side effect; return a result dict stored on the Action."""
        raise NotImplementedError


class NoopHandler(ActionHandler):
    """Default handler: records intent without touching the outside world."""

    def __init__(self, label: str) -> None:
        self.label = label

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"executed": "noop", "handler": self.label, "echo": params}
