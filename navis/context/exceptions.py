"""Typed failure surface for the Context layer.

Every error a caller can hit is a subclass of ContextError, so an orchestrator can
catch the whole layer with one type while still distinguishing causes.
"""
from __future__ import annotations


class ContextError(Exception):
    """Base for every Context-layer error."""


class UnknownContextTask(ContextError):
    """Requested a ContextTask with no registered selection spec."""


class ContextCacheError(ContextError):
    """The bundle cache could not be read/written (never silently swallowed)."""


class PermissionConfigurationError(ContextError):
    """A PermissionScope is internally inconsistent (e.g. allow_all with predicates)."""
