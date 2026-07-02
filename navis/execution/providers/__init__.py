"""Provider registry — action_type -> Provider. The only place providers are wired."""
from collections.abc import Mapping
from types import MappingProxyType

from navis.execution.models import ActionType
from navis.execution.providers.base import Provider, ProviderError, ProviderValidationError
from navis.execution.providers.calendar import CalendarProvider
from navis.execution.providers.gmail import GmailProvider
from navis.execution.providers.slack import SlackProvider
from navis.execution.providers.tasks import TasksProvider

# Dry-run-safe defaults. Inject real senders here to go live (deliberate, per provider).
# Read-only Mapping (MappingProxyType) — the default registry can never be mutated at
# runtime, so provider dispatch stays deterministic (same guarantee the prompt registry makes).
_DEFAULT_PROVIDERS: Mapping[str, Provider] = MappingProxyType({
    GmailProvider.action_type: GmailProvider(),
    CalendarProvider.action_type: CalendarProvider(),
    TasksProvider.action_type: TasksProvider(),
    SlackProvider.action_type: SlackProvider(),
})

# Fail FAST at import: a declared ActionType with no registered provider is a misdeploy.
# Without this guard it surfaces only at claim time — every request of that type fails
# closed into the DLQ. Boot must refuse the deploy instead.
_MISSING = {t.value for t in ActionType} - set(_DEFAULT_PROVIDERS)
if _MISSING:  # pragma: no cover — import-time deploy guard
    raise RuntimeError(f"ActionType(s) without a registered execution provider: {sorted(_MISSING)}")


def get_provider(action_type: str, registry: Mapping[str, Provider] | None = None) -> Provider | None:
    # `is None`, never `or`: an explicitly injected EMPTY registry must mean "no
    # providers" (every lookup misses -> requests fail closed), not silently fall back
    # to the LIVE defaults and dispatch real providers the caller never wired.
    lookup = _DEFAULT_PROVIDERS if registry is None else registry
    return lookup.get(action_type)


__all__ = [
    "Provider",
    "ProviderError",
    "ProviderValidationError",
    "GmailProvider",
    "CalendarProvider",
    "TasksProvider",
    "SlackProvider",
    "get_provider",
]
