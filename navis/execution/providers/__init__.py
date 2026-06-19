"""Provider registry — action_type -> Provider. The only place providers are wired."""
from navis.execution.providers.base import Provider, ProviderError, ProviderValidationError
from navis.execution.providers.calendar import CalendarProvider
from navis.execution.providers.gmail import GmailProvider
from navis.execution.providers.slack import SlackProvider
from navis.execution.providers.tasks import TasksProvider

# Dry-run-safe defaults. Inject real senders here to go live (deliberate, per provider).
_DEFAULT_PROVIDERS: dict[str, Provider] = {
    GmailProvider.action_type: GmailProvider(),
    CalendarProvider.action_type: CalendarProvider(),
    TasksProvider.action_type: TasksProvider(),
    SlackProvider.action_type: SlackProvider(),
}


def get_provider(action_type: str, registry: dict[str, Provider] | None = None) -> Provider | None:
    return (registry or _DEFAULT_PROVIDERS).get(action_type)


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
