"""Provider registry — action_type -> Provider. The only place providers are wired.

Two registries are exposed:

  * `_DEFAULT_PROVIDERS` — dry-run defaults. Every provider's sender is None so
    validation runs but no external API is hit. This is what tests + local dev
    see when `execute_pass` is called without an explicit `providers=` arg.

  * `production_providers()` — real Google-side senders wired in. The Celery
    worker entry point (`navis.execution.worker._execute`) passes THIS registry
    to `execute_pass`, so production dispatches real Calendar events + Gmail
    messages while unit tests stay hermetic.

Keeping the two separate is the seam that lets us go live per-provider (Slack /
Tasks stay dry-run today, but drop in their real senders here without touching
the worker code or the test suite).
"""
from navis.execution.providers.base import Provider, ProviderError, ProviderValidationError
from navis.execution.providers.calendar import CalendarProvider
from navis.execution.providers.gmail import GmailProvider
from navis.execution.providers.slack import SlackProvider
from navis.execution.providers.tasks import TasksProvider

# Dry-run-safe defaults. Never invoke real Google APIs — tests + local dev
# depend on this staying inert. Inject real senders via `production_providers()`
# below to go live.
_DEFAULT_PROVIDERS: dict[str, Provider] = {
    GmailProvider.action_type: GmailProvider(),
    CalendarProvider.action_type: CalendarProvider(),
    TasksProvider.action_type: TasksProvider(),
    SlackProvider.action_type: SlackProvider(),
}


def get_provider(action_type: str, registry: dict[str, Provider] | None = None) -> Provider | None:
    return (registry or _DEFAULT_PROVIDERS).get(action_type)


def production_providers() -> dict[str, Provider]:
    """Registry with real senders wired in. Used by the Celery worker only.

    Senders are imported lazily so `_DEFAULT_PROVIDERS` (loaded at import time
    by tests) stays free of the DB / signals dependencies that live inside the
    sender module. Slack + Tasks stay dry-run until their real senders exist.
    """
    from navis.execution.providers._senders import calendar_sender, gmail_sender

    return {
        GmailProvider.action_type: GmailProvider(sender=gmail_sender),
        CalendarProvider.action_type: CalendarProvider(sender=calendar_sender),
        TasksProvider.action_type: TasksProvider(),
        SlackProvider.action_type: SlackProvider(),
    }


__all__ = [
    "Provider",
    "ProviderError",
    "ProviderValidationError",
    "GmailProvider",
    "CalendarProvider",
    "TasksProvider",
    "SlackProvider",
    "get_provider",
    "production_providers",
]
