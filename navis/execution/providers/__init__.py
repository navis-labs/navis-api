"""Provider registry — action_type -> Provider. The only place providers are wired.

Two registries are exposed:

  * `_DEFAULT_PROVIDERS` — dry-run defaults. Every provider's sender is None so
    validation runs but no external API is hit. This is what tests + local dev
    see when `execute_pass` is called without an explicit `providers=` arg.

  * `production_providers()` — real senders wired in. The Celery worker entry
    point (`navis.execution.worker._execute`) passes THIS registry to
    `execute_pass`, so production dispatches real Calendar events, Gmail messages
    and Slack posts while unit tests stay hermetic.

Keeping the two separate is the seam that lets us go live per-provider. Slack was
withheld here until it had a real sender and now ships live; `create_task` stays
`unsupported` (it refuses at validate() — there is no task backend), so a future
connector goes live by dropping in its sender, with no worker or test change.
"""
from navis.execution.actions import all_specs, get_spec
from navis.execution.providers.base import Provider, ProviderError, ProviderValidationError
from navis.execution.providers.calendar import (
    CalendarProvider,
    CancelMeetingProvider,
    RescheduleMeetingProvider,
)
from navis.execution.providers.gmail import GmailProvider
from navis.execution.providers.reply import ReplyToEmailProvider
from navis.execution.providers.slack import SlackProvider
from navis.execution.providers.tasks import TasksProvider
from navis.platform.monitoring import get_logger

logger = get_logger("execution.providers")

# Dry-run-safe defaults. Never invoke real Google APIs — tests + local dev
# depend on this staying inert. Inject real senders via `production_providers()`
# below to go live.
_DEFAULT_PROVIDERS: dict[str, Provider] = {
    GmailProvider.action_type: GmailProvider(),
    ReplyToEmailProvider.action_type: ReplyToEmailProvider(),
    CalendarProvider.action_type: CalendarProvider(),
    RescheduleMeetingProvider.action_type: RescheduleMeetingProvider(),
    CancelMeetingProvider.action_type: CancelMeetingProvider(),
    TasksProvider.action_type: TasksProvider(),
    SlackProvider.action_type: SlackProvider(),
}


def _assert_providers_match_specs(registry: dict[str, Provider]) -> None:
    """One action, one provider, one spec — checked at import.

    The enum, the spec registry and this wiring table were three hand-maintained
    lists. Nothing made them agree, so the failure modes were:

      * spec with no provider  -> approved, then dead-lettered on "no provider";
      * provider with no spec  -> unreachable (get_provider refuses it), i.e. a
        connector that looks wired and silently never runs;
      * provider.name != spec.provider -> the rate limiter throttles a key that
        no action uses, and the ProviderReference names a provider the registry
        does not know.

    Raising at import turns all three into a failed startup instead of a
    production incident discovered from the dead-letter queue.
    """
    specs = all_specs()
    missing_provider = sorted(set(specs) - set(registry))
    missing_spec = sorted(set(registry) - set(specs))
    mismatched = sorted(
        f"{at}: provider.name={registry[at].name!r} spec.provider={specs[at].provider!r}"
        for at in set(specs) & set(registry)
        if registry[at].name != specs[at].provider
    )
    problems = []
    if missing_provider:
        problems.append(f"specs with no provider: {missing_provider}")
    if missing_spec:
        problems.append(f"providers with no ActionSpec: {missing_spec}")
    if mismatched:
        problems.append(f"provider name disagrees with spec: {mismatched}")
    if problems:
        raise RuntimeError("execution provider registry is inconsistent — " + "; ".join(problems))


_assert_providers_match_specs(_DEFAULT_PROVIDERS)


def get_provider(action_type: str, registry: dict[str, Provider] | None = None) -> Provider | None:
    """The provider for an action, or None.

    FAIL CLOSED ON AN UNSPECCED ACTION. A provider registered for an action_type
    with no ActionSpec is unreachable by construction: the spec is what carries
    the capability the org must hold, the risk, and the approval policy, so
    dispatching without one would run a side effect that no gate ever evaluated.
    Refusing here means the worker's existing unroutable path (EXECUTING ->
    FAILED + dead letter) explains it, rather than the action quietly working.

    `registry` overrides which provider INSTANCES are used (production senders vs
    dry-run defaults vs a test double). It never overrides whether the action is
    known — that is the registry's job, not the caller's.

    `is None`, NOT truthiness. An EMPTY registry is a real and meaningful state:
    `_production_safe` returns one when every provider it was given would
    simulate. Falling back to `_DEFAULT_PROVIDERS` there would hand the worker
    the DRY-RUN providers in production and report success for side effects that
    never happened — precisely the false-success this module exists to prevent,
    reintroduced by a falsy `{}`. An empty registry must mean "nothing may run",
    which routes to the honest `no provider` failure and a dead letter.
    """
    if get_spec(action_type) is None:
        logger.error(
            "no ActionSpec for action_type=%s; refusing to dispatch a provider "
            "for an action with no capability, risk or approval metadata",
            action_type,
        )
        return None
    return (_DEFAULT_PROVIDERS if registry is None else registry).get(action_type)


def _production_safe(registry: dict[str, Provider]) -> dict[str, Provider]:
    """Withhold every provider that would SIMULATE instead of acting.

    THE BUG THIS EXISTS TO FIX
    --------------------------
    `Provider.execute()` falls back to a dry run when no sender is injected,
    returning `ok=True` with a synthetic `dry_run=True` reference. That is
    CORRECT for `_DEFAULT_PROVIDERS`: tests and local dev depend on it, and it
    is what keeps the suite from ever reaching Google.

    In the PRODUCTION registry it is a misconfiguration, and it was silent.
    `send_slack` was wired here with no sender, so an approved Slack action ran
    the dry-run branch, the worker recorded EXECUTED, `executed_at` was stamped,
    and Observation captured an ACTION_EXECUTED outcome that Learning scores as
    a success — for a message that was never sent. Reproduced end to end before
    this guard existed:

        execute_pass -> {'executed': 1, ...}   status=executed
        provider_reference={'provider':'slack','external_id':'dryrun-7e30aad5…',
                            'dry_run': True}

    This is the same class of defect as the old `create_task` synthetic id, and
    it is fixed the same way: refuse to claim work that did not happen.

    WHY HERE AND NOT IN THE WORKER
    ------------------------------
    The worker cannot tell a legitimate dry run from a broken one — the test
    suite drives `execute_pass` with dry-run providers on purpose and asserts
    EXECUTED. Only the registry knows that real side effects are expected, so
    the registry is where "would this simulate?" can be answered. Nothing about
    the status machine, idempotency, retry or the Observation envelope changes.

    A withheld action type resolves to no provider, and the worker's existing
    unroutable path takes over: EXECUTING -> FAILED with `no provider for
    <type>` plus a dead letter. Failing is the honest outcome — the action
    genuinely cannot be performed in this deployment.

    `unsupported` providers are KEPT: they refuse at validate() rather than
    simulating, so they cannot report a false success, and dropping them would
    dead-end historical rows on "no provider" instead of an explanation.
    """
    safe: dict[str, Provider] = {}
    for action_type, provider in registry.items():
        if provider.is_live or provider.unsupported:
            safe[action_type] = provider
            continue
        logger.error(
            "provider %r for action_type=%s has NO sender wired; withholding it "
            "from the production registry so it cannot report a simulated "
            "success. Executions of this action type will FAIL with 'no "
            "provider' until a real sender is injected.",
            provider.name,
            action_type,
        )
    return safe


def production_providers() -> dict[str, Provider]:
    """Registry with real senders wired in. Used by the Celery worker only.

    Senders are imported lazily so `_DEFAULT_PROVIDERS` (loaded at import time
    by tests) stays free of the DB / signals dependencies that live inside the
    sender module.

    Passed through `_production_safe`, so a provider that would dry-run never
    reaches the worker — see that function for the incident this prevents.
    Registering a new connector here without its sender is therefore a loud,
    failing action rather than a silent fake success.
    """
    from navis.execution.providers._senders import (
        calendar_cancel_sender,
        calendar_reschedule_sender,
        calendar_sender,
        gmail_sender,
        slack_sender,
    )

    return _production_safe({
        GmailProvider.action_type: GmailProvider(sender=gmail_sender),
        # Replies go out through the SAME Gmail sender — it already accepts
        # `reply_thread_id` / `in_reply_to_message_id` for threading.
        ReplyToEmailProvider.action_type: ReplyToEmailProvider(sender=gmail_sender),
        CalendarProvider.action_type: CalendarProvider(sender=calendar_sender),
        RescheduleMeetingProvider.action_type: RescheduleMeetingProvider(
            sender=calendar_reschedule_sender
        ),
        CancelMeetingProvider.action_type: CancelMeetingProvider(
            sender=calendar_cancel_sender
        ),
        # Refuses at validate(); safe here precisely because it never simulates.
        TasksProvider.action_type: TasksProvider(),
        # LIVE. `slack_sender` posts through chat.postMessage using the org's own
        # bot token, resolved through the same integration boundary Gmail and
        # Calendar use. It was withheld until this sender existed — that is the
        # seam working as designed, not a special case: `_production_safe` still
        # withholds any provider that would simulate, and this one no longer would.
        SlackProvider.action_type: SlackProvider(sender=slack_sender),
    })


__all__ = [
    "Provider",
    "ProviderError",
    "ProviderValidationError",
    "GmailProvider",
    "ReplyToEmailProvider",
    "CalendarProvider",
    "RescheduleMeetingProvider",
    "CancelMeetingProvider",
    "TasksProvider",
    "SlackProvider",
    "get_provider",
    "production_providers",
]
