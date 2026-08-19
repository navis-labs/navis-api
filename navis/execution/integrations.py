"""The generic integration / credential boundary.

WHAT THE GENERIC EXECUTION LAYER IS ALLOWED TO KNOW
---------------------------------------------------
    organization + provider  ->  "is there a usable credential, or not?"

That is all. It does not know what OAuth is, that Google refreshes tokens, that
Gmail scopes differ from Calendar scopes, or that Slack uses a bot token. Each
provider's own module owns its authentication and registers a resolver here.

THE SHAPE THIS EXISTS TO PREVENT
--------------------------------
    if provider == "gmail":   resolve_google_token()
    elif provider == "slack": resolve_slack_token()
    elif provider == "notion": resolve_notion_token()

...appearing in the worker, the router, or the action registry. Before this
module, two GENERIC files already imported `navis.signals.google.resolve`
directly — `execution/router.py` (the draft sign-off) and `execution/actions.py`
(the approve pre-flight). Each new connector would have added a branch to both.

AUTHENTICATION IS NOT AUTHORIZATION
-----------------------------------
Resolving a credential answers "can Navis physically act as this org on this
system". It says NOTHING about whether the org is permitted to. The entitlement
gate (organization lifecycle -> EXECUTION -> the ActionSpec's connector
capability -> kill switch) is a separate question, asked FIRST and independently,
in `worker._authorization_denial`. Possession of a token must never imply
permission, so nothing in this module consults entitlements and nothing in the
authorization path consults credentials.

ORGANIZATION SCOPE IS MANDATORY
-------------------------------
There is no resolver signature that omits `org_id`, and `resolve_integration`
refuses a falsy one before dispatching. There is no global credential, no
process-level token cache, no developer fallback and no "most recent
integration" path — a credential is always looked up under exactly one
organization or not at all.
"""
from __future__ import annotations

import uuid
from importlib import import_module
from typing import Any, Protocol

from navis.platform.monitoring import get_logger

logger = get_logger("execution.integrations")


class IntegrationUnavailable(Exception):
    """No usable credential for (organization, provider).

    PERMANENT by contract. A missing, revoked, unscoped or token-less
    integration cannot be fixed by trying again in thirty seconds — it needs a
    human to reconnect the account. The worker therefore treats this like a
    validation error: fail once, dead-letter with this message, do not burn the
    retry budget rediscovering that Google is still not connected.

    Deliberately distinct from a provider BUSINESS failure (Gmail rejected the
    recipient) and from an AUTHORIZATION denial (the org is no longer entitled).
    Three different causes, three different things a human must do about them.

    `reconsent_required` separates "connect this account" from "reconnect it to
    grant more" — the two produce different instructions in the UI, and the
    existing Google resolvers already draw exactly that distinction.

    Carries NO credential material: the message names the provider and what the
    user must do, never a token, header or encrypted blob. It reaches the dead
    letter, the outbox and Observation, so it must stay safe to print.
    """

    def __init__(
        self, message: str, *, provider: str, reconsent_required: bool = False
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.reconsent_required = reconsent_required


class IntegrationResolver(Protocol):
    """A provider's own credential lookup.

    Returns an opaque handle — whatever that provider's sender needs (for Google
    it is the `Integration` row). The generic layer NEVER inspects it; only the
    module that registered the resolver knows its shape. Raises
    `IntegrationUnavailable` when the org has no usable credential.

    Takes the CALLER'S session so resolution joins the caller's transaction
    instead of opening its own.
    """

    async def __call__(self, db: Any, org_id: uuid.UUID) -> Any: ...


# provider name (ActionSpec.provider) -> "module.path:function".
#
# Stored as a string and imported on demand so this module stays free of the
# connector stacks. Importing the Google adapter eagerly would drag the whole
# Signals dependency chain into every process that merely wants to ask whether a
# credential exists.
_RESOLVERS: dict[str, str] = {
    "gmail": "navis.execution.providers._senders:resolve_gmail_credential",
    "calendar": "navis.execution.providers._senders:resolve_calendar_credential",
    "slack": "navis.execution.providers._senders:resolve_slack_credential",
}


def register_integration_resolver(
    provider: str, target: str | IntegrationResolver, *, replace: bool = False
) -> None:
    """Register a provider's credential resolver. The connector extension point.

    `target` is either a callable or a "module.path:function" string (preferred —
    it keeps the import lazy). Duplicate registration raises unless `replace` is
    explicit, so two connectors cannot silently claim one provider and hand the
    wrong org's credential shape to a sender.
    """
    if not replace and provider in _RESOLVERS:
        raise ValueError(
            f"provider {provider!r} already has a credential resolver; "
            "one provider has one credential authority"
        )
    _RESOLVERS[provider] = target  # type: ignore[assignment]


def has_integration_resolver(provider: str) -> bool:
    return provider in _RESOLVERS


def _load(provider: str) -> IntegrationResolver:
    target = _RESOLVERS[provider]
    if callable(target):
        return target
    module_path, _, attr = target.partition(":")
    return getattr(import_module(module_path), attr)


async def resolve_integration(db: Any, org_id: uuid.UUID, provider: str) -> Any:
    """The org's credential for this provider, or raise.

    THE only way generic Execution obtains a credential. Every caller — the
    approve pre-flight, the draft sign-off, the production senders — goes through
    here, so organization scoping is enforced in one place rather than re-argued
    at each call site.
    """
    if not org_id:
        # Fail closed. A resolver that ran without an organization would be free
        # to pick "some" integration, which is precisely the cross-tenant leak
        # this boundary exists to make impossible.
        raise IntegrationUnavailable(
            "credential resolution requires an organization", provider=provider
        )
    if provider not in _RESOLVERS:
        # An action whose provider has no registered credential authority cannot
        # act. Silence here would mean a sender improvising its own lookup.
        raise IntegrationUnavailable(
            f"no credential resolver is registered for provider {provider!r}; "
            "this connector cannot perform write actions in this deployment",
            provider=provider,
        )
    return await _load(provider)(db, org_id)


async def integration_preflight(db: Any, org_id: uuid.UUID, provider: str) -> None:
    """Raise `IntegrationUnavailable` unless the org can act on this provider.

    The read-only form of `resolve_integration`, used before a human approves so
    they are told to reconnect BEFORE reviewing and freezing a draft, rather than
    discovering it from a dead letter minutes later.
    """
    await resolve_integration(db, org_id, provider)
