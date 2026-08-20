"""Observation webhook HTTP surface. External providers POST outcome notifications;
we parse + record (idempotent). Verified by an optional shared secret per provider.
"""
from __future__ import annotations

import os
import secrets
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import Depends

from navis.identity.models import IntegrationProvider
from navis.identity.org_service import AmbiguityReason, resolve_owning_integration
from navis.observation.service import ObservationService
from navis.observation.webhooks.calendar import parse_calendar
from navis.observation.webhooks.gmail import parse_gmail
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("observation.webhooks")

router = APIRouter(prefix="/webhooks/observation", tags=["observation"])

_PARSERS = {"gmail": parse_gmail, "calendar": parse_calendar}

# Observation's provider vocabulary -> the Integration row that owns the account.
# Resolution happens HERE, at the edge, for the same reason `get_principal` resolves
# the tenant at the edge for HTTP: the domain service should receive an organization
# it can trust, not the raw material to guess one from. Keeping the map here also
# keeps `navis/observation/**` free of any Identity import — it stayed a pure
# bounded context before this change and it still is.
_PROVIDER_INTEGRATION: dict[str, IntegrationProvider] = {
    "gmail": IntegrationProvider.GMAIL,
    "calendar": IntegrationProvider.GOOGLE_CALENDAR,
}


def _verify(provider: str, token: str | None) -> None:
    """Refuse unless the caller presents this provider's configured secret.

    FAILS CLOSED. This previously read `if expected and token != expected`, so an
    unset `OBSERVATION_WEBHOOK_TOKEN_<PROVIDER>` disabled the check entirely —
    and the variable is not set in this deployment, which made a mounted,
    internet-reachable endpoint accept anonymous POSTs.

    What that allowed is not a cross-tenant leak (the organization is inherited
    from the matched prior outcome, never from the request), but it is worse than
    it looks: an anonymous caller who knows or guesses a provider reference could
    append `reply_received` / `meeting_accepted` outcomes to a real
    organization's append-only log. Outcomes are immutable and feed Learning, so
    fabricated evidence could not be retracted, only out-voted.

    An absent secret now refuses every request rather than accepting every
    request — the safe direction for an endpoint whose entire job is to write
    permanent facts about someone else's business.
    """
    expected = os.environ.get(f"OBSERVATION_WEBHOOK_TOKEN_{provider.upper()}")
    if not expected:
        logger.error(
            "observation webhook for %s is not configured "
            "(OBSERVATION_WEBHOOK_TOKEN_%s unset); refusing the request",
            provider, provider.upper(),
        )
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"observation webhooks for {provider} are not configured",
        )
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad webhook token")


@router.post("/{provider}")
async def receive(
    provider: str,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    parser = _PARSERS.get(provider)
    if parser is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown provider {provider}")
    _verify(provider, request.query_params.get("token"))

    body = await request.json()
    webhook = parser(body or {})
    if webhook is None:
        return {"status": "ignored"}  # ack so the provider stops retrying

    # ── TENANT RESOLUTION — FROM A PERSISTED INTEGRATION, NEVER FROM THE BODY ──
    #
    # The shared secret verified above proves the CALLER is authorized to deliver
    # notifications. It says nothing about WHICH organization this one concerns,
    # because there is one secret per provider for the whole deployment. So the
    # organization is resolved the same way every other inbound provider event
    # resolves it: the notification names an external account, and that account is
    # looked up against `integrations` to find its owner.
    #
    # Every outcome is fail-closed. An unnamed account, an account we have no
    # active integration for, or an account two organizations have both connected
    # all ACK (200, `status: unresolved`) WITHOUT recording anything.
    #
    # ACK rather than 4xx because a provider retries a 4xx into the same refusal
    # forever, and the delivery itself was well-formed and authenticated — what is
    # missing is a tenant, which retrying cannot supply. The `reason` in the body
    # and the bounded metric are how an operator sees it instead.
    #
    # Recording nothing because `outcomes` is append-only and feeds Learning: a
    # fact written under the wrong tenant cannot be retracted, only out-voted.
    integration_provider = _PROVIDER_INTEGRATION.get(provider)
    if integration_provider is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown provider {provider}")

    integration, reason = await resolve_owning_integration(
        db,
        provider=integration_provider,
        external_account=webhook.external_account,
    )
    if integration is None:
        # Identifier-only. `external_account` is an address the caller supplied and
        # is NOT logged, so a probe cannot use the log as an oracle; the bounded
        # reason label is what an operator alerts on.
        logger.warning(
            "observation webhook %s refused: tenant unresolved (%s)", provider, reason
        )
        metrics.incr(
            "observation.webhook.unresolved",
            labels={"provider": provider, "reason": reason or AmbiguityReason.NO_MATCH},
        )
        return {"status": "unresolved", "reason": reason, "outcome_id": None}

    outcome = await ObservationService(db).record_from_webhook(
        webhook, org_id=integration.organization_id
    )
    return {
        "status": "recorded" if outcome else "unmatched",
        "outcome_id": str(outcome.outcome_id) if outcome else None,
    }
