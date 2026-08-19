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

from navis.observation.service import ObservationService
from navis.observation.webhooks.calendar import parse_calendar
from navis.observation.webhooks.gmail import parse_gmail
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger

logger = get_logger("observation.webhooks")

router = APIRouter(prefix="/webhooks/observation", tags=["observation"])

_PARSERS = {"gmail": parse_gmail, "calendar": parse_calendar}


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

    outcome = await ObservationService(db).record_from_webhook(webhook)
    return {
        "status": "recorded" if outcome else "unmatched",
        "outcome_id": str(outcome.outcome_id) if outcome else None,
    }
