"""Observation webhook HTTP surface. External providers POST outcome notifications;
we parse + record (idempotent). Verified by an optional shared secret per provider.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import Depends

from navis.observation.service import ObservationService
from navis.observation.webhooks.calendar import parse_calendar
from navis.observation.webhooks.gmail import parse_gmail
from navis.platform.database import get_session

router = APIRouter(prefix="/webhooks/observation", tags=["observation"])

_PARSERS = {"gmail": parse_gmail, "calendar": parse_calendar}


def _verify(provider: str, token: str | None) -> None:
    expected = os.environ.get(f"OBSERVATION_WEBHOOK_TOKEN_{provider.upper()}")
    if expected and token != expected:
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
