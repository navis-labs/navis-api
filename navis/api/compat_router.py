"""Compatibility stubs for frontend routes not yet wired to real domains.

The frontend polls a set of routes that don't correspond to shipped backend endpoints
yet (e.g. `/decide/stack`, `/skills`, `/context`, `/execution/slack/blockers`,
`/workspace/meetings/{id}/brief`). Without stubs those return 404 and, while the
frontend's `fetchJsonOr` treats a 404 as an empty result, the server log noise and
the one call that uses `fetchJson` (`/workspace/meetings/{id}/brief`) both benefit
from a clean 200-with-empty-payload response.

Each handler here is intentionally trivial — no auth gate, no persistence, no
capability check. When a real implementation ships in its domain router, delete
the matching stub here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["compat"])


@router.get("/decide/stack")
async def decide_stack() -> dict[str, Any]:
    return {"decisions": []}


@router.get("/skills")
async def skills() -> dict[str, Any]:
    return {"skills": []}


@router.get("/signals")
async def signals() -> dict[str, Any]:
    return {"signals": []}


@router.get("/execution/slack/blockers")
async def slack_blockers() -> dict[str, Any]:
    return {"blockers": []}


@router.get("/workspace/meetings/{event_id}/brief")
async def meeting_brief(event_id: str) -> dict[str, Any]:
    return {
        "meeting": {"id": event_id, "title": ""},
        "stakeholders": [],
        "relatedEmails": [],
        "relatedDocs": [],
        "priorDecisions": [],
        "angle": None,
        "asOf": datetime.now(timezone.utc).isoformat(),
    }
