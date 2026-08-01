"""Context edge — serves the intent-scoped view over the current State.

Kills the frontend's `/context` mock (Navweb/src/services/api.ts:1878). The
endpoint is a thin adapter over ``state.v2.scope.scope_for``; no business
logic lives here. Auth is capability-gated identically to /brief so the same
principals that see today's decisions see the underlying scoped state.

Dependency arrow: API → State (via StateService.incremental_update). Never
imports Reasoning or Decision — the scope utility is intentionally upstream
of both.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.authz import Principal
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import timed
from navis.state.v2 import MorningBrief, StateService, scope_for

logger = get_logger("context")

router = APIRouter(prefix="/context", tags=["context"])


@router.get("")
async def context_bundle(
    user_id: str | None = None,  # accepted for FE compat; scoping is org-wide today
    p: Principal = Depends(require_capability(Capability.DECISION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Return the founder's morning-brief scoped view over current State.

    ``user_id`` is accepted but ignored — Phase 1 is org-wide. Multi-user
    perspective scoping lives in an authorization layer over State, not here
    (see the Context sanity check).
    """
    from navis.orchestration.runtime_policy import resolve_policy

    start = time.perf_counter()
    # incremental_update (not get_state) so counts reflect the newest memory
    # rather than a stale snapshot — matches build_brief's discipline.
    with timed("context.state"):
        state = await StateService(db).incremental_update(p.org_id)
    with timed("context.policy"):
        policy = await resolve_policy(db, p.org_id)
    with timed("context.scope"):
        view = scope_for(state, MorningBrief(), policy, as_of=state.as_of)
    with timed("context.serialize"):
        body = view.to_dict()

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    # Structured audit log — replaces the deferred `context_assemblies` table.
    # Give us "why did the brief look like X on day Y" replay without a schema.
    logger.info(
        "context assembled org=%s intent=%s watermark=%s policy_version=%s "
        "entities=%d elapsed_ms=%.1f",
        p.org_id, view.intent_name, view.state_watermark,
        view.policy_version, len(view.entities), elapsed_ms,
    )
    return body
