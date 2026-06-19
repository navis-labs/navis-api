"""Adaptive runtime injection (no dependency inversion).

Reasoning/Decision/Execution must NOT import Adaptive. Instead, the orchestration
layer (which legally sits above everything) resolves an org's ACTIVE adaptive policy
and injects the relevant values into the domain call. Domains stay pure; policy
flows: Adaptive → active snapshot → this resolver → domain params.

Today Adaptive proposes exactly two governing variables that the brief consumes —
brief_min_confidence and brief_max_items — so injecting them here makes the loop
real end-to-end. As Adaptive's vocabulary grows, extend the mapping below; no domain
needs to change.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from navis.adaptive.repository import AdaptiveRepository


async def resolve_policy(db: AsyncSession, org_id: uuid.UUID) -> dict[str, Any]:
    """The org's active policy payload merged over code defaults (never raises)."""
    return await AdaptiveRepository(db).get_active_payload(org_id)
