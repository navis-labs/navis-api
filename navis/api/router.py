"""Aggregate router. Each domain owns its routes; the edge just mounts them."""
from __future__ import annotations

from fastapi import APIRouter

from navis.decisions.router import router as decisions_router
from navis.execution.router import router as execution_router
from navis.identity.router import router as identity_router
from navis.intelligence.router import router as intelligence_router
from navis.signals.router import router as signals_router

api_router = APIRouter()
api_router.include_router(identity_router)
api_router.include_router(signals_router)
api_router.include_router(decisions_router)
api_router.include_router(execution_router)
api_router.include_router(intelligence_router)

