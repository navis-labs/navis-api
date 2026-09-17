"""FastAPI application factory — the composition root.

Wires the edge (routers) to the domains and the platform. Importing
`navis.registry` here registers all ORM models and event subscribers.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import navis.registry  # noqa: F401  (model + event-handler registration side effects)
from navis.api import api_router
from navis.platform.cache import redis as cache_redis
from navis.platform.config import settings
from navis.platform.database import dispose_engine
from navis.platform.monitoring import configure_logging, get_logger
from navis.platform.monitoring.health import liveness, readiness
from navis.platform.monitoring.metrics import metrics as _metrics
from navis.platform.monitoring.middleware import RequestContextMiddleware
from navis.platform.monitoring.sentry import init_sentry

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_sentry()
    logger.info(
        "Navis %s starting (env=%s debug=%s)",
        settings.app_version,
        settings.environment,
        settings.debug,
    )
    yield
    logger.info("Navis shutting down — draining connections")
    await dispose_engine()
    await cache_redis.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs" if settings.show_docs else None,
        redoc_url="/redoc" if settings.show_docs else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Correlation id on every request (outermost so all logs carry it).
    app.add_middleware(RequestContextMiddleware)

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/health", tags=["platform"])
    @app.get("/live", tags=["platform"])
    async def health() -> dict:
        """Liveness: the process is up. Cheap, no dependency checks."""
        return liveness()

    @app.get("/ready", tags=["platform"])
    async def ready() -> dict:
        """Readiness: Postgres + Redis reachable. For LB/k8s gating."""
        return await readiness()

    @app.get("/metrics", tags=["platform"])
    async def metrics_endpoint() -> dict:
        """In-process metrics snapshot (counters + latency histograms).

        Returns a JSON-serialisable dict with two top-level keys:
          * ``counters``  — all named counters (events published, auth rejections, …)
          * ``histograms`` — all named histograms with count, sum_ms, avg_ms, buckets_ms

        Not authenticated — expose only on an internal/admin network in production
        (behind a firewall or k8s NetworkPolicy).
        """
        return _metrics.snapshot()

    @app.get("/", tags=["platform"])
    async def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "pipeline": "signals -> memory -> state -> reasoning -> decisions "
            "-> execution -> observation -> learning -> intelligence",
            "docs": "/docs",
        }

    return app


app = create_app()
