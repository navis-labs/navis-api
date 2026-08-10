"""FastAPI application factory — the composition root.

Wires the edge (routers) to the domains and the platform. Importing
`navis.registry` here registers all ORM models and event subscribers.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

import navis.registry  # noqa: F401  (model + event-handler registration side effects)
from navis.api import api_router
from navis.identity.google_auth import aclose_http_client as aclose_google_oauth_client
from navis.platform.cache import redis as cache_redis
from navis.platform.config import settings
from navis.platform.database import dispose_engine
from navis.platform.monitoring import configure_logging, get_logger
from navis.platform.monitoring.health import liveness, readiness
from navis.platform.monitoring.metrics import PROMETHEUS_CONTENT_TYPE, render_prometheus
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
        settings.environment.value,
        settings.debug,
    )
    # An unrecognised ENVIRONMENT is now a startup error, but an UNSET one still
    # defaults to development — so a production box that forgot the variable would
    # run with every control relaxed. Say so, loudly, once per boot.
    relaxed = settings.relaxed_controls()
    if relaxed:
        logger.warning(
            "ENVIRONMENT=%s — the following are NOT enforced: %s",
            settings.environment.value,
            "; ".join(relaxed),
        )
    yield
    logger.info("Navis shutting down — draining connections")
    await aclose_google_oauth_client()
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

    @app.get("/metrics", tags=["platform"], include_in_schema=False)
    async def metrics_endpoint() -> PlainTextResponse:
        """Prometheus scrape target. Renders the in-process counters/histograms as
        text exposition — operational counts only, no secrets. Lightweight (a locked
        dict copy + string render). Restrict at the network/ingress layer as usual."""
        return PlainTextResponse(render_prometheus(), media_type=PROMETHEUS_CONTENT_TYPE)

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
