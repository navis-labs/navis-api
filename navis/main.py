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
from navis.ml.embeddings.pins import read_pin, write_pin
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
    # A split embedding pin is a MIGRATION IN FLIGHT, and it is invisible from
    # the outside: retrieval keeps answering, just from whatever subset of the
    # corpus the read index happens to hold. Live, that had degraded /chat to
    # 6.7% of Memory with no error anywhere. Say which index reads and writes
    # are addressing, once per boot, so a half-finished cutover cannot sit
    # unnoticed. Never fatal — holding the pins apart is the supported way to
    # re-embed safely.
    try:
        r, w = read_pin(), write_pin()
        if (r.model, r.version) != (w.model, w.version):
            logger.warning(
                "embedding pins SPLIT: reads=%s/v%d writes=%s/v%d — retrieval sees only "
                "the read index. Finish the backfill, then set EMBEDDING_READ_MODEL=%s "
                "to complete the cutover.",
                r.model, r.version, w.model, w.version, w.model,
            )
    except Exception:  # noqa: BLE001 — a diagnostic must never block boot
        logger.exception("could not resolve embedding pins")
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
        # `docs_url`/`redoc_url` only hide the HTML UIs. Without this, /openapi.json
        # kept serving the COMPLETE schema — 81 paths with every parameter, body
        # model and status code — so `SHOW_DOCS=False` (which production requires,
        # see settings._enforce_production_safety) suppressed the viewer while
        # leaving the machine-readable original public. The schema is the part worth
        # having: it enumerates the whole attack surface without any crawling.
        openapi_url="/openapi.json" if settings.show_docs else None,
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
