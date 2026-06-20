"""Logging configuration + correlation-id propagation.

One entry point (`configure_logging`) so every process — API, worker, beat — logs
identically. Supports human-readable console logs in dev and structured JSON in
production (`LOG_FORMAT=json`). Every record carries the current `request_id` so
a single request can be traced across async hops.
"""
from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone

from navis.platform.config import settings

# Correlation id for the current request/task. Set by middleware (API) or per-job.
_request_id: ContextVar[str] = ContextVar("request_id", default="-")
# Tenant context: when set, every log line (and every event) carries the org, so a
# single tenant's activity is traceable end-to-end across async hops and workers.
_org_id: ContextVar[str] = ContextVar("org_id", default="-")


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str:
    return _request_id.get()


def set_org_id(value: str | None) -> None:
    _org_id.set(value or "-")


def get_org_id() -> str:
    return _org_id.get()


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        record.org_id = _org_id.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "org_id": getattr(record, "org_id", "-"),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    if settings.log_format.lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s [%(request_id)s] | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third parties unless explicitly debugging.
    if level > logging.DEBUG:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"navis.{name}")
