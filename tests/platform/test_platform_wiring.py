"""PLATFORM-001/002/003 wiring: Prometheus /metrics exposition, org_id log binding,
and the scheduled session/token purge tasks."""
from __future__ import annotations

import logging
import uuid

from fastapi.security import HTTPAuthorizationCredentials


# ---- PLATFORM-001: Prometheus exposition ----

def test_render_prometheus_counters_and_histograms():
    from navis.platform.monitoring.metrics import Metrics, render_prometheus

    m = Metrics()
    m.incr("auth.rejected", labels={"status": "401"})
    m.incr("auth.rejected", labels={"status": "401"})
    m.incr("events.published")
    m.observe_ms("pipeline.brief", 7.0)
    m.observe_ms("pipeline.brief", 120.0)
    out = render_prometheus(m.snapshot())

    # names sanitized to the Prometheus charset ('.' -> '_'); labels quoted.
    assert "# TYPE auth_rejected counter" in out
    assert 'auth_rejected{status="401"} 2.0' in out
    assert "events_published 1.0" in out
    # histogram: TYPE + cumulative le buckets + +Inf == count + sum + count.
    assert "# TYPE pipeline_brief histogram" in out
    assert 'pipeline_brief_bucket{le="250"} 2' in out      # both observations <= 250ms
    assert 'pipeline_brief_bucket{le="+Inf"} 2' in out
    assert "pipeline_brief_count 2" in out
    assert "pipeline_brief_sum 127.0" in out               # 7 + 120


def test_render_prometheus_empty_is_valid():
    from navis.platform.monitoring.metrics import render_prometheus

    assert render_prometheus({"counters": {}, "histograms": {}}) == "\n"


def test_metrics_endpoint_scrapeable():
    from fastapi.testclient import TestClient

    from navis.main import app
    from navis.platform.monitoring.metrics import metrics

    metrics.incr("test.scrape.marker")
    client = TestClient(app)  # bare client: no lifespan, /metrics needs no DB/redis
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "version=0.0.4" in r.headers["content-type"]
    assert "test_scrape_marker" in r.text


# ---- PLATFORM-002: org_id bound to the logging context after auth ----

async def test_get_principal_binds_org_id(monkeypatch):
    from navis.identity import authz
    from navis.identity.authz import Principal, PrincipalKind, get_principal
    from navis.platform.monitoring import get_org_id, set_org_id

    set_org_id(None)  # reset to "-"
    org = uuid.uuid4()

    async def fake_jwt(db, token):
        return Principal(
            kind=PrincipalKind.USER, org_id=org, subject_id=uuid.uuid4(), permissions=frozenset()
        )

    monkeypatch.setattr(authz, "_principal_from_jwt", fake_jwt)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="a.jwt.token")

    p = await get_principal(creds=creds, x_api_key=None, db=None)  # type: ignore[arg-type]

    assert p.org_id == org
    assert get_org_id() == str(org)   # every log line in this request now carries it
    set_org_id(None)


def test_json_formatter_emits_org_id_and_request_id():
    from navis.platform.monitoring.logging import (
        _JsonFormatter,
        _RequestIdFilter,
        set_org_id,
        set_request_id,
    )

    set_request_id("req-xyz")
    set_org_id("org-123")
    rec = logging.LogRecord("navis.test", logging.INFO, __file__, 1, "hello", None, None)
    _RequestIdFilter().filter(rec)
    out = _JsonFormatter().format(rec)
    assert '"org_id": "org-123"' in out
    assert '"request_id": "req-xyz"' in out
    set_org_id(None)


# ---- PLATFORM-003: purge tasks registered + scheduled ----

def test_purge_tasks_registered_and_scheduled():
    import navis.registry  # noqa: F401  (imports identity.maintenance -> registers tasks)
    from navis.platform.queue import celery_app

    task_names = set(celery_app.tasks.keys())
    sessions = "navis.identity.maintenance.purge_expired_sessions"
    tokens = "navis.identity.maintenance.purge_expired_tokens"
    assert sessions in task_names
    assert tokens in task_names

    scheduled = {e["task"]: e["schedule"] for e in celery_app.conf.beat_schedule.values()}
    assert scheduled.get(sessions) == 60 * 60   # hourly
    assert scheduled.get(tokens) == 60 * 60
