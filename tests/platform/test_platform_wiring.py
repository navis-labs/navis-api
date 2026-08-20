"""PLATFORM-001/002/003 wiring: Prometheus /metrics exposition, org_id log binding,
and the scheduled session/token purge tasks."""
from __future__ import annotations

import logging
import uuid

import pytest
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


# ---- T-21: background work binds tenant context from a persisted row ----
#
# `set_org_id` was called in exactly ONE place — get_principal — so every log line
# emitted by a Celery task, an event subscriber or an outbox relay carried
# org_id="-". Since sync, brief generation, execution, observation, learning,
# intelligence and adaptive all run in workers, the MAJORITY of tenant-affecting
# work in the system was unattributable in logs.


def test_org_context_binds_and_restores():
    from navis.platform.monitoring import get_org_id, org_context, set_org_id

    set_org_id(None)
    assert get_org_id() == "-"
    with org_context("org-a"):
        assert get_org_id() == "org-a"
    assert get_org_id() == "-", "context leaked out of the block"


def test_org_context_restores_rather_than_clears():
    """RESTORE, not clear: on the request path an inner block must not wipe the
    org the auth dependency bound for the rest of the request."""
    from navis.platform.monitoring import get_org_id, org_context, set_org_id

    set_org_id("outer-org")
    try:
        with org_context("inner-org"):
            assert get_org_id() == "inner-org"
        assert get_org_id() == "outer-org", "restored the default instead of the caller's org"
    finally:
        set_org_id(None)


def test_org_context_restores_even_when_the_body_raises():
    from navis.platform.monitoring import get_org_id, org_context, set_org_id

    set_org_id(None)
    with pytest.raises(RuntimeError):
        with org_context("org-a"):
            raise RuntimeError("boom")
    assert get_org_id() == "-", "a failing task poisoned the next task's context"


def test_org_context_accepts_a_uuid_and_none():
    from navis.platform.monitoring import get_org_id, org_context, set_org_id

    org = uuid.uuid4()
    set_org_id(None)
    with org_context(org):
        assert get_org_id() == str(org)
    with org_context(None):
        assert get_org_id() == "-"


def test_two_tenants_tasks_do_not_bleed_into_each_other():
    """Org A's task logs org A, org B's logs org B, and neither survives the
    other — the property that makes worker logs usable at all."""
    from navis.platform.monitoring import get_org_id, org_context, set_org_id

    set_org_id(None)
    seen = []
    for org in ("org-a", "org-b", "org-c"):
        with org_context(org):
            seen.append(get_org_id())
        assert get_org_id() == "-", f"{org} leaked past its own task"
    assert seen == ["org-a", "org-b", "org-c"]


def test_every_tenant_bound_background_entry_point_binds_context():
    """Structural: the day a new background producer is added, this fails unless
    it binds the tenant it is acting for."""
    import ast
    import inspect
    import textwrap

    import navis.execution.worker as W
    import navis.ml.embeddings.jobs as J
    import navis.orchestration.brief as B
    import navis.orchestration.sync as S

    for fn in (
        S.sync_integration,
        S._process_slack_event,
        B.run_brief_for_org,
        J._embed_org,
        W._execute_one,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "org_context" in names, f"{fn.__qualname__} does not bind tenant log context"


def test_background_context_is_never_taken_from_a_task_payload():
    """The org bound by the sync path comes from the FRESHLY RE-READ row, not the
    caller's argument — pinning that the re-read is what feeds it."""
    import inspect

    import navis.orchestration.sync as S

    src = inspect.getsource(S.sync_integration)
    assert "with org_context(fresh.organization_id)" in src
