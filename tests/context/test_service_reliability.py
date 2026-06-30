"""Reliability regression tests for navis/context/service.py: telemetry isolation (a
metrics/audit failure must not discard a valid bundle) and failure propagation.
"""
from __future__ import annotations

import pytest

from navis.context import ContextRequest, ContextService
from navis.state.v2.builders import build_projection
from tests.context.fakes import FakeStateReader
from tests.state.fakes import ORG_A, corpus


def _svc():
    return ContextService(state_service=FakeStateReader(build_projection(ORG_A, corpus())))


# --- Telemetry isolation: a logging/metrics failure never fails a valid build -----

async def test_audit_failure_does_not_discard_bundle(monkeypatch):
    import navis.context.service as svc_mod

    def _boom(**_kwargs):
        raise RuntimeError("audit sink down")

    monkeypatch.setattr(svc_mod, "log_audit", _boom)
    bundle = await _svc().build(ContextRequest.full_org(ORG_A))  # must NOT raise
    assert bundle.context_id and bundle.state.counts()["person"] >= 1  # valid bundle returned


async def test_metrics_failure_does_not_discard_bundle(monkeypatch):
    import navis.context.metrics as metrics_mod

    def _boom(*_a, **_k):
        raise RuntimeError("metrics sink down")

    monkeypatch.setattr(metrics_mod, "observe_ms", _boom)
    bundle = await _svc().build(ContextRequest.full_org(ORG_A))  # must NOT raise
    assert bundle.context_id


# --- Failure propagation: a real build failure is not swallowed -------------

async def test_build_failure_propagates():
    class BoomReader:
        async def get_state(self, org_id):
            raise RuntimeError("state unavailable")

    with pytest.raises(RuntimeError, match="state unavailable"):
        await ContextService(state_service=BoomReader()).build(ContextRequest.full_org(ORG_A))
