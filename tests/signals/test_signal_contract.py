"""Contract tests: immutability, idempotency, version-awareness, tenancy."""
from __future__ import annotations

import dataclasses
import uuid

import pytest

from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalEventType,
    SignalSource,
    SignalValidationError,
)
from tests.signals.fakes import make_ctx


def _event(ctx, *, external_id="x1", version="1"):
    return SignalEvent.create(
        ctx,
        source=SignalSource.GMAIL,
        event_type=SignalEventType.EMAIL_RECEIVED,
        external_id=external_id,
        external_version=version,
    )


def test_event_id_is_deterministic_uuid5_of_dedupe_key():
    ctx = make_ctx()
    a = _event(ctx, external_id="m1", version="7")
    b = _event(ctx, external_id="m1", version="7")
    assert a.event_id == b.event_id  # idempotent identity
    assert a.dedupe_key == f"{ctx.org_id}:gmail:m1:7"


def test_version_awareness_distinguishes_updates():
    ctx = make_ctx()
    v1 = _event(ctx, external_id="m1", version="1")
    v2 = _event(ctx, external_id="m1", version="2")
    assert v1.event_id != v2.event_id  # a new version is a NEW fact, not a dup
    assert v1.dedupe_key != v2.dedupe_key


def test_event_is_immutable():
    ev = _event(make_ctx())
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.external_id = "tampered"  # type: ignore[misc]


def test_tenant_agnostic_signal_is_rejected():
    bad = SignalContext(org_id=None, user_id=uuid.uuid4(), integration_id=uuid.uuid4())  # type: ignore[arg-type]
    with pytest.raises(SignalValidationError):
        _event(bad)


def test_missing_user_or_integration_is_rejected():
    with pytest.raises(SignalValidationError):
        _event(SignalContext(org_id=uuid.uuid4(), user_id=None, integration_id=uuid.uuid4()))  # type: ignore[arg-type]
    with pytest.raises(SignalValidationError):
        _event(SignalContext(org_id=uuid.uuid4(), user_id=uuid.uuid4(), integration_id=None))  # type: ignore[arg-type]


def test_missing_version_is_rejected():
    with pytest.raises(SignalValidationError):
        SignalEvent.create(
            make_ctx(),
            source=SignalSource.GMAIL,
            event_type=SignalEventType.EMAIL_RECEIVED,
            external_id="m1",
            external_version="",
        )


def test_to_dict_round_trips_required_fields():
    ev = _event(make_ctx())
    d = ev.to_dict()
    for key in (
        "event_id",
        "org_id",
        "user_id",
        "integration_id",
        "source",
        "event_type",
        "external_id",
        "external_version",
        "timestamp",
        "payload",
        "metadata",
    ):
        assert key in d
