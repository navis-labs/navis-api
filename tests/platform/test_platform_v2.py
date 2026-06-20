"""Platform V2 hardening: event lineage, metrics, JWT algorithm-agnostic signing,
Redis failure-domain config, DB timeout config."""
from __future__ import annotations

from navis.platform.config.settings import Settings
from navis.platform.events import DomainEvent, EventType
from navis.platform.events.events import EVENT_SCHEMA_VERSION
from navis.platform.monitoring.metrics import Metrics
from navis.platform.security import create_access_token, decode_token


# ---- event lineage (correlation + causation + schema version) ----

def test_event_autofills_correlation_and_schema_version():
    e = DomainEvent(type=EventType.ORG_CREATED, payload={"x": 1})
    assert e.correlation_id is not None          # auto-filled from ambient trace/id
    assert e.schema_version == EVENT_SCHEMA_VERSION
    d = e.to_dict()
    assert {"correlation_id", "causation_id", "schema_version"} <= set(d)


def test_caused_builds_lineage_chain():
    parent = DomainEvent(type=EventType.ORG_CREATED, org_id=None)
    child = parent.caused(EventType.ORG_ACTIVATED)
    assert child.causation_id == str(parent.id)          # precise parent→child link
    assert child.correlation_id == parent.correlation_id  # same logical flow


def test_event_construction_is_backward_compatible():
    # Existing call sites pass only type/org_id/payload/source — still valid.
    e = DomainEvent(type=EventType.STATE_UPDATED, payload={}, source="state")
    assert e.source == "state" and e.causation_id is None


# ---- metrics ----

def test_metrics_counters_and_histogram_snapshot():
    m = Metrics()
    m.incr("events.published", labels={"type": "OrgCreated"})
    m.incr("events.published", labels={"type": "OrgCreated"})
    m.observe_ms("events.publish_latency", 12.0)
    snap = m.snapshot()
    assert snap["counters"]["events.published{type=OrgCreated}"] == 2
    hist = snap["histograms"]["events.publish_latency"]
    assert hist["count"] == 1 and hist["avg_ms"] == 12.0


# ---- JWT: algorithm-agnostic signing (HS256 default, asymmetric migration path) ----

def test_jwt_hs256_round_trip():
    token = create_access_token("user-1", "org-1")
    payload = decode_token(token, expected_type="access")
    assert payload["sub"] == "user-1" and payload["org"] == "org-1"
    assert payload["jti"] and payload["iss"]  # revocable + issuer-bound


def test_jwt_asymmetric_requires_keys():
    import pytest

    s = Settings(algorithm="RS256", jwt_private_key=None, jwt_public_key=None)
    assert s.jwt_is_asymmetric is True
    with pytest.raises(ValueError):
        _ = s.jwt_signing_key   # RS256 without a private key fails loudly
    with pytest.raises(ValueError):
        _ = s.jwt_verify_key


def test_jwt_hs256_uses_shared_secret_for_both():
    s = Settings(algorithm="HS256", secret_key="x" * 40)
    assert s.jwt_signing_key == s.jwt_verify_key == "x" * 40
    assert s.jwt_is_asymmetric is False


# ---- Redis failure-domain separation (config-only, defaults to redis_url) ----

def test_redis_urls_default_to_single_instance():
    s = Settings(redis_url="redis://r:6379/0")
    assert s.effective_redis_cache_url == "redis://r:6379/0"
    assert s.effective_redis_events_url == "redis://r:6379/0"


def test_redis_urls_can_be_split():
    s = Settings(
        redis_url="redis://r:6379/0",
        redis_cache_url="redis://cache:6379/0",
        redis_events_url="redis://events:6379/0",
    )
    assert s.effective_redis_cache_url == "redis://cache:6379/0"
    assert s.effective_redis_events_url == "redis://events:6379/0"


# ---- DB statement timeout config present ----

def test_db_statement_timeout_configurable():
    assert Settings(db_statement_timeout_seconds=10.0).db_statement_timeout_seconds == 10.0
