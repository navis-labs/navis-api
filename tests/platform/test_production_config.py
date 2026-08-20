"""TASK 6.4.4 — production configuration must fail closed.

Every check here is about CONFIGURATION, not request handling: what the process
refuses to start with, and what a running process exposes. The invariant is that
production cannot silently behave like development, and that missing
security-critical configuration is a startup error rather than a disabled control.

`Settings(...)` is constructed directly (never the module singleton) so these
never depend on the machine's own .env.
"""
from __future__ import annotations

import pytest

from navis.platform.config.settings import Environment, Settings

# A minimally-valid production configuration. Each test perturbs ONE field, so a
# failure names exactly which control refused.
PROD = dict(
    environment="production",
    secret_key="x" * 40,
    debug=False,
    show_docs=False,
    database_url="postgresql://u:p@db.example.com/neondb",
    cors_origins="https://app.navis.com",
    signals_reject_legacy_plaintext=True,
    execution_inline_fallback=False,
)


def _build(**overrides) -> Settings:
    # `_env_file=None` isolates from the developer's .env — otherwise these assert
    # against whatever that machine happens to have configured.
    return Settings(_env_file=None, **{**PROD, **overrides})  # type: ignore[call-arg]


def _refusal(**overrides) -> str:
    with pytest.raises(Exception) as exc:
        _build(**overrides)
    return str(exc.value)


def _startup_refusal(monkeypatch, **env) -> str:
    """Drive the REAL construction path (`get_settings`), which is what a booting
    process runs, rather than constructing Settings directly."""
    # NOT `from navis.platform.config import settings` — that name is the settings
    # INSTANCE re-exported by the package, which shadows the submodule.
    from navis.platform.config.settings import ConfigurationError, get_settings

    for k, v in env.items():
        monkeypatch.setenv(k.upper(), v)
    get_settings.cache_clear()
    try:
        with pytest.raises(ConfigurationError) as exc:
            get_settings()
        return str(exc.value)
    finally:
        get_settings.cache_clear()


# ── Production startup: each control refuses independently ───────────────────

@pytest.mark.parametrize(
    "override, expected",
    [
        ({"secret_key": "change-me"}, "SECRET_KEY"),
        ({"secret_key": "short"}, "SECRET_KEY"),
        ({"debug": True}, "DEBUG"),
        ({"show_docs": True}, "SHOW_DOCS"),
        ({"database_url": "postgresql://localhost/navis"}, "DATABASE_URL"),
        ({"database_url": "postgresql://127.0.0.1/navis"}, "DATABASE_URL"),
        ({"signals_reject_legacy_plaintext": False}, "SIGNALS_REJECT_LEGACY_PLAINTEXT"),
        ({"cors_origins": "*"}, "CORS_ORIGINS"),
        ({"cors_origins": "https://app.navis.com,*"}, "CORS_ORIGINS"),
        ({"cors_origins": ""}, "CORS_ORIGINS"),
        ({"execution_inline_fallback": True}, "EXECUTION_INLINE_FALLBACK"),
    ],
)
def test_production_refuses_each_insecure_setting(override, expected):
    assert expected in _refusal(**override)


def test_the_same_settings_are_permitted_outside_production():
    """The checks are gated on ENVIRONMENT, not applied globally — development
    keeps its conveniences."""
    s = Settings(
        environment="development", secret_key="change-me", debug=True,
        show_docs=True, database_url="postgresql://localhost/navis",
        cors_origins="*", signals_reject_legacy_plaintext=False,
    )
    assert s.is_production is False


def test_production_refusal_names_every_problem_at_once():
    """An operator fixing config should see the whole list, not one per restart."""
    msg = _refusal(debug=True, show_docs=True, cors_origins="*",
                   signals_reject_legacy_plaintext=False)
    for token in ("DEBUG", "SHOW_DOCS", "CORS_ORIGINS", "SIGNALS_REJECT_LEGACY_PLAINTEXT"):
        assert token in msg


def test_startup_refusal_names_variables_and_never_their_values(monkeypatch):
    """The boot error reaches the process supervisor, log aggregation and tickets.

    pydantic renders `input_value=<the whole settings dict>` next to the reason,
    truncated by LENGTH — which field lands inside that window is an accident of
    declaration order. `get_settings` re-raises with the reasons only.
    """
    msg = _startup_refusal(
        monkeypatch,
        environment="production",
        secret_key="hunter2",  # weak on purpose: short enough to FAIL the check
        database_url="postgresql://dbuser:SYNTHETICPW@db.example.com/neondb",
        debug="true",
    )
    assert "SECRET_KEY" in msg and "DEBUG" in msg      # names the problems
    assert "hunter2" not in msg                        # ...never the values
    assert "SYNTHETICPW" not in msg
    assert "input_value" not in msg


# ── G–J. Pub/Sub provisioning matrix ─────────────────────────────────────────

def test_pubsub_unprovisioned_is_a_supported_production_state():
    """G. No topic => push is simply not in use; ingestion runs on the poll
    schedule. This must NOT be a startup error."""
    s = _build()
    assert s.google_pubsub_topic is None
    assert s.is_production is True


def test_pubsub_topic_without_audience_refuses_to_start():
    """H. A topic makes Google POST to the public webhook; without an audience the
    push token could not be verified for THIS endpoint."""
    msg = _refusal(google_pubsub_topic="projects/p/topics/t")
    assert "GOOGLE_PUBSUB_AUDIENCE" in msg


def test_pubsub_topic_and_audience_without_identity_refuses_to_start():
    """I. Audience alone would accept any Google-issued token carrying it —
    including one minted for someone else's push subscription."""
    msg = _refusal(
        google_pubsub_topic="projects/p/topics/t",
        google_pubsub_audience="https://api.navis.com/api/v1/webhooks/google/pubsub",
    )
    assert "GOOGLE_PUBSUB_SA_EMAIL" in msg or "GOOGLE_PUBSUB_SUB" in msg


@pytest.mark.parametrize("identity", [
    {"google_pubsub_sa_email": "push@p.iam.gserviceaccount.com"},
    {"google_pubsub_sub": "1234567890"},
])
def test_fully_configured_pubsub_is_allowed(identity):
    """J. Either identity pin satisfies the requirement."""
    s = _build(
        google_pubsub_topic="projects/p/topics/t",
        google_pubsub_audience="https://api.navis.com/api/v1/webhooks/google/pubsub",
        **identity,
    )
    assert s.is_production is True


def test_pubsub_topic_is_single_sourced_through_settings():
    """The startup validator keys off `settings.google_pubsub_topic`. If the Gmail
    connector read a different source, validation and behaviour could disagree
    about whether push is configured."""
    import inspect

    from navis.signals.gmail.connector import GmailConnector

    src = inspect.getsource(GmailConnector.create_watch)
    # Compare CODE only — the comment above the assignment names the old source.
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "settings.google_pubsub_topic" in code
    assert "os.environ" not in code


# ── Docs / schema exposure ───────────────────────────────────────────────────

def test_show_docs_false_hides_the_openapi_schema_too(monkeypatch):
    """Disabling the docs UIs while still serving /openapi.json published the whole
    route inventory — the part actually worth having."""
    from fastapi.testclient import TestClient

    import navis.main as M
    from navis.platform.config import settings as live

    monkeypatch.setattr(live, "show_docs", False)
    c = TestClient(M.create_app(), raise_server_exceptions=False)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path).status_code == 404, path

    monkeypatch.setattr(live, "show_docs", True)
    c = TestClient(M.create_app(), raise_server_exceptions=False)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path).status_code == 200, path


# ── Health / metrics exposure ────────────────────────────────────────────────

def test_health_endpoints_expose_no_configuration(monkeypatch):
    from fastapi.testclient import TestClient

    import navis.main as M

    c = TestClient(M.create_app(), raise_server_exceptions=False)
    body = c.get("/health").text + c.get("/").text
    lowered = body.lower()
    for leak in ("postgres", "redis://", "password", "secret", "token", "key=", "@"):
        assert leak not in lowered, f"health/root leaked {leak!r}"


def test_readiness_reports_booleans_not_connection_details():
    import inspect

    from navis.platform.monitoring import health

    src = inspect.getsource(health.readiness)
    assert "database_url" not in src and "redis_url" not in src


# ── T-20: production must not execute provider calls on the API event loop ───


def test_production_refuses_the_inline_execution_fallback():
    """ENVIRONMENT=production + EXECUTION_INLINE_FALLBACK=True -> startup refusal.

    The setting's own docstring already said "In production, run a real Celery
    worker and set this to False" and nothing enforced it, while the DEFAULT is
    True — so a deployment that simply never set the variable ran provider calls
    (real Gmail sends, real Calendar writes) inside the web process.
    """
    msg = _refusal(execution_inline_fallback=True)
    assert "EXECUTION_INLINE_FALLBACK" in msg
    # The refusal must say what to do, not just what is wrong.
    assert "Celery worker" in msg


def test_production_with_the_fallback_off_is_allowed():
    s = _build(execution_inline_fallback=False)
    assert s.is_production is True
    assert s.execution_inline_fallback is False


def test_the_inline_fallback_stays_available_outside_production():
    """Local dev drives the pipeline end to end without a broker; that must keep
    working. The check is gated on ENVIRONMENT, exactly like every sibling."""
    dev = Settings(_env_file=None, environment="development")  # type: ignore[call-arg]
    assert dev.execution_inline_fallback is True
    assert dev.is_production is False


def test_a_dev_environment_announces_that_execution_may_run_inline():
    dev = Settings(_env_file=None, environment="development")  # type: ignore[call-arg]
    joined = " ".join(dev.relaxed_controls()).lower()
    assert "inline" in joined, "a developer must be told side effects can fire from the API process"
