"""Production-readiness guards for the two P0 blockers.

P0-1: the deployment ran the API only, so the outbox relay never executed and
      Gmail never became memory.
P0-2: `environment` was a free-form string compared with == "production", so a
      typo silently relaxed encryption enforcement, entitlements and the startup
      safety checks all at once.
"""
from __future__ import annotations

import pathlib

import pytest

from navis.platform.config.settings import Environment, Settings

ROOT = pathlib.Path(__file__).resolve().parents[2]


# ── P0-2: environment can no longer fail open ──────────────────────────────


@pytest.mark.parametrize("value", ["production", "PRODUCTION", "  Production  "])
def test_production_spellings_are_recognised(value):
    """Case and stray whitespace are deployment noise, not a different environment."""
    s = Settings(environment=value, _env_file=None, secret_key="x" * 40,
                 debug=False, show_docs=False,
                 database_url="postgresql://u:p@db.example.com/n")
    assert s.environment is Environment.PRODUCTION
    assert s.is_production is True


@pytest.mark.parametrize("value", ["prod", "producton", "PROD", "live", "", "dev"])
def test_unrecognised_environments_are_rejected_not_downgraded(value):
    """The original bug: anything unrecognised took the permissive branch."""
    with pytest.raises(Exception) as excinfo:
        Settings(environment=value, _env_file=None)
    assert "not recognised" in str(excinfo.value)


def test_error_names_the_allowed_values():
    with pytest.raises(Exception) as excinfo:
        Settings(environment="prod", _env_file=None)
    message = str(excinfo.value)
    for allowed in ("development", "staging", "production"):
        assert allowed in message


def test_production_still_enforces_the_config_safety_net():
    """A weak SECRET_KEY must fail the boot it previously slipped past."""
    with pytest.raises(Exception) as excinfo:
        Settings(environment="production", _env_file=None, secret_key="change-me")
    assert "Insecure production config" in str(excinfo.value)


def test_relaxed_controls_are_empty_in_production_and_listed_otherwise():
    prod = Settings(environment="production", _env_file=None, secret_key="x" * 40,
                    debug=False, show_docs=False,
                    database_url="postgresql://u:p@db.example.com/n")
    assert prod.relaxed_controls() == []

    dev = Settings(environment="development", _env_file=None)
    relaxed = dev.relaxed_controls()
    assert relaxed, "an unset/dev environment must announce what it is not enforcing"
    joined = " ".join(relaxed).lower()
    for control in ("encryption", "entitlement", "plan"):
        assert control in joined


def test_default_is_development_and_says_so():
    """Unset ENVIRONMENT is the remaining fail-open path; it must be visible."""
    s = Settings(_env_file=None)
    assert s.environment is Environment.DEVELOPMENT
    assert s.is_production is False
    assert s.relaxed_controls()


# ── P0-1: the deployment must run the whole pipeline ───────────────────────


def _compose() -> str:
    path = ROOT / "docker-compose.yml"
    assert path.exists(), "no compose file: the pipeline has no supervised runtime"
    return path.read_text(encoding="utf-8")


def test_deployment_defines_api_worker_and_beat():
    """API-only was the blocker: signals accumulate and nothing relays them."""
    compose = _compose()
    for service in ("api:", "worker:", "beat:"):
        assert service in compose, f"docker-compose is missing the {service} service"
    assert "worker" in compose and "celery" in compose
    assert "beat" in compose


def test_every_service_restarts_on_death():
    """A worker that dies silently stalls the product behind a healthy API."""
    compose = _compose()
    # one restart policy per service (redis, api, worker, beat)
    assert compose.count("restart: unless-stopped") >= 4


def test_worker_does_not_use_a_thread_pool():
    """`run_async` holds ONE event loop per process; asyncpg pools bind to it.
    A thread pool would drive that loop from several threads and corrupt it."""
    # Strip comments: the file WARNS about thread pools in prose, and the check
    # is about what is actually executed.
    directives = "\n".join(
        line for line in _compose().splitlines() if not line.lstrip().startswith("#")
    )
    assert "--pool=prefork" in directives
    for unsafe in ("--pool=threads", "--pool=gevent", "--pool=eventlet"):
        assert unsafe not in directives


def test_procfile_covers_the_same_three_processes():
    path = ROOT / "Procfile"
    assert path.exists(), "no Procfile: non-Docker deploys would run the API alone"
    text = path.read_text(encoding="utf-8")
    for proc in ("web:", "worker:", "beat:"):
        assert proc in text
    assert "alembic upgrade head" in text, "migrations must run on release"


def test_beat_schedule_is_persisted():
    """A beat restart must not re-fire the whole schedule from scratch."""
    assert "--schedule=" in _compose()


def test_is_production_survives_raw_string_assignment():
    """Pydantic does not re-validate attribute assignment. An identity check would
    report "not production" for `settings.environment = "production"` — the same
    fail-open, relocated to runtime. Tests monkeypatch this attribute."""
    s = Settings(_env_file=None)
    s.environment = "production"          # type: ignore[assignment]
    assert s.is_production is True
    s.environment = "  PRODUCTION  "      # type: ignore[assignment]
    assert s.is_production is True
    s.environment = "development"         # type: ignore[assignment]
    assert s.is_production is False


# ── migrations must apply to a FRESH database ──────────────────────────────


def test_revision_ids_fit_alembics_version_column():
    """Alembic hardcodes `alembic_version.version_num` as VARCHAR(32) — see
    `DefaultImpl.version_table_impl`; there is no option to widen it.

    A longer revision id still works on a database whose version table already
    holds a shorter id, so this stays invisible until someone provisions a NEW
    environment: the upgrade then aborts partway with
    StringDataRightTruncationError and no deployment can be created. Two ids
    (34 and 39 chars) were over the limit and had to be shortened.
    """
    import re

    offenders = []
    for path in sorted((ROOT / "migrations" / "versions").glob("*.py")):
        match = re.search(r'^revision: str = "([^"]+)"', path.read_text(encoding="utf-8"), re.M)
        if match and len(match.group(1)) > 32:
            offenders.append(f"{match.group(1)} ({len(match.group(1))} chars)")
    assert not offenders, (
        "revision ids exceed alembic's VARCHAR(32) version column, which breaks "
        "migration of a fresh database: " + ", ".join(offenders)
    )
