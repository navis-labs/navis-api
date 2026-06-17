"""Central configuration. Reads from environment / .env.

This is the ONLY place that touches raw environment variables. Everything else
imports `settings`.
"""
from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App ----
    app_name: str = "Navis AI Backend"
    app_version: str = "2.0.0"
    debug: bool = True
    secret_key: str = "change-me"

    api_v1_prefix: str = "/api/v1"
    # Comma-separated in env; use the `cors_origins_list` property for the parsed value.
    cors_origins: str = "http://localhost:3000"

    # ---- Database ----
    database_url: str = "postgresql://localhost/navis"

    # ---- Redis ----
    redis_url: str = "redis://localhost:6379/0"

    # ---- Security / JWT ----
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30
    algorithm: str = "HS256"

    # ---- AI / ML ----
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_ai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-12-01-preview"

    # ---- OAuth ----
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str | None = None
    google_gmail_redirect_uri: str | None = None
    slack_client_id: str | None = None
    slack_client_secret: str | None = None
    slack_redirect_uri: str | None = None

    # ---- Background tasks ----
    celery_broker_url: str = "redis://localhost:6379/2"
    celery_result_backend: str = "redis://localhost:6379/3"

    # ---- Feature flags ----
    enable_google_auth: bool = True
    enable_ai_caching: bool = True

    # ---- Dev ----
    reload: bool = True
    show_docs: bool = True
    log_level: str = "INFO"

    # ---- Environment ----
    environment: str = "development"  # development | staging | production

    # ---- Database pool ----
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_recycle_seconds: int = 300  # Neon drops idle conns ~5m; recycle below that
    db_pool_timeout_seconds: int = 30

    # ---- Redis pool ----
    redis_max_connections: int = 50
    redis_socket_timeout_seconds: float = 5.0

    # ---- Storage (S3) ----
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    s3_bucket_name: str | None = None

    # ---- Observability ----
    sentry_dsn: str | None = None
    log_format: str = "console"  # console | json
    request_id_header: str = "X-Request-ID"

    # ---- Event bus ----
    event_handler_timeout_seconds: float = 15.0

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def db_is_pooled(self) -> bool:
        """Neon/PgBouncer pooled endpoints forbid asyncpg prepared statements."""
        url = self.database_url.lower()
        return "-pooler" in url or "pgbouncer" in url

    @model_validator(mode="after")
    def _enforce_production_safety(self) -> "Settings":
        """Fail fast at startup if a production deployment is misconfigured."""
        if not self.is_production:
            return self
        problems: list[str] = []
        if self.secret_key in ("", "change-me") or len(self.secret_key) < 32:
            problems.append("SECRET_KEY must be a strong (>=32 char) random value")
        if self.debug:
            problems.append("DEBUG must be False")
        if "localhost" in self.database_url or "127.0.0.1" in self.database_url:
            problems.append("DATABASE_URL points at localhost")
        if self.show_docs:
            problems.append("SHOW_DOCS should be False (do not expose /docs publicly)")
        if problems:
            raise ValueError("Insecure production config: " + "; ".join(problems))
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def async_database_url(self) -> str:
        """asyncpg-compatible URL.

        Neon ships URLs like `postgresql://...?sslmode=require&channel_binding=require`.
        asyncpg rejects libpq query params, so we switch the driver and strip the
        query string. TLS is re-applied via connect_args (see database/session.py).
        """
        parts = urlsplit(self.database_url)
        scheme = parts.scheme
        if scheme == "postgresql":
            scheme = "postgresql+asyncpg"
        elif scheme.startswith("postgresql+") and "asyncpg" not in scheme:
            scheme = "postgresql+asyncpg"
        return urlunsplit((scheme, parts.netloc, parts.path, "", ""))

    @property
    def db_requires_ssl(self) -> bool:
        return "sslmode=require" in self.database_url or "neon.tech" in self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
