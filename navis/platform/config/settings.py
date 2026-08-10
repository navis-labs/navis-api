"""Central configuration. Reads from environment / .env.

This is the ONLY place that touches raw environment variables. Everything else
imports `settings`.
"""
from __future__ import annotations

import enum
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, enum.Enum):
    """The deployment environment. A CLOSED set, deliberately.

    `is_production` gates six independent controls — production config checks,
    field-encryption fail-fast, OAuth-token-encryption fail-fast, entitlement
    enforcement, the capability guard, and plan defaults. It was previously a
    free-form string compared with `== "production"`, so `ENVIRONMENT=prod`, a
    trailing space, or a misspelling silently took the PERMISSIVE branch of all
    six at once: credentials stored unencrypted, every capability granted, and
    the startup safety net skipped — with no error and no log line.

    Making it an enum means an unrecognised value fails at import, before the
    process can serve a request. Wrong config now cannot masquerade as dev.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


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
    # Failure-domain separation (Phase 6): cache and the event stream can be split
    # onto independent Redis instances later WITHOUT code change — each defaults to
    # `redis_url`. A cache outage then can't take down event mirroring, and vice versa.
    redis_cache_url: str | None = None
    redis_events_url: str | None = None

    # ---- Security / JWT ----
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30
    # HS256 (shared secret) by default. Set algorithm to RS256/ES256 + provide PEM
    # keys to migrate to asymmetric signing (public-key verification, key rotation)
    # without changing any call site. See platform/security/jwt.py.
    algorithm: str = "HS256"
    jwt_private_key: str | None = None  # PEM — signing key for RS256/ES256
    jwt_public_key: str | None = None   # PEM — verification key for RS256/ES256
    jwt_key_id: str | None = None       # `kid` header for key rotation / JWKS
    # Fernet keys (comma-separated, newest first) for at-rest encryption. Defined here
    # so they load from .env like the rest of config; the ciphers still honor a real
    # exported env var first (prod secret managers / tests). Empty => passthrough.
    field_encryption_keys: str | None = None  # FieldCipher — SSO secrets, sensitive fields
    signals_token_keys: str | None = None     # TokenCipher — integration OAuth tokens
    # Sunset switch (SIGNAL-CRYPTO-002): once every token is confirmed encrypted, flip this
    # to reject untagged legacy-plaintext tokens on read instead of trusting them forever.
    signals_reject_legacy_plaintext: bool = False
    # Ingestion batching — one multi-row insert + one commit + one publish-fan-out per
    # chunk (vs per-event). On a batch failure the sync engine falls back to per-event so a
    # single poison event can't sink the chunk. sync_timeout bounds a hung connector.
    signals_ingest_batch_size: int = 500
    signals_sync_timeout_seconds: float = 120.0
    # DLQ replay: after N failed re-ingestion attempts a poison item is permanently
    # dead-lettered (resolved) instead of retried forever — the poison-loop guard.
    signals_dlq_max_replay_attempts: int = 5
    # Outbox retention: published pointer rows older than this are purged (bounds growth).
    signals_outbox_retention_seconds: int = 7 * 24 * 60 * 60
    # Memory outbox retention — mirrors the signal outbox (same rationale: bound growth
    # of the published-pointer backlog without dropping unpublished rows).
    memory_outbox_retention_seconds: int = 7 * 24 * 60 * 60
    # Calendar capture: how far back a full sync reaches (future events are unbounded), and
    # the page-drain safety cap (exceeding it fails LOUD/retryable, never silently truncates).
    calendar_full_sync_window_days: int = 60
    calendar_max_sync_pages: int = 1000  # 1000 * 250 = 250k events/sync before a hard stop
    # Drive capture: onboarding backfill cap (observable when hit — never silent) + the
    # change-feed page-drain safety cap (exceeding it fails LOUD/retryable, never stalls).
    drive_onboard_file_cap: int = 2000
    drive_max_sync_pages: int = 1000
    # Gmail capture: onboarding backfill cap (observable when hit) + the history-drain
    # page-safety cap. Exceeding the history cap fails LOUD/retryable and does NOT advance
    # the cursor — silently stopping mid-drain would skip messages forever (the cursor
    # would jump past unread history pages).
    gmail_onboard_message_cap: int = 500
    # Onboarding backfill window (in days). full_sync passes this as
    # Gmail's `q=newer_than:{N}d` query so we only pull recent mail —
    # older messages stay in the user's inbox but never enter our
    # signal log, memory, or state. Prevents the dashboard from
    # surfacing 3-week-old threads as "slipping commitments" on first
    # connect. Frontend surface filters (15 days) sit on top of this.
    gmail_onboard_lookback_days: int = 30
    gmail_max_history_pages: int = 200
    # Slack capture: the signing secret verifies inbound Events API webhooks (HMAC-SHA256);
    # the timestamp window rejects replayed captures. Backfill caps bound onboarding REST
    # polling. Scope version anchors re-consent when the read-only scope set changes.
    slack_signing_secret: str | None = None
    # Optional rotation: comma-separated secrets, NEWEST first. Any one validating accepts the
    # request, so a secret can be rotated with zero downtime. Falls back to the single secret.
    slack_signing_secrets: str | None = None
    slack_webhook_max_age_seconds: int = 300  # 5-min replay window per Slack guidance
    slack_scope_version: str = "v1"
    slack_backfill_message_cap: int = 1000
    slack_backfill_max_pages: int = 100

    # ---- AI / ML ----
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_ai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None
    # Embeddings live behind their OWN Azure deployment (chat and embedding
    # models are different SKUs in Azure — one deployment cannot serve both).
    # Leave unset to fall back to the deterministic hash-based embedder, which
    # keeps chat working offline without touching Azure at all. When set, this
    # value MUST be the deployment name in your Azure OpenAI resource, not the
    # underlying model name — Azure routes by deployment. The model column in
    # `memory_embeddings` will be recorded as this string, so changing it will
    # transparently invalidate old embeddings (the version-and-model index
    # keeps them isolated — no bad hits, just a re-embed on the next backfill).
    azure_openai_embed_deployment: str | None = None
    azure_openai_api_version: str = "2024-12-01-preview"

    # ---- Embedding read pin (migration control) ----
    # Retrieval reads a SPECIFIC (model, version) index. Writes go to whatever
    # the configured embedding provider produces. Keeping the two independent is
    # what makes a re-embed safe: the new index is built to completion in the
    # background while every query keeps hitting the old, complete one.
    #
    # Both unset (the default) => reads follow writes, i.e. exactly today's
    # behaviour. During a migration, pin reads to the CURRENT index first, then
    # switch the write provider, backfill, verify, and finally move the pin.
    # Rollback is moving the pin back — the old vectors are never deleted.
    #
    # `embedding_read_model` must name an index that exists: either the
    # deterministic embedder's model or the configured Azure embed deployment.
    embedding_read_model: str | None = None
    embedding_read_version: int | None = None

    # ---- Embedding backfill ----
    # Records claimed per keyset page, and texts per provider embed call. The
    # embed batch is bounded well under Azure's 2048-inputs-per-request limit.
    embedding_backfill_page_size: int = 500
    embedding_backfill_batch_size: int = 128

    # ---- OAuth ----
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str | None = None
    google_gmail_redirect_uri: str | None = None
    slack_client_id: str | None = None
    slack_client_secret: str | None = None
    slack_redirect_uri: str | None = None
    # Where OAuth callbacks bounce the browser after finishing the
    # code exchange. This is a FRONTEND URL (not a Google redirect_uri —
    # that stays pointed at the backend callback so client_secret stays
    # server-side). Callback appends `?connected=<provider>` on success
    # or `?error=<code>` on failure so the frontend can toast.
    frontend_url: str = "http://localhost:3000"

    # ---- Background tasks ----
    celery_broker_url: str = "redis://localhost:6379/2"
    celery_result_backend: str = "redis://localhost:6379/3"
    # When True, POST /actions/{id}/approve schedules an INLINE execution
    # pass (asyncio.create_task) alongside the Celery enqueue. This lets
    # meeting scheduling work end-to-end WITHOUT a running Celery worker
    # — the approve endpoint returns immediately, and the execution
    # (real Google Calendar events.insert) runs in the same FastAPI
    # process seconds later. Suitable for local dev / demos. In
    # production, run a real Celery worker and set this to False so
    # retry / rate-limiting / dead-letter policies are honoured.
    execution_inline_fallback: bool = True

    # ---- Feature flags ----
    enable_google_auth: bool = True
    enable_ai_caching: bool = True

    # ---- Dev ----
    reload: bool = True
    show_docs: bool = True
    log_level: str = "INFO"

    # ---- Environment ----
    # Validated against the Environment enum below — an unknown value is a startup
    # error, never a silent downgrade to permissive behaviour.
    environment: Environment = Environment.DEVELOPMENT

    # ---- Database pool ----
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_recycle_seconds: int = 300  # Neon drops idle conns ~5m; recycle below that
    db_pool_timeout_seconds: int = 30
    # Server-side cap: a single statement may not run longer than this. Prevents a
    # runaway query from pinning a pooled connection forever (reliability at scale).
    db_statement_timeout_seconds: float = 30.0

    # ---- Redis pool ----
    redis_max_connections: int = 50
    redis_socket_timeout_seconds: float = 5.0

    # ---- Login brute-force throttle (IDENTITY-003) ----
    # Lock an email/IP after N failed attempts; the lock clears after the window (sliding
    # — extended on each new failure). Redis-backed; fails OPEN if Redis is unavailable.
    login_max_attempts: int = 5
    login_lockout_seconds: int = 15 * 60

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
    # Redis stream is a best-effort MIRROR for out-of-process consumers + recent
    # audit — NOT the replay source of truth (that is the append-only domain tables
    # in Postgres). Bounded so it can never grow unbounded.
    event_stream_maxlen: int = 10000

    # ---- Metrics ----
    metrics_enabled: bool = True

    @property
    def effective_redis_cache_url(self) -> str:
        return self.redis_cache_url or self.redis_url

    @property
    def effective_redis_events_url(self) -> str:
        return self.redis_events_url or self.redis_url

    @property
    def jwt_is_asymmetric(self) -> bool:
        return self.algorithm.upper().startswith(("RS", "ES", "PS"))

    @property
    def jwt_signing_key(self) -> str:
        """Private key for asymmetric algorithms; the shared secret for HS*."""
        if self.jwt_is_asymmetric:
            if not self.jwt_private_key:
                raise ValueError(f"{self.algorithm} requires JWT_PRIVATE_KEY")
            return self.jwt_private_key
        return self.secret_key

    @property
    def jwt_verify_key(self) -> str:
        """Public key for asymmetric algorithms; the shared secret for HS*."""
        if self.jwt_is_asymmetric:
            if not self.jwt_public_key:
                raise ValueError(f"{self.algorithm} requires JWT_PUBLIC_KEY")
            return self.jwt_public_key
        return self.secret_key

    @field_validator("environment", mode="before")
    @classmethod
    def _normalize_environment(cls, value: object) -> object:
        """Accept ` Production ` / `PRODUCTION`, reject `prod`.

        Case and surrounding whitespace are deployment noise and are forgiven.
        Anything else is rejected with the allowed values named, because the
        alternative — guessing — is what made this fail open. Notably `prod` is
        NOT accepted: silently treating an abbreviation as production would be
        the same class of guess in the opposite direction.
        """
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        allowed = {e.value for e in Environment}
        if normalized not in allowed:
            raise ValueError(
                f"ENVIRONMENT={value!r} is not recognised. "
                f"Use one of: {', '.join(sorted(allowed))}. "
                "This gates encryption enforcement, entitlements and the "
                "production config checks — it must never be guessed."
            )
        return normalized

    @property
    def is_production(self) -> bool:
        """True only in production.

        Compares by VALUE rather than enum identity. The field is validated at
        startup so it can only hold one of three values, but pydantic does not
        re-validate plain attribute assignment — and tests monkeypatch this
        attribute with a raw string. An identity check would quietly report
        "not production" for `settings.environment = "production"`, which is the
        same fail-open this enum exists to remove, just moved to runtime.
        """
        env = self.environment
        value = env.value if isinstance(env, Environment) else str(env).strip().lower()
        return value == Environment.PRODUCTION.value

    def relaxed_controls(self) -> list[str]:
        """Security/billing controls NOT enforced in this environment.

        Surfaced at startup so a machine running with development defaults says
        so out loud. An unset ENVIRONMENT still defaults to development — the
        remaining fail-open path — and this is what makes that visible instead
        of silent. Empty in production.
        """
        if self.is_production:
            return []
        return [
            "production config safety checks (SECRET_KEY strength, DEBUG, /docs)",
            "field-encryption fail-fast (PII may be stored unencrypted if keys are unset)",
            "OAuth-token-encryption fail-fast (tokens may be stored plaintext if keys are unset)",
            "organization entitlement enforcement (every capability granted)",
            "capability guard on API routes",
            "plan defaults (new orgs get ENTERPRISE, not FREE)",
        ]

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
