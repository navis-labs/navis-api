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

    `is_production` gates four independent controls — production config checks,
    field-encryption fail-fast, OAuth-token-encryption fail-fast, and the default
    plan a new org is provisioned with. It was previously a free-form string
    compared with `== "production"`, so `ENVIRONMENT=prod`, a trailing space, or a
    misspelling silently took the PERMISSIVE branch of all of them at once:
    credentials stored unencrypted and the startup safety net skipped — with no
    error and no log line.

    It deliberately no longer gates authorization. Entitlement resolution and the
    capability guard used to consult it and grant everything outside production,
    which made a single missing env var equivalent to disabling the capability
    boundary. Authorization now depends only on org state, never on deployment.

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
    redis_url: str = "redis://localhost:6380/0"
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
    # Microsoft Graph capture (Outlook mail + calendar, OneDrive). One set of
    # caps for the whole suite rather than three, because all three connectors
    # drive the SAME mechanism — a Graph delta enumeration — and the numbers
    # that matter are properties of that mechanism, not of the resource.
    #
    # Mirrors the Google caps in intent: the onboarding backfill is bounded and
    # LOUD when it hits the cap (never silently truncated), and exceeding the
    # page-drain cap raises rather than advancing a partial cursor, because a
    # cursor that jumps past undrained pages skips those items forever.
    #
    # The mail lookback matches gmail_onboard_lookback_days so a founder who
    # connects Outlook and a founder who connects Gmail see the same horizon —
    # a different window would make the two suites disagree about what counts
    # as a slipping commitment on day one.
    microsoft_onboard_item_cap: int = 500
    microsoft_onboard_lookback_days: int = 30
    microsoft_max_sync_pages: int = 200
    # How far forward a calendar full sync reaches. Graph's calendarView needs
    # an explicit end; Google's does not, which is why this has no Google twin.
    microsoft_calendar_window_days: int = 60
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

    # ---- Microsoft 365 / Entra ID OAuth ----
    # One multi-tenant app registration serving TWO distinct consent screens,
    # which is why there are two redirect URIs rather than one:
    #
    #   microsoft_redirect_uri         the CONNECTOR grant (Mail/Calendar/Files)
    #   microsoft_signin_redirect_uri  the IDENTITY grant (openid/profile/email)
    #
    # Collapsing them would put mailbox scopes on the sign-in screen, which is
    # the single worst moment to ask for them — the frontend keeps the two
    # behind separate feature flags for exactly this reason.
    #
    # `microsoft_tenant` is the authority segment. "common" accepts both work/
    # school and personal accounts; set it to a tenant GUID to lock the
    # deployment to one organisation. Note that personal accounts are the case
    # where the `email` claim can be UNVERIFIED — see identity/microsoft_auth.
    #
    # Every value defaults to None, so an environment that has not configured
    # Microsoft behaves exactly as it does today: the connect route reports
    # "not configured" instead of building a half-formed authorization URL.
    microsoft_client_id: str | None = None
    microsoft_client_secret: str | None = None
    microsoft_tenant: str = "common"
    microsoft_redirect_uri: str | None = None
    microsoft_signin_redirect_uri: str | None = None

    # ---- Gmail push (authenticated Cloud Pub/Sub) ----
    # Defined HERE rather than read straight from os.environ so they load from .env
    # like the rest of config AND can be validated at startup. A real exported env
    # var still wins (pydantic-settings reads the environment first), so production
    # secret managers are unaffected.
    #
    # `topic` is what `GmailConnector.create_watch` registers with Google; the other
    # three are what `verify_pubsub_request` checks on every push. They are a SET:
    # registering a topic without pinning the audience and the pushing identity would
    # open an authenticated-looking endpoint that verifies almost nothing, so
    # `_enforce_production_safety` refuses that combination.
    google_pubsub_topic: str | None = None       # projects/<p>/topics/<t>
    google_pubsub_audience: str | None = None    # the push endpoint URL Google signs for
    google_pubsub_sa_email: str | None = None    # push service account (email claim)
    google_pubsub_sub: str | None = None         # push service account (stable OIDC `sub`)
    # Where OAuth callbacks bounce the browser after finishing the
    # code exchange. This is a FRONTEND URL (not a Google redirect_uri —
    # that stays pointed at the backend callback so client_secret stays
    # server-side). Callback appends `?connected=<provider>` on success
    # or `?error=<code>` on failure so the frontend can toast.
    frontend_url: str = "http://localhost:3000"

    # ---- Background tasks ----
    celery_broker_url: str = "redis://localhost:6380/2"
    celery_result_backend: str = "redis://localhost:6380/3"
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
    # Liveness-check every pooled connection before handing it out. It costs a
    # FULL EXTRA ROUND TRIP per checkout, which is invisible on a local database
    # and dominant on a remote one: measured against the production Neon
    # endpoint, `SELECT 1` on a fresh session went 2689ms -> 1457ms with this
    # off — 1232ms, 46% of the checkout cost, paid by every HTTP request.
    #
    # `db_pool_recycle_seconds` (above) is the real protection: it retires
    # connections before Neon's ~5m idle drop, so pre-ping was guarding a window
    # recycling already closes. Left configurable because a deployment with an
    # unreliable network path may legitimately want to trade the latency back.
    db_pool_pre_ping: bool = False
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

    # ---- Per-ORGANIZATION rate limits on expensive routes (T-23) ----
    # Tenant isolation for shared compute: `/chat` alone embeds a query, runs a
    # full-scan vector search over the org's corpus, folds state and calls an LLM,
    # and had NO throttle of any kind — one organization could exhaust the DB pool,
    # the Redis pool, the LLM quota and the API event loop for everyone else.
    #
    # Counted per (bucket, organization) in a fixed window, so the limit a tenant
    # spends is its own. Deliberately NOT tiered by plan: the product contract does
    # not define per-plan request quotas, and inventing them here would put a
    # commercial policy in an availability control. These are capacity guards.
    #
    # Generous by default — a demo must not trip them — and every value is
    # overridable per deployment.
    rate_limit_enabled: bool = True
    rate_limit_window_seconds: int = 60
    # Conversational AI: retrieval + generation, the most expensive request.
    rate_limit_chat_per_window: int = 30
    # Whole-pipeline recomputes (brief build, action proposal, intelligence
    # projection) — heavier than chat but driven by a button, not a conversation.
    rate_limit_pipeline_per_window: int = 12
    # Context assembly + connector-backed reads: cheaper, but still vector search
    # and/or an outbound provider call per request.
    rate_limit_read_per_window: int = 120

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
            # Entitlement ENFORCEMENT is no longer relaxed anywhere — the gate is the
            # same code in every environment. What is still relaxed is the plan a new
            # org is seeded with, which grants breadth through real entitlement rows
            # the gate then evaluates normally.
            "plan defaults / seeded entitlements (new orgs get ENTERPRISE, not FREE)",
            # Likewise not an authorization relaxation: the inline pass runs the
            # SAME claim/authorize/execute code the worker runs, just on the API
            # event loop and outside the rate limiter and retry budget. Named here
            # because a developer should know side effects can fire from the web
            # process without a worker running.
            "execution worker isolation (approved actions may run inline on the API loop)",
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
        # A wildcard origin is never a deliberate production choice. Starlette
        # returns `Access-Control-Allow-Origin: *` for it, which makes every
        # unauthenticated response readable by any site on the internet, and with
        # `allow_headers=["*"]` a preflight for `Authorization` also succeeds — so
        # any origin may call the API on behalf of anyone holding a token. Verified
        # empirically before adding this check.
        if any(o == "*" for o in self.cors_origins_list):
            problems.append(
                "CORS_ORIGINS must not contain '*' — name the frontend origins explicitly"
            )
        if not self.cors_origins_list:
            problems.append("CORS_ORIGINS must name at least one frontend origin")
        # SIGNAL-CRYPTO-002. Production storage was verified free of legacy
        # plaintext in TASK 6.4.1, and this switch is what keeps it that way: with
        # it off, an untagged token is silently trusted again and encryption-at-rest
        # can regress unnoticed (which is exactly how it regressed the first time).
        #
        # Deliberately coupled to ENVIRONMENT rather than defaulted to True: turning
        # it on where tokens have NOT been migrated would break those integrations,
        # so it must stay an explicit per-environment decision. Requiring it here
        # means the operator who sets ENVIRONMENT=production is told, at that
        # moment, that this is the other half of the same change.
        if not self.signals_reject_legacy_plaintext:
            problems.append(
                "SIGNALS_REJECT_LEGACY_PLAINTEXT must be true in production "
                "(run navis.signals.token_backfill.reencrypt_legacy_tokens first, "
                "then confirm zero plaintext tokens)"
            )
        # EXECUTION_INLINE_FALLBACK. The setting's own docstring already states the
        # contract — "In production, run a real Celery worker and set this to
        # False" — and nothing enforced it, while the default is True. Left on, the
        # approve endpoint schedules `asyncio.create_task(_execute())` and the
        # provider call (a real Gmail send, a real Calendar write) runs on the API
        # process's event loop, sharing it with every other tenant's requests.
        #
        # It is not only a latency concern. The inline pass bypasses the worker's
        # operational envelope: the per-provider rate limiter, the retry budget,
        # dead-lettering and stale-execution recovery all live on the Celery path.
        # A tenant approving a batch would run unthrottled side effects inside the
        # web process, and a process restart mid-flight loses the pass with no
        # lease to recover from.
        #
        # A startup refusal rather than a silent override, for the same reason as
        # every other check here: a production box that quietly behaved like
        # development is exactly the failure this function exists to prevent.
        if self.execution_inline_fallback:
            problems.append(
                "EXECUTION_INLINE_FALLBACK must be false in production "
                "(run a Celery worker; the inline pass bypasses rate limiting, "
                "retries and dead-lettering, and executes on the API event loop)"
            )
        # Gmail push is all-or-nothing. Registering a Pub/Sub topic makes Google POST
        # to the public webhook, and that endpoint can only authenticate the push if it
        # knows BOTH the audience Google signs for and which service account is allowed
        # to send it. Half-configured is the dangerous state — it looks provisioned
        # while pinning nothing — so it is a startup error rather than a runtime
        # surprise. Configuring none of them is fine: the endpoint then refuses every
        # request (503) and ingestion runs on the polling schedule.
        if self.google_pubsub_topic:
            if not self.google_pubsub_audience:
                problems.append(
                    "GOOGLE_PUBSUB_TOPIC is set but GOOGLE_PUBSUB_AUDIENCE is not — "
                    "push tokens could not be audience-verified"
                )
            if not (self.google_pubsub_sa_email or self.google_pubsub_sub):
                problems.append(
                    "GOOGLE_PUBSUB_TOPIC is set but neither GOOGLE_PUBSUB_SA_EMAIL nor "
                    "GOOGLE_PUBSUB_SUB is — the pushing service account would not be pinned"
                )
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


class ConfigurationError(RuntimeError):
    """Startup configuration is invalid. Carries the reasons, never the values."""


@lru_cache
def get_settings() -> Settings:
    """Build the settings singleton, converting a validation failure into a message
    that CANNOT carry configuration values.

    pydantic's `ValidationError` renders `input_value=` alongside the reason — the
    whole settings dict, truncated to a fixed head and tail. That truncation is a
    length accident, not a redaction: which fields land inside the window depends on
    field order, so a future field could put a key or password into a message that
    is printed by the process supervisor, shipped to log aggregation and pasted into
    tickets. Re-raising with only the reason strings removes the possibility rather
    than relying on where the ellipsis falls.
    """
    from pydantic import ValidationError

    try:
        return Settings()
    except ValidationError as exc:
        reasons = "; ".join(
            str(e.get("msg", "")).replace("Value error, ", "") for e in exc.errors()
        )
        raise ConfigurationError(f"invalid configuration: {reasons}") from None


settings = get_settings()
