# Settings, read from the environment.
#
# Holds a pydantic-settings Settings class, and no constants anywhere else in
# the codebase.
#
# Why every value lives here: a benchmark has to sweep thresholds, TTLs and
# bucket sizes without a code change, or a run is not reproducible from a
# committed configuration.

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["stub", "openai", "anthropic"]


class Settings(BaseSettings):
    """Runtime configuration, read from the environment and never hardcoded."""

    model_config = SettingsConfigDict(
        env_prefix="SEMCACHE_",
        env_file=".env",
        env_file_encoding="utf-8",
        # Mandatory, not cosmetic. The dotenv file also carries unprefixed
        # provider credentials, and pydantic-settings reads the whole file, so
        # forbidding extras would make a valid .env fail to load.
        extra="ignore",
    )

    # State stores. Defaults point at localhost for a developer running the
    # gateway outside Compose; the Compose file overrides both with service
    # names on the internal network. Chroma's own default port is 8000, which
    # the gateway already holds when both run on one machine, so a local Chroma
    # is expected on 8001.
    redis_url: str = "redis://localhost:6379/0"
    chroma_host: str = "http://localhost:8001"

    # Cache behaviour. The threshold is bounded because a value outside [0, 1]
    # is not a cosine similarity, and catching that at startup is cheaper than
    # discovering it as a cache that never hits.
    similarity_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=3600, gt=0)

    # Admission control. Off by default, like Tier 2: with it off the gateway
    # admits everything exactly as it did before the limiter existed, so no one
    # running locally or in the test suite is throttled by a budget they never set.
    limiter_enabled: bool = False
    bucket_capacity: int = Field(default=100_000, gt=0)
    bucket_refill_rate: float = Field(default=1000.0, gt=0)

    # Provider tokens reserved for the reply when a request does not say how long
    # it may be. The true reply length is unknown until after the call, so the
    # estimate has to assume something, and assuming zero would admit a caller
    # for far less than a long answer will cost.
    limiter_default_completion_tokens: int = Field(default=256, gt=0)

    # Providers. The default is the stub rather than OpenAI so that the gateway
    # starts and serves with no credentials present, which is what makes the
    # test suite and a local demo runnable offline. A real deployment sets
    # SEMCACHE_PRIMARY_PROVIDER=openai in its environment.
    primary_provider: ProviderName = "stub"

    # The provider to fail over to when the primary is unhealthy, or None for no
    # failover. Opt in rather than defaulting to a real vendor: with the default
    # primary being the stub, a default secondary of anthropic would demand a key
    # nobody configured, and an unconfigured failover that fails silently is worse
    # than none that says so. Setting it to the primary is rejected, because a
    # provider cannot fail over to itself.
    secondary_provider: ProviderName | None = None

    # The model the secondary is called with. The caller asks for a model by name,
    # and one vendor's name means nothing to another: sending gpt-4o-mini to
    # Anthropic is a 404, which would turn a failover into an error. None means
    # the secondary's own default model.
    secondary_model: str | None = None

    # Circuit breaker.
    breaker_failure_threshold: int = Field(default=5, gt=0)
    breaker_window_seconds: int = Field(default=60, gt=0)
    breaker_base_backoff: float = Field(default=1.0, gt=0)
    breaker_backoff_ceiling: float = Field(default=60.0, gt=0)

    # Embedding.
    embedding_model: str = "all-MiniLM-L6-v2"

    # Tier 2. Off by default, so the gateway still starts and serves when no
    # vector store is running, which is the situation for the test suite and
    # for anyone running it locally on the stub provider. Compose turns it on.
    semantic_cache_enabled: bool = False

    # Refuse a semantic hit when the cached prompt and the new one disagree on
    # negation ("open" against "not open"), which the embedding model scores as
    # near identical. Off by default: it trades some correct hits for provider
    # calls, and the benchmark measures both sides. See app/cache/negation_guard.py.
    semantic_negation_guard: bool = False

    # How often expired Tier 2 entries are deleted, in seconds, or 0 to never.
    # Expired entries are never served either way, because every query filters
    # them out; this only reclaims the space, so a long interval costs storage
    # and nothing else.
    eviction_interval_seconds: float = Field(default=300.0, ge=0)

    # The most Tier 2 entries kept. Each eviction pass that finds more deletes the
    # oldest, by when they were stored, until this many are left. Every entry
    # carries its vector, prompt and the whole response, so an unbounded
    # collection grows the vector store for as long as new questions arrive.
    tier2_max_entries: int = Field(default=100_000, gt=0)

    # The collection carries a schema version for the same reason Tier 1 keys
    # do: if the stored shape changes, a new name orphans the old entries rather
    # than requiring a migration. Chroma rejects names outside this pattern, so
    # it is checked at startup instead of on the first write.
    chroma_collection: str = Field(
        default="semcache-v1",
        min_length=3,
        max_length=512,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]$",
    )

    # How long to wait before trying Chroma again after a failed connection.
    # Without a cooldown, every request during an outage would pay for a
    # connection attempt that is going to fail.
    chroma_reconnect_seconds: float = Field(default=30.0, gt=0)

    # How long one call to Chroma may take before it is treated as an outage. A
    # vector store that has hung raises nothing, so without a bound a request
    # would wait on it for as long as it stayed hung, and the reconnect cooldown
    # above would never get the chance to start. Two seconds is far beyond a
    # healthy local query and well short of a caller giving up on the request.
    chroma_timeout_seconds: float = Field(default=2.0, gt=0)

    # OpenTelemetry tracing. Disabled by default so the gateway starts with no
    # collector present and no errors.  The endpoint is where the OTLP exporter
    # sends spans; None means tracing is a no-op even if enabled, which is the
    # safe behaviour when no collector is reachable.
    tracing_enabled: bool = False
    otel_exporter_endpoint: str | None = None

    # Observability. Prompt text is caller data, so it reaches the logs only
    # when this is switched on deliberately in a development configuration.
    log_prompts: bool = False

    # Metrics. The latency lists are capped at this many samples; exact
    # percentiles are then computed over the most recent N records rather than
    # since process start.  A larger window gives more accurate long-run
    # percentiles; a smaller one reduces steady-state memory use.  Must be
    # greater than zero because a window of zero would make every sample
    # invisible.
    metrics_latency_window: int = Field(default=10_000, gt=0)

    # Caller authentication. The key is the identity a token budget is charged
    # against, so it cannot be optional once the limiter exists. Until then
    # there is nothing to authorise against, and any non-empty bearer token is
    # accepted rather than validated against a key store that does not exist.
    require_api_key: bool = True

    # The key for the cache admin endpoints (stats and flush). Unset means the
    # endpoints do not exist and answer 404, because emptying the cache in the
    # middle of a benchmark is something only an operator should be able to do.
    # It is separate from caller keys so that no caller can flush the cache.
    admin_api_key: str | None = Field(default=None, repr=False)

    # The largest request body accepted, in bytes. A chat request is a short JSON
    # document, so 1 MiB is far above any real one while still bounding what a
    # single caller can make the gateway read, embed and forward to a provider
    # that bills by the token.
    max_request_bytes: int = Field(default=1_048_576, gt=0)

    # Provider credentials. Read unprefixed, because that is the name the
    # provider SDKs and every deployment guide already use. Optional because
    # the stub needs neither, and absent means the corresponding adapter
    # refuses to construct rather than failing at the first request.
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "SEMCACHE_OPENAI_API_KEY"),
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "SEMCACHE_ANTHROPIC_API_KEY"),
    )

    # Where the adapters send requests. Settings rather than constants so a
    # deployment can go through a proxy or a compatible endpoint, and so a test or
    # a demo can point the real adapters at a local stand in without a single
    # patched module global.
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com/v1"

    # How long one provider call may take before it counts as a failure, which is
    # what lets a hung provider trip the breaker and fail over instead of holding
    # the request. Generous, because a long completion legitimately takes tens of
    # seconds; a limit tuned to the median would fail the requests that cost most.
    provider_timeout_seconds: float = Field(default=30.0, gt=0)

    # Model routing. When enabled, single-turn short prompts (no code fence, under
    # routing_simple_max_tokens) are sent to the cheap model instead of whatever
    # model the caller requested. The cache key still uses the requested model (see
    # section 33.7 in the PRD), so routing changes do not invalidate existing entries.
    # routing_cheap_model is required when routing_enabled=True, and rejected at
    # startup otherwise, so wiring errors surface before the first request.
    routing_enabled: bool = False
    routing_cheap_model: str | None = None
    routing_simple_max_tokens: int = Field(default=512, gt=0)

    @field_validator("chroma_host")
    @classmethod
    def _chroma_host_is_a_url(cls, value: str) -> str:
        """Reject a Chroma address that cannot be split into host and port.

        Checked here so a typo fails at startup with a clear message, rather
        than as a connection error on the first Tier 2 lookup.
        """
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("chroma_host must be an http or https URL, such as http://chromadb:8000")
        return value

    @field_validator("otel_exporter_endpoint")
    @classmethod
    def _otel_endpoint_reaches_the_traces_path(cls, value: str | None) -> str | None:
        """Require an http URL, and point a bare collector address at /v1/traces.

        The OTLP/HTTP exporter posts to an endpoint passed in code exactly as
        given; only the environment variable form gets /v1/traces appended. So
        the usual collector address, http://collector:4318, would send every
        span to the root, where the collector rejects it, and tracing would fail
        with nothing visible. A path is added only when none was given, so an
        explicit one is never second guessed.
        """
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(
                "otel_exporter_endpoint must be an http or https URL, such as http://collector:4318"
            )
        if parsed.path in ("", "/"):
            return value.rstrip("/") + "/v1/traces"
        return value

    @field_validator("openai_base_url", "anthropic_base_url")
    @classmethod
    def _provider_url_is_http(cls, value: str) -> str:
        """Require an http or https URL, without a trailing slash.

        The adapters append a path to it, so a trailing slash would produce a
        double slash that some endpoints reject. Checked here so a typo fails at
        startup, not on the first request that needs a provider.
        """
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("a provider base URL must be an http or https URL")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _routing_cheap_model_required_when_enabled(self) -> Settings:
        """Require routing_cheap_model when routing_enabled=True.

        A misconfigured router silently falls back to the requested model,
        which is misleading. Catching the missing value at startup makes the
        misconfiguration explicit before any request is served.
        """
        if self.routing_enabled and not self.routing_cheap_model:
            raise ValueError(
                "routing_cheap_model is required when routing_enabled=True"
            )
        return self

    @model_validator(mode="after")
    def _secondary_differs_from_primary(self) -> Settings:
        """Refuse a failover target that is the provider it would fail over from."""
        if self.secondary_provider is not None and self.secondary_provider == self.primary_provider:
            raise ValueError("secondary_provider must differ from primary_provider")
        return self

    @property
    def chroma_endpoint(self) -> tuple[str, int, bool]:
        """Return the Chroma host, port and whether to use TLS.

        One URL setting, like redis_url, rather than three separate fields, so the
        Compose file overrides a single value. The Chroma client wants the parts,
        so they are derived here and nowhere else.
        """
        parsed = urlsplit(self.chroma_host)
        ssl = parsed.scheme == "https"
        port = parsed.port or (443 if ssl else 80)
        return str(parsed.hostname), port, ssl


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process wide Settings, cached so the environment is read once.

    A cached accessor rather than a module level instance, because a module
    level instance reads the environment at import time, which is the thing the
    lifespan design forbids, and it cannot be substituted in a test. Tests that
    manipulate the environment must call get_settings.cache_clear().
    """
    return Settings()
