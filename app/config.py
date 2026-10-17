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

from pydantic import AliasChoices, Field, field_validator
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
    # names on the internal network.
    redis_url: str = "redis://localhost:6379/0"
    chroma_host: str = "http://localhost:8000"

    # Cache behaviour. The threshold is bounded because a value outside [0, 1]
    # is not a cosine similarity, and catching that at startup is cheaper than
    # discovering it as a cache that never hits.
    similarity_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=3600, gt=0)

    # Admission control.
    bucket_capacity: int = Field(default=100_000, gt=0)
    bucket_refill_rate: float = Field(default=1000.0, gt=0)

    # Providers. The default is the stub rather than OpenAI so that the gateway
    # starts and serves with no credentials present, which is what makes the
    # test suite and a local demo runnable offline. A real deployment sets
    # SEMCACHE_PRIMARY_PROVIDER=openai in its environment.
    primary_provider: ProviderName = "stub"
    secondary_provider: ProviderName = "anthropic"

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

    # Observability. Prompt text is caller data, so it reaches the logs only
    # when this is switched on deliberately in a development configuration.
    log_prompts: bool = False

    # Caller authentication. The key is the identity a token budget is charged
    # against, so it cannot be optional once the limiter exists. Until then
    # there is nothing to authorise against, and any non-empty bearer token is
    # accepted rather than validated against a key store that does not exist.
    require_api_key: bool = True

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
