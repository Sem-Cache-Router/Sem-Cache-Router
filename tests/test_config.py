# Settings tests.
#
# Configuration is validated at startup so that a bad value fails loudly before
# the first request rather than surfacing later as a confusing runtime error.
# These tests pin that behaviour for the Tier 2 settings.

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip SEMCACHE_ variables so a developer's shell cannot change the result."""
    import os

    for name in list(os.environ):
        if name.startswith("SEMCACHE_"):
            monkeypatch.delenv(name)


def settings(**overrides: Any) -> Settings:
    """Build Settings from defaults plus overrides, ignoring any local .env file."""
    # _env_file is accepted at runtime by pydantic-settings but is not part of
    # the generated signature, so the type checker cannot see it.
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_semantic_cache_is_off_by_default() -> None:
    """Off by default, so nothing changes for anyone running without Chroma."""
    assert settings().semantic_cache_enabled is False


def test_collection_name_carries_a_schema_version() -> None:
    """The default name versions the stored shape, like the Tier 1 key prefix."""
    assert settings().chroma_collection == "semcache-v1"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:8000", ("localhost", 8000, False)),
        ("http://chromadb:8000", ("chromadb", 8000, False)),
        ("https://vectors.example.com", ("vectors.example.com", 443, True)),
        ("http://vectors.example.com", ("vectors.example.com", 80, False)),
    ],
)
def test_chroma_endpoint_is_derived_from_one_url(
    url: str, expected: tuple[str, int, bool]
) -> None:
    """Host, port and TLS come from the single chroma_host setting."""
    assert settings(chroma_host=url).chroma_endpoint == expected


@pytest.mark.parametrize("url", ["localhost:8000", "ftp://chromadb:8000", "not a url", "http://"])
def test_unusable_chroma_host_is_rejected_at_startup(url: str) -> None:
    """A typo should fail when the process starts, not on the first Tier 2 lookup."""
    with pytest.raises(ValidationError):
        settings(chroma_host=url)


@pytest.mark.parametrize("name", ["ab", "-leading-hyphen", "trailing.", "has space", "x" * 513])
def test_collection_name_chroma_would_reject_is_rejected_here(name: str) -> None:
    """Chroma refuses these names. Catching them at startup beats failing on first write."""
    with pytest.raises(ValidationError):
        settings(chroma_collection=name)


def test_reconnect_cooldown_must_be_positive() -> None:
    """A zero cooldown would retry a dead Chroma on every request."""
    with pytest.raises(ValidationError):
        settings(chroma_reconnect_seconds=0)


def test_the_limiter_is_off_by_default() -> None:
    """Off by default, so nobody is throttled by a budget they never configured."""
    assert settings().limiter_enabled is False


def test_the_limiter_reserves_a_reply_allowance_by_default() -> None:
    """Without one, a request that sets no max_tokens would be admitted for its prompt alone."""
    assert settings().limiter_default_completion_tokens == 256


@pytest.mark.parametrize("value", [0, -1])
def test_the_reply_allowance_must_be_positive(value: int) -> None:
    """A zero allowance would admit an open ended reply at no cost."""
    with pytest.raises(ValidationError):
        settings(limiter_default_completion_tokens=value)


@pytest.mark.parametrize("name", ["bucket_capacity", "bucket_refill_rate"])
def test_a_bucket_must_have_room_and_refill(name: str) -> None:
    """A bucket with no capacity or no refill rate would refuse every request, or refuse forever."""
    with pytest.raises(ValidationError):
        settings(**{name: 0})


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("http://collector:4318", "http://collector:4318/v1/traces"),
        ("http://collector:4318/", "http://collector:4318/v1/traces"),
        ("http://collector:4318/v1/traces", "http://collector:4318/v1/traces"),
        ("https://otel.example.com/custom/path", "https://otel.example.com/custom/path"),
    ],
)
def test_a_bare_collector_address_is_pointed_at_the_traces_path(given: str, expected: str) -> None:
    """The exporter posts to a code-supplied endpoint verbatim, so a bare address
    would send every span to the root and lose them all silently. An explicit
    path is left exactly as given."""
    assert settings(otel_exporter_endpoint=given).otel_exporter_endpoint == expected


def test_tracing_has_no_endpoint_by_default() -> None:
    """Nothing is exported unless someone configures where to."""
    assert settings().otel_exporter_endpoint is None


@pytest.mark.parametrize("url", ["collector:4318", "grpc://collector:4317", "not a url"])
def test_an_unusable_otel_endpoint_is_rejected_at_startup(url: str) -> None:
    """A typo should fail when the process starts, not as spans that never arrive."""
    with pytest.raises(ValidationError):
        settings(otel_exporter_endpoint=url)


def test_routing_is_disabled_by_default() -> None:
    """Routing off by default means existing deployments are unaffected."""
    assert settings().routing_enabled is False


def test_routing_requires_cheap_model_when_enabled() -> None:
    """Enabling routing without a cheap model is a startup error, not a silent no-op."""
    with pytest.raises(ValidationError, match="routing_cheap_model is required"):
        settings(routing_enabled=True)


def test_routing_enabled_with_cheap_model_is_valid() -> None:
    """The correct configuration is accepted without raising."""
    s = settings(routing_enabled=True, routing_cheap_model="gpt-4o-mini")
    assert s.routing_enabled is True
    assert s.routing_cheap_model == "gpt-4o-mini"


def test_routing_simple_max_tokens_defaults_to_512() -> None:
    """The default ceiling matches what the .env.example documents."""
    assert settings().routing_simple_max_tokens == 512


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_routing_simple_max_tokens_rejects_non_positive(bad: int) -> None:
    """Zero or negative tokens cannot be a ceiling for a prompt."""
    with pytest.raises(ValidationError):
        settings(routing_simple_max_tokens=bad)


def test_metrics_latency_window_defaults_to_ten_thousand() -> None:
    """The default window is large enough for a typical benchmark run."""
    assert settings().metrics_latency_window == 10_000


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_metrics_latency_window_rejects_non_positive(bad: int) -> None:
    """A window of zero or less would make every sample invisible at startup."""
    with pytest.raises(ValidationError):
        settings(metrics_latency_window=bad)


# --- providers -------------------------------------------------------------------------------


def test_failover_is_opt_in() -> None:
    """No secondary by default, so the default stub primary demands no vendor's key."""
    assert settings().secondary_provider is None
    assert settings().secondary_model is None


def test_a_secondary_provider_can_be_chosen() -> None:
    chosen = settings(primary_provider="openai", secondary_provider="anthropic")
    assert chosen.secondary_provider == "anthropic"


def test_a_provider_cannot_fail_over_to_itself() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        settings(primary_provider="openai", secondary_provider="openai")


def test_the_adapters_default_to_the_vendors_endpoints() -> None:
    assert settings().openai_base_url == "https://api.openai.com/v1"
    assert settings().anthropic_base_url == "https://api.anthropic.com/v1"


@pytest.mark.parametrize("name", ["openai_base_url", "anthropic_base_url"])
def test_a_trailing_slash_on_a_provider_url_is_dropped(name: str) -> None:
    """The adapters append a path, and a double slash is rejected by some endpoints."""
    assert getattr(settings(**{name: "http://proxy.internal:9000/v1/"}), name) == (
        "http://proxy.internal:9000/v1"
    )


@pytest.mark.parametrize("name", ["openai_base_url", "anthropic_base_url"])
@pytest.mark.parametrize("url", ["api.openai.com/v1", "ftp://proxy/v1", "not a url", "http://"])
def test_an_unusable_provider_url_is_rejected_at_startup(name: str, url: str) -> None:
    with pytest.raises(ValidationError):
        settings(**{name: url})


def test_the_provider_timeout_is_generous_by_default() -> None:
    """A long completion legitimately takes tens of seconds."""
    assert settings().provider_timeout_seconds == 30.0


@pytest.mark.parametrize("value", [0, -1])
def test_a_provider_timeout_that_allows_no_time_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        settings(provider_timeout_seconds=value)


def test_the_example_environment_loads_and_asks_for_nothing_extra() -> None:
    """scripts/setup.sh copies .env.example into every fresh checkout.

    If it ever set a secondary provider, every new checkout would refuse to start
    for want of that vendor's key. Loading it here keeps that from happening
    unnoticed.
    """
    loaded = Settings(_env_file=".env.example")  # type: ignore[call-arg]

    assert loaded.secondary_provider is None
    assert loaded.primary_provider == "stub"
