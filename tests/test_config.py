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
