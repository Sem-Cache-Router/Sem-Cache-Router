# Lifespan wiring tests for Tier 2.
#
# The lifespan builds what is missing from app.state and leaves alone anything
# a test put there. These pin how Tier 2 fits that pattern without ever loading
# the real embedding weights or reaching a Chroma server.

from __future__ import annotations

from typing import Any

import pytest
from fakeredis import aioredis
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings
from app.providers.stub import StubProvider


def build_app(settings: Settings, fake_redis: aioredis.FakeRedis, **state: Any) -> Any:
    """An app with fakes seeded, plus any extra state the test supplies."""
    app = main.create_app()
    app.state.settings = settings
    app.state.redis = fake_redis
    app.state.provider = StubProvider()
    for name, value in state.items():
        setattr(app.state, name, value)
    return app


def test_tier_two_is_absent_when_switched_off(
    settings: Settings, fake_redis: aioredis.FakeRedis
) -> None:
    """Off means None on app.state, which the route guards on."""
    app = build_app(settings, fake_redis)
    with TestClient(app):
        assert app.state.semantic_cache is None


def test_tier_two_is_built_when_switched_on(
    settings: Settings, fake_redis: aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On means the lifespan builds it, once, from the settings."""
    built: list[Settings] = []
    sentinel = object()

    async def fake_build(received: Settings) -> Any:
        built.append(received)
        return sentinel

    monkeypatch.setattr(main, "build_semantic_cache", fake_build)
    enabled = settings.model_copy(update={"semantic_cache_enabled": True})
    app = build_app(enabled, fake_redis)

    with TestClient(app):
        assert app.state.semantic_cache is sentinel
    assert built == [enabled]


def test_a_semantic_cache_supplied_by_a_test_is_left_alone(
    settings: Settings, fake_redis: aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test seam: seeded state is never replaced by the lifespan."""

    async def must_not_build(received: Settings) -> Any:
        raise AssertionError("the lifespan rebuilt a semantic cache it was given")

    monkeypatch.setattr(main, "build_semantic_cache", must_not_build)
    supplied = object()
    enabled = settings.model_copy(update={"semantic_cache_enabled": True})
    app = build_app(enabled, fake_redis, semantic_cache=supplied)

    with TestClient(app):
        assert app.state.semantic_cache is supplied


def test_the_chroma_factory_connects_lazily_with_the_configured_endpoint(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building the factory connects to nothing; calling it uses the parsed URL."""
    import chromadb

    calls: list[dict[str, Any]] = []

    def fake_http_client(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "client"

    monkeypatch.setattr(chromadb, "HttpClient", fake_http_client)
    configured = settings.model_copy(update={"chroma_host": "https://vectors.example.com:9000"})

    factory = main.chroma_client_factory(configured)
    assert calls == []

    assert factory() == "client"
    assert calls[0]["host"] == "vectors.example.com"
    assert calls[0]["port"] == 9000
    assert calls[0]["ssl"] is True
    assert calls[0]["settings"].anonymized_telemetry is False


async def test_building_the_cache_loads_the_embedder_first(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weights load at startup, so the first request does not pay for them."""
    loaded: list[str] = []

    class FakeEmbedder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            loaded.append(self.model_name)

    monkeypatch.setattr(main, "OnnxEmbedder", FakeEmbedder)
    cache = await main.build_semantic_cache(settings)

    assert loaded == [settings.embedding_model]
    assert cache is not None
