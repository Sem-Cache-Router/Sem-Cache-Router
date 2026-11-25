"""Cache admin endpoints: counting and emptying both tiers, behind the admin key."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fakeredis import aioredis
from fastapi.testclient import TestClient

from app.cache.exact_cache import ExactCache
from app.cache.semantic_cache import SemanticCache
from app.config import Settings
from app.main import create_app
from app.providers.stub import StubProvider
from tests.conftest import FakeClock, MappingEmbedder, base_vector, unit_vector_at

ADMIN_KEY = "admin-secret"
ADMIN = {"Authorization": f"Bearer {ADMIN_KEY}"}
CALLER = {"Authorization": "Bearer caller-key"}


def _settings(settings: Settings, admin_key: str | None) -> Settings:
    return settings.model_copy(update={"admin_api_key": admin_key})


def _client(settings: Settings, redis: aioredis.FakeRedis, semantic: SemanticCache | None) -> Any:
    app = create_app()
    app.state.settings = settings
    app.state.redis = redis
    app.state.provider = StubProvider()
    app.state.cache = ExactCache(redis, settings.cache_ttl_seconds)
    app.state.limiter = None
    app.state.semantic_cache = semantic
    return TestClient(app)


def _ask(client: TestClient, content: str) -> None:
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": content}]}
    assert client.post("/v1/chat/completions", json=body, headers=CALLER).status_code == 200


@pytest.fixture
def semantic(
    chroma_factory: Callable[[], Any], mapping_embedder: MappingEmbedder,
    collection_name: str, fake_clock: FakeClock,
) -> SemanticCache:
    return SemanticCache(chroma_factory, mapping_embedder, collection_name=collection_name,
                         ttl_seconds=60, threshold=0.90, clock=fake_clock)


@pytest.fixture
def admin(
    settings: Settings, fake_redis: aioredis.FakeRedis, semantic: SemanticCache,
    mapping_embedder: MappingEmbedder,
) -> Iterator[TestClient]:
    # Unrelated vectors, so the second question is stored rather than served the first.
    mapping_embedder.register("first question", base_vector())
    mapping_embedder.register("second question", unit_vector_at(0.0))
    with _client(_settings(settings, ADMIN_KEY), fake_redis, semantic) as client:
        yield client


def test_stats_count_both_tiers(admin: TestClient) -> None:
    _ask(admin, "first question")
    _ask(admin, "second question")
    response = admin.get("/admin/cache", headers=ADMIN)
    assert response.status_code == 200
    assert response.json() == {"tier1_entries": 2, "tier2_entries": 2, "tier2": "ok"}


async def test_flush_empties_both_tiers_and_nothing_else(
    admin: TestClient, fake_redis: aioredis.FakeRedis
) -> None:
    _ask(admin, "first question")
    _ask(admin, "second question")
    await fake_redis.set("rl:bucket", "42")

    response = admin.delete("/admin/cache", headers=ADMIN)
    assert response.json() == {"tier1_removed": 2, "tier2_removed": 2, "tier2": "ok"}
    assert admin.get("/admin/cache", headers=ADMIN).json()["tier1_entries"] == 0
    assert admin.get("/admin/cache", headers=ADMIN).json()["tier2_entries"] == 0
    assert await fake_redis.get("rl:bucket") == "42"


def test_after_a_flush_the_same_question_is_a_miss_again(admin: TestClient) -> None:
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "again"}]}
    admin.post("/v1/chat/completions", json=body, headers=CALLER)
    admin.delete("/admin/cache", headers=ADMIN)
    status = admin.post("/v1/chat/completions", json=body, headers=CALLER).json()
    assert status["semcache_status"] == "miss"


def test_flushing_an_empty_cache_removes_nothing(admin: TestClient) -> None:
    assert admin.delete("/admin/cache", headers=ADMIN).json() == {
        "tier1_removed": 0, "tier2_removed": 0, "tier2": "ok",
    }


async def test_flush_clears_more_than_one_batch_of_keys(
    admin: TestClient, fake_redis: aioredis.FakeRedis
) -> None:
    for n in range(1203):
        await fake_redis.set(f"semcache:exact:v1:{n}", "x")
    assert admin.delete("/admin/cache", headers=ADMIN).json()["tier1_removed"] == 1203
    assert admin.get("/admin/cache", headers=ADMIN).json()["tier1_entries"] == 0


@pytest.mark.parametrize("method", ["get", "delete"])
def test_without_an_admin_key_configured_the_routes_do_not_exist(
    settings: Settings, fake_redis: aioredis.FakeRedis, method: str
) -> None:
    with _client(_settings(settings, None), fake_redis, None) as client:
        assert getattr(client, method)("/admin/cache", headers=ADMIN).status_code == 404


@pytest.mark.parametrize("method", ["get", "delete"])
def test_a_caller_key_is_not_the_admin_key(admin: TestClient, method: str) -> None:
    response = getattr(admin, method)("/admin/cache", headers=CALLER)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "INVALID_API_KEY"


def test_no_key_at_all_is_refused(admin: TestClient) -> None:
    response = admin.delete("/admin/cache")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "MISSING_API_KEY"


def test_a_refused_flush_leaves_the_cache_alone(admin: TestClient) -> None:
    _ask(admin, "keep me")
    admin.delete("/admin/cache", headers=CALLER)
    assert admin.get("/admin/cache", headers=ADMIN).json()["tier1_entries"] == 1


def test_with_tier2_disabled_only_tier1_is_counted(
    settings: Settings, fake_redis: aioredis.FakeRedis
) -> None:
    with _client(_settings(settings, ADMIN_KEY), fake_redis, None) as client:
        _ask(client, "question")
        assert client.get("/admin/cache", headers=ADMIN).json() == {
            "tier1_entries": 1, "tier2_entries": None, "tier2": "disabled",
        }
        assert client.delete("/admin/cache", headers=ADMIN).json() == {
            "tier1_removed": 1, "tier2_removed": None, "tier2": "disabled",
        }


def test_a_chroma_outage_still_counts_and_clears_tier1(
    settings: Settings, fake_redis: aioredis.FakeRedis, mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock,
) -> None:
    def down() -> Any:
        raise ConnectionError("chroma is down")

    broken = SemanticCache(down, mapping_embedder, collection_name="x", ttl_seconds=60,
                           threshold=0.90, clock=fake_clock)
    with _client(_settings(settings, ADMIN_KEY), fake_redis, broken) as client:
        _ask(client, "question")
        assert client.get("/admin/cache", headers=ADMIN).json() == {
            "tier1_entries": 1, "tier2_entries": None, "tier2": "unavailable",
        }
        assert client.delete("/admin/cache", headers=ADMIN).json() == {
            "tier1_removed": 1, "tier2_removed": None, "tier2": "unavailable",
        }


async def test_a_delete_that_fails_midway_reports_tier2_unavailable(
    semantic: SemanticCache, mapping_embedder: MappingEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.models import Message

    await semantic.add("gpt-4o-mini", [Message(role="user", content="q")], "e1", {"ok": True})
    real_call = semantic._call

    async def failing_delete(operation: str, *args: Any, **kwargs: Any) -> Any:
        if operation == "delete":
            from app.cache.semantic_cache import UNAVAILABLE
            return UNAVAILABLE
        return await real_call(operation, *args, **kwargs)

    monkeypatch.setattr(semantic, "_call", failing_delete)
    assert await semantic.clear() is None


def test_the_admin_key_is_not_shown_in_settings_repr(settings: Settings) -> None:
    assert ADMIN_KEY not in repr(_settings(settings, ADMIN_KEY))
