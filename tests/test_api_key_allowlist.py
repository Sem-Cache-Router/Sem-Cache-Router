"""The API key allowlist: off by default, and when set, only listed keys get in."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fakeredis import aioredis
from fastapi.testclient import TestClient

from app.cache.exact_cache import ExactCache
from app.config import Settings
from app.main import create_app
from app.providers.stub import StubProvider
from tests.conftest import router_around

BODY = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def client_with(settings: Settings, redis: aioredis.FakeRedis) -> TestClient:
    app = create_app()
    app.state.settings = settings
    app.state.redis = redis
    app.state.provider = StubProvider()
    app.state.cache = ExactCache(redis, settings.cache_ttl_seconds)
    app.state.limiter = None
    return TestClient(app)


@pytest.fixture
def listed(settings: Settings, fake_redis: aioredis.FakeRedis) -> Iterator[TestClient]:
    configured = settings.model_copy(update={"allowed_api_keys": " team-a , team-b,,"})
    with client_with(configured, fake_redis) as client:
        yield client


def test_without_an_allowlist_any_key_is_accepted(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json=BODY, headers=bearer("anything"))
    assert response.status_code == 200


@pytest.mark.parametrize("key", ["team-a", "team-b"])
def test_a_listed_key_is_accepted(listed: TestClient, key: str) -> None:
    assert listed.post("/v1/chat/completions", json=BODY, headers=bearer(key)).status_code == 200


@pytest.mark.parametrize("key", ["team-c", "TEAM-A", "team", "team-ab"])
def test_any_other_key_is_refused_before_any_work(listed: TestClient, key: str) -> None:
    provider = StubProvider()
    listed.app.state.router = router_around(provider)  # type: ignore[attr-defined]
    response = listed.post("/v1/chat/completions", json=BODY, headers=bearer(key))
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_API_KEY"
    assert provider.call_count == 0


def test_an_allowlist_requires_a_key_even_when_keys_are_optional(
    settings: Settings, fake_redis: aioredis.FakeRedis,
) -> None:
    configured = settings.model_copy(
        update={"allowed_api_keys": "team-a", "require_api_key": False}
    )
    with client_with(configured, fake_redis) as client:
        response = client.post("/v1/chat/completions", json=BODY)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "MISSING_API_KEY"


def test_the_allowlist_is_parsed_and_blank_entries_ignored(settings: Settings) -> None:
    assert settings.api_key_allowlist == frozenset()
    parsed = settings.model_copy(update={"allowed_api_keys": " a, b ,, ,c"}).api_key_allowlist
    assert parsed == frozenset({"a", "b", "c"})


def test_the_keys_never_appear_in_the_settings_repr(settings: Settings) -> None:
    configured = settings.model_copy(update={"allowed_api_keys": "team-secret"})
    assert "team-secret" not in repr(configured)


def test_the_allowlist_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMCACHE_ALLOWED_API_KEYS", "k1,k2")
    assert Settings(_env_file=None).api_key_allowlist == frozenset({"k1", "k2"})  # type: ignore[call-arg]
