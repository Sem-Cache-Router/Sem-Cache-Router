"""Cache-Control: no-cache and no-store skip both tiers, store nothing, and are counted apart."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.providers.stub import StubProvider
from tests.conftest import router_around

AUTH = {"Authorization": "Bearer caller-key"}


def ask(client: TestClient, content: str, cache_control: str | None = None) -> Any:
    headers = dict(AUTH)
    if cache_control is not None:
        headers["Cache-Control"] = cache_control
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": content}]}
    return client.post("/v1/chat/completions", json=body, headers=headers)


@pytest.fixture
def counted(client: TestClient) -> tuple[TestClient, StubProvider]:
    provider = StubProvider()
    client.app.state.router = router_around(provider)  # type: ignore[attr-defined]
    return client, provider


@pytest.mark.parametrize("directive", ["no-cache", "no-store", "max-age=0, no-cache", "NO-STORE"])
def test_a_bypass_skips_a_cached_answer_and_asks_the_provider(
    counted: tuple[TestClient, StubProvider], directive: str,
) -> None:
    client, provider = counted
    ask(client, "a question")
    response = ask(client, "a question", directive)
    assert response.status_code == 200
    assert response.json()["semcache_status"] == "bypass"
    assert provider.call_count == 2


def test_a_bypassed_answer_is_not_stored(counted: tuple[TestClient, StubProvider]) -> None:
    client, provider = counted
    ask(client, "never stored", "no-store")
    assert ask(client, "never stored").json()["semcache_status"] == "miss"
    assert provider.call_count == 2


@pytest.mark.parametrize("directive", ["private", "max-age=60", ""])
def test_other_directives_use_the_cache(
    counted: tuple[TestClient, StubProvider], directive: str,
) -> None:
    client, _ = counted
    ask(client, "a question")
    assert ask(client, "a question", directive).json()["semcache_status"] == "exact_hit"


def test_a_bypass_is_counted_apart_and_leaves_the_hit_ratio_alone(
    counted: tuple[TestClient, StubProvider],
) -> None:
    client, _ = counted
    ask(client, "a question")
    ask(client, "a question")
    ask(client, "a question", "no-cache")
    metrics = client.get("/metrics").json()
    assert metrics["cache_bypassed"] == 1
    assert (metrics["misses"], metrics["tier1_hits"]) == (1, 1)
    assert metrics["hit_ratio"] == 0.5


def test_a_bypass_returns_the_provider_answer_with_its_usage(
    counted: tuple[TestClient, StubProvider],
) -> None:
    client, _ = counted
    body = ask(client, "with usage", "no-cache").json()
    assert body["choices"][0]["message"]["content"] == "Stub answer to: with usage"
    assert body["usage"]["total_tokens"] > 0
    assert body["semcache_similarity"] is None
