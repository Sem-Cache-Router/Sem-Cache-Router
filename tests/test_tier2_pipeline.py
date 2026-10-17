# Tier 2 through the real request path.
#
# These drive POST /v1/chat/completions with a real SemanticCache over
# in-memory Chroma and an embedder whose vectors the test chooses. That is what
# lets them assert a semantic hit at a known similarity, end to end.

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
from app.models import Message
from app.providers.stub import StubProvider
from tests.conftest import FakeClock, MappingEmbedder, base_vector, unit_vector_at

AUTH = {"Authorization": "Bearer test-key"}
MODEL = "gpt-4o-mini"
STORED = "What is the capital of France?"
REWORDED = "Tell me France's capital city."


def body(content: str, model: str = MODEL) -> dict[str, Any]:
    """A minimal chat completion request."""
    return {"model": model, "messages": [{"role": "user", "content": content}]}


class FakeLimiter:
    """Records what the route does with a reservation."""

    def __init__(self) -> None:
        self.released: list[str] = []
        self.reconciled: list[str] = []

    async def reserve(self, api_key: str, estimate: int) -> str:
        return f"reservation-{len(self.released) + len(self.reconciled)}"

    async def release(self, reservation: str) -> None:
        self.released.append(reservation)

    async def reconcile(self, reservation: str, actual: int) -> None:
        self.reconciled.append(reservation)

    def retry_after(self, api_key: str) -> int:
        return 1


class Tier2Harness:
    """The pieces a Tier 2 test needs to set up and inspect."""

    def __init__(
        self,
        client: TestClient,
        provider: StubProvider,
        embedder: MappingEmbedder,
        exact: ExactCache,
        semantic: SemanticCache,
    ) -> None:
        self.client = client
        self.provider = provider
        self.embedder = embedder
        self.exact = exact
        self.semantic = semantic

    def ask(self, content: str, model: str = MODEL) -> dict[str, Any]:
        response = self.client.post("/v1/chat/completions", json=body(content, model), headers=AUTH)
        assert response.status_code == 200, response.text
        return response.json()  # type: ignore[no-any-return]


@pytest.fixture
def harness(
    settings: Settings,
    fake_redis: aioredis.FakeRedis,
    mapping_embedder: MappingEmbedder,
    chroma_factory: Callable[[], Any],
    collection_name: str,
    fake_clock: FakeClock,
) -> Iterator[Tier2Harness]:
    """The real app with Tier 1 on fakeredis and Tier 2 on in-memory Chroma."""
    provider = StubProvider()
    exact = ExactCache(fake_redis, settings.cache_ttl_seconds)
    semantic = SemanticCache(
        chroma_factory,
        mapping_embedder,
        collection_name=collection_name,
        ttl_seconds=60,
        threshold=0.90,
        clock=fake_clock,
    )
    mapping_embedder.register(STORED, base_vector())

    app = create_app()
    app.state.settings = settings
    app.state.redis = fake_redis
    app.state.provider = provider
    app.state.cache = exact
    app.state.limiter = None
    app.state.semantic_cache = semantic

    with TestClient(app) as client:
        yield Tier2Harness(client, provider, mapping_embedder, exact, semantic)


# --- the semantic hit ------------------------------------------------------


def test_a_reworded_question_is_a_semantic_hit(harness: Tier2Harness) -> None:
    """The point of the project: same meaning, different words, no second
    provider call, and the score that justified serving it."""
    first = harness.ask(STORED)
    harness.embedder.register(REWORDED, unit_vector_at(0.95))
    second = harness.ask(REWORDED)

    assert first["semcache_status"] == "miss"
    assert second["semcache_status"] == "semantic_hit"
    assert second["semcache_similarity"] == pytest.approx(0.95, abs=1e-3)
    assert second["choices"] == first["choices"]
    assert harness.provider.call_count == 1


def test_a_question_below_the_threshold_goes_to_the_provider(harness: Tier2Harness) -> None:
    """0.85 does not clear 0.90, so the provider answers it."""
    harness.ask(STORED)
    harness.embedder.register(REWORDED, unit_vector_at(0.85))
    second = harness.ask(REWORDED)

    assert second["semcache_status"] == "miss"
    assert second["semcache_similarity"] is None
    assert harness.provider.call_count == 2


def test_a_different_model_does_not_get_a_semantic_hit(harness: Tier2Harness) -> None:
    """Same meaning, different model, different answer."""
    harness.ask(STORED, model="gpt-4o-mini")
    harness.embedder.register(REWORDED, unit_vector_at(0.99))
    second = harness.ask(REWORDED, model="gpt-4o")

    assert second["semcache_status"] == "miss"
    assert harness.provider.call_count == 2


def test_an_identical_question_is_answered_by_tier_one_first(harness: Tier2Harness) -> None:
    """Exact repeats never reach the embedder; that is why Tier 1 comes first."""
    harness.ask(STORED)
    calls_before = len(harness.embedder.calls)
    second = harness.ask(STORED)

    assert second["semcache_status"] == "exact_hit"
    assert second["semcache_similarity"] is None
    assert len(harness.embedder.calls) == calls_before


# --- what gets written -----------------------------------------------------


async def test_a_miss_indexes_the_answer_under_tier_one_entry_id(harness: Tier2Harness) -> None:
    """The shared id traces a hit in either tier back to one origin request."""
    harness.ask(STORED)
    entry = await harness.exact.get(MODEL, [Message(role="user", content=STORED)])
    assert entry is not None
    assert harness.semantic._get_collection().get()["ids"] == [entry.entry_id]


async def test_a_semantic_hit_is_not_copied_into_tier_one(harness: Tier2Harness) -> None:
    """Copying it would later serve a paraphrase's answer as an exact hit with no
    score, laundering an uncertain match into a certain one."""
    harness.ask(STORED)
    harness.embedder.register(REWORDED, unit_vector_at(0.95))
    harness.ask(REWORDED)

    assert await harness.exact.get(MODEL, [Message(role="user", content=REWORDED)]) is None
    third = harness.ask(REWORDED)
    assert third["semcache_status"] == "semantic_hit"


# --- degradation -----------------------------------------------------------


def test_with_tier_two_failing_the_request_still_succeeds(harness: Tier2Harness) -> None:
    """TEST-015 end to end. Tier 2 may cost the hit, never the request."""

    async def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("embedding failed")

    harness.embedder.encode = explode  # type: ignore[method-assign]
    first = harness.ask(STORED)
    second = harness.ask("Some other question entirely")

    assert first["semcache_status"] == "miss"
    assert second["semcache_status"] == "miss"
    assert harness.provider.call_count == 2


# --- accounting ------------------------------------------------------------


def test_a_semantic_hit_releases_the_whole_reservation(harness: Tier2Harness) -> None:
    """A hit spends no provider tokens, so nothing is reconciled against usage."""
    limiter = FakeLimiter()
    harness.client.app.state.limiter = limiter  # type: ignore[attr-defined]
    harness.ask(STORED)
    harness.embedder.register(REWORDED, unit_vector_at(0.95))
    harness.ask(REWORDED)

    assert len(limiter.reconciled) == 1
    assert len(limiter.released) == 1


def test_a_semantic_hit_is_counted_in_metrics(harness: Tier2Harness) -> None:
    """The benchmark reads these counters, so the tiers must be told apart."""
    harness.ask(STORED)
    harness.embedder.register(REWORDED, unit_vector_at(0.95))
    harness.ask(REWORDED)

    counters = harness.client.get("/metrics").json()
    assert counters["tier2_hits"] == 1
    assert counters["tier1_hits"] == 0
    assert counters["misses"] == 1
