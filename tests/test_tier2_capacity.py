"""The Tier 2 capacity bound: the oldest entries beyond max_entries are evicted."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.cache.eviction import evict_once
from app.cache.semantic_cache import UNAVAILABLE, SemanticCache
from app.config import Settings
from app.models import Message
from tests.conftest import FakeClock, MappingEmbedder, unit_vector_at

MODEL = "gpt-4o-mini"


def _cache(factory: Callable[[], Any], embedder: MappingEmbedder, clock: FakeClock,
           name: str, max_entries: int | None, ttl: int = 600) -> SemanticCache:
    return SemanticCache(factory, embedder, collection_name=name, ttl_seconds=ttl,
                         threshold=0.99, max_entries=max_entries, clock=clock)


async def _store(cache: SemanticCache, embedder: MappingEmbedder, clock: FakeClock,
                 n: int) -> None:
    """Store n entries one second apart, so entry i is older than entry i + 1."""
    for i in range(n):
        prompt = f"question {i}"
        embedder.register(prompt, unit_vector_at(i / (n + 1)))
        await cache.add(MODEL, [Message(role="user", content=prompt)], f"e{i}", {"n": i})
        clock.advance(1)


async def _ids(cache: SemanticCache) -> list[str]:
    result = await cache._call("get", include=[])
    return sorted(result["ids"])


async def test_the_oldest_entries_beyond_the_bound_are_removed(
    chroma_factory: Callable[[], Any], mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock, collection_name: str,
) -> None:
    cache = _cache(chroma_factory, mapping_embedder, fake_clock, collection_name, max_entries=3)
    await _store(cache, mapping_embedder, fake_clock, 5)
    assert await cache.evict_over_capacity() == 2
    assert await _ids(cache) == ["e2", "e3", "e4"]


async def test_at_or_under_the_bound_nothing_is_removed(
    chroma_factory: Callable[[], Any], mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock, collection_name: str,
) -> None:
    cache = _cache(chroma_factory, mapping_embedder, fake_clock, collection_name, max_entries=3)
    await _store(cache, mapping_embedder, fake_clock, 3)
    assert await cache.evict_over_capacity() == 0
    assert len(await _ids(cache)) == 3


async def test_without_a_bound_nothing_is_ever_removed(
    chroma_factory: Callable[[], Any], mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock, collection_name: str,
) -> None:
    cache = _cache(chroma_factory, mapping_embedder, fake_clock, collection_name, max_entries=None)
    await _store(cache, mapping_embedder, fake_clock, 4)
    assert await cache.evict_over_capacity() == 0


async def test_a_pass_removes_the_expired_first_and_then_only_what_is_still_too_many(
    chroma_factory: Callable[[], Any], mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock, collection_name: str,
) -> None:
    cache = _cache(chroma_factory, mapping_embedder, fake_clock, collection_name,
                   max_entries=2, ttl=52)

    async def store(i: int) -> None:
        prompt = f"question {i}"
        mapping_embedder.register(prompt, unit_vector_at(i / 10))
        await cache.add(MODEL, [Message(role="user", content=prompt)], f"e{i}", {"n": i})

    await store(0)                 # t=0, expires at 52
    fake_clock.advance(50)
    for i in range(1, 5):          # t=50..53, expire at 102..105
        await store(i)
        fake_clock.advance(1)      # ends at t=54: only e0 has expired

    reported: list[int] = []
    await evict_once(cache, reported.append)
    # One expired (e0), then the two oldest live entries (e1, e2) to get down to 2.
    assert reported == [3]
    assert await _ids(cache) == ["e3", "e4"]


async def test_at_the_bound_the_entries_are_not_even_read(
    chroma_factory: Callable[[], Any], mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock, collection_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reading every entry's metadata is the expensive part, so it happens only over the bound.
    cache = _cache(chroma_factory, mapping_embedder, fake_clock, collection_name, max_entries=3)
    await _store(cache, mapping_embedder, fake_clock, 3)
    real_call = cache._call
    operations: list[str] = []

    async def spy(operation: str, *args: Any, **kwargs: Any) -> Any:
        operations.append(operation)
        return await real_call(operation, *args, **kwargs)

    monkeypatch.setattr(cache, "_call", spy)
    assert await cache.evict_over_capacity() == 0
    assert operations == ["count"]


async def test_an_outage_removes_nothing(
    chroma_factory: Callable[[], Any], mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock, collection_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache(chroma_factory, mapping_embedder, fake_clock, collection_name, max_entries=1)
    await _store(cache, mapping_embedder, fake_clock, 3)
    real_call = cache._call

    for failing in ("count", "get", "delete"):
        async def call(operation: str, *args: Any, _failing: str = failing, **kwargs: Any) -> Any:
            if operation == _failing:
                return UNAVAILABLE
            return await real_call(operation, *args, **kwargs)

        monkeypatch.setattr(cache, "_call", call)
        assert await cache.evict_over_capacity() == 0
    monkeypatch.setattr(cache, "_call", real_call)
    assert len(await _ids(cache)) == 3


def test_the_bound_defaults_to_one_hundred_thousand_and_must_be_positive() -> None:
    assert Settings(_env_file=None).tier2_max_entries == 100_000  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Settings(_env_file=None, tier2_max_entries=0)  # type: ignore[call-arg]


async def test_the_gateway_builds_tier2_with_the_configured_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main

    class NoLoad:
        def __init__(self, model: str, **options: object) -> None:
            pass

        def load(self) -> None:
            pass

    monkeypatch.setattr(main, "OnnxEmbedder", NoLoad)
    settings = Settings(_env_file=None, tier2_max_entries=7)  # type: ignore[call-arg]
    cache = await main.build_semantic_cache(settings)
    assert cache._max_entries == 7
