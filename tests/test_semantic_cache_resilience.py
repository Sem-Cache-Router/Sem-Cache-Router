# Tier 2 degradation and housekeeping tests.
#
# The rule under test: losing the vector store costs hit rate, never
# availability. Every failure here must come back as a miss or a no-op, never
# as an exception that would reach a request.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.cache.semantic_cache import SemanticCache
from app.models import Message
from tests.conftest import FakeClock, MappingEmbedder, base_vector

MODEL = "gpt-4o-mini"
BODY = {"choices": [{"message": {"role": "assistant", "content": "Paris."}}]}
PROMPT = "What is the capital of France?"
COOLDOWN = 30.0


def user(content: str) -> list[Message]:
    """One user message."""
    return [Message(role="user", content=content)]


class FlakyFactory:
    """A Chroma client factory that can be switched off, and counts its calls.

    When down it raises the same error the real client raises at construction
    with no server, so the cache sees exactly what an outage looks like.
    """

    def __init__(self, real: Callable[[], Any], up: bool = True) -> None:
        self._real = real
        self.up = up
        self.calls = 0

    def __call__(self) -> Any:
        self.calls += 1
        if not self.up:
            raise ValueError("Could not connect to a Chroma server. Are you sure it is running?")
        return self._real()


class BrokenCollection:
    """A collection whose server has gone away after the connection was made."""

    def __getattr__(self, name: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("server went away")

        return fail


@pytest.fixture
def flaky(chroma_factory: Callable[[], Any]) -> FlakyFactory:
    """A factory over in-memory Chroma that starts up."""
    return FlakyFactory(chroma_factory)


@pytest.fixture
def cache(
    flaky: FlakyFactory,
    mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock,
    collection_name: str,
) -> SemanticCache:
    """A semantic cache whose Chroma can be taken down by the test."""
    mapping_embedder.register(PROMPT, base_vector())
    return SemanticCache(
        flaky,
        mapping_embedder,
        collection_name=collection_name,
        ttl_seconds=60,
        threshold=0.90,
        reconnect_seconds=COOLDOWN,
        clock=fake_clock,
    )


# --- outage ---------------------------------------------------------------


async def test_unreachable_chroma_is_a_miss_not_an_error(
    cache: SemanticCache, flaky: FlakyFactory
) -> None:
    """TEST-015 at module level. The gateway carries on with Tier 1 only."""
    flaky.up = False
    assert await cache.search(MODEL, user(PROMPT)) is None


async def test_writing_with_chroma_down_does_not_raise(
    cache: SemanticCache, flaky: FlakyFactory
) -> None:
    """A failed write must not turn a successful provider answer into an error."""
    flaky.up = False
    await cache.add(MODEL, user(PROMPT), "e1", BODY)


async def test_no_reconnect_is_attempted_inside_the_cooldown(
    cache: SemanticCache, flaky: FlakyFactory, fake_clock: FakeClock
) -> None:
    """During an outage, requests must not each pay for a doomed connection."""
    flaky.up = False
    await cache.search(MODEL, user(PROMPT))
    fake_clock.advance(COOLDOWN - 1)
    await cache.search(MODEL, user(PROMPT))
    await cache.add(MODEL, user(PROMPT), "e1", BODY)
    assert flaky.calls == 1


async def test_chroma_coming_back_is_picked_up_without_a_restart(
    cache: SemanticCache, flaky: FlakyFactory, fake_clock: FakeClock
) -> None:
    """Once the cooldown passes, the next request reconnects."""
    flaky.up = False
    assert await cache.search(MODEL, user(PROMPT)) is None

    flaky.up = True
    fake_clock.advance(COOLDOWN + 1)
    await cache.add(MODEL, user(PROMPT), "e1", BODY)

    assert await cache.search(MODEL, user(PROMPT)) is not None
    assert flaky.calls == 2


async def test_a_still_down_server_starts_a_fresh_cooldown(
    cache: SemanticCache, flaky: FlakyFactory, fake_clock: FakeClock
) -> None:
    """A failed retry backs off again rather than retrying on every request."""
    flaky.up = False
    await cache.search(MODEL, user(PROMPT))
    fake_clock.advance(COOLDOWN + 1)
    await cache.search(MODEL, user(PROMPT))
    fake_clock.advance(1)
    await cache.search(MODEL, user(PROMPT))
    assert flaky.calls == 2


async def test_a_server_lost_mid_flight_degrades_and_backs_off(
    cache: SemanticCache, flaky: FlakyFactory
) -> None:
    """The connection worked, then the server vanished. The request still gets a
    miss, the dead connection is dropped, and the cooldown applies."""
    await cache.add(MODEL, user(PROMPT), "e1", BODY)
    cache._collection = BrokenCollection()

    assert await cache.search(MODEL, user(PROMPT)) is None
    assert cache._collection is None

    await cache.search(MODEL, user(PROMPT))
    assert flaky.calls == 1


# --- availability ---------------------------------------------------------


async def test_reports_available_when_chroma_answers(cache: SemanticCache) -> None:
    """The health endpoint relies on this to call the gateway healthy."""
    assert await cache.is_available() is True


async def test_reports_unavailable_when_chroma_is_down(
    cache: SemanticCache, flaky: FlakyFactory
) -> None:
    """The health endpoint relies on this to report a degraded gateway."""
    flaky.up = False
    assert await cache.is_available() is False


async def test_a_dead_server_behind_a_live_client_reports_unavailable(
    cache: SemanticCache,
) -> None:
    """Availability is a real round trip, not trust in a cached connection."""
    await cache.add(MODEL, user(PROMPT), "e1", BODY)
    cache._collection = BrokenCollection()
    assert await cache.is_available() is False


# --- eviction -------------------------------------------------------------


async def test_eviction_removes_only_expired_entries(
    cache: SemanticCache, mapping_embedder: MappingEmbedder, fake_clock: FakeClock
) -> None:
    """Housekeeping must not take anything still live with it."""
    await cache.add(MODEL, user(PROMPT), "old", BODY)
    fake_clock.advance(45)
    mapping_embedder.register("fresh question", base_vector())
    await cache.add(MODEL, user("fresh question"), "new", BODY)
    fake_clock.advance(20)

    assert await cache.evict_expired() == 1
    assert cache._get_collection().get()["ids"] == ["new"]


async def test_eviction_with_nothing_expired_removes_nothing(cache: SemanticCache) -> None:
    """A routine run on a fresh cache is a no-op."""
    await cache.add(MODEL, user(PROMPT), "e1", BODY)
    assert await cache.evict_expired() == 0


async def test_eviction_with_chroma_down_reports_nothing_and_does_not_raise(
    cache: SemanticCache, flaky: FlakyFactory
) -> None:
    """Housekeeping during an outage is skipped, not escalated."""
    flaky.up = False
    assert await cache.evict_expired() == 0
