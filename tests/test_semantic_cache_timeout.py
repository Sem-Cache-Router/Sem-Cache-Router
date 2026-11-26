# Tier 2 timeout tests.
#
# The failure these cover is a vector store that hangs rather than refuses. A
# refused connection raises, and the degradation tests already prove that costs a
# hit and nothing else. A hung one raises nothing, so without a timeout the
# request waits on it for as long as it stays hung. Every test here makes Chroma
# stop answering, in one place or another, and asserts the request still comes
# back promptly as a miss.
#
# The stand in Chroma blocks on an event rather than sleeping, so a test decides
# exactly when it recovers, and releases every blocked thread when it finishes so
# none outlives the test.

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
import pytest_asyncio

from app.cache import semantic_cache
from app.cache.semantic_cache import SemanticCache
from app.config import Settings
from app.models import Message
from tests.conftest import FakeClock, MappingEmbedder, base_vector

MODEL = "gpt-4o-mini"
BODY = {"choices": [{"message": {"role": "assistant", "content": "Paris."}}]}
PROMPT = "What is the capital of France?"
COOLDOWN = 30.0
TIMEOUT = 0.1

# Generous against scheduler noise on a busy CI machine, yet far below the ten
# seconds a hung call would otherwise hold a test for.
PROMPTLY = 1.0


def user(content: str) -> list[Message]:
    return [Message(role="user", content=content)]


class SlowChroma:
    """A Chroma client factory whose connection and operations can be made to hang.

    Wraps a real in-memory Chroma, so anything that is not hung behaves exactly
    as it does in production. Hanging means blocking on `release`, capped at ten
    seconds so a test that forgets to release cannot leave a thread behind for good.
    """

    def __init__(self, real_factory: Callable[[], Any], release: threading.Event) -> None:
        self._real_factory = real_factory
        self.release = release
        self.hang_on: set[str] = set()
        self.hang_connect = False
        self.delay = 0.0
        self.connects = 0
        self.operations: list[str] = []

    def __call__(self) -> Any:
        self.connects += 1
        if self.hang_connect:
            self.release.wait(10)
        return _Client(self, self._real_factory())


class _Client:
    def __init__(self, owner: SlowChroma, real: Any) -> None:
        self._owner = owner
        self._real = real

    def get_or_create_collection(self, name: str, metadata: dict[str, str]) -> Any:
        real = self._real.get_or_create_collection(name, metadata=metadata)
        return _Collection(self._owner, real)


class _Collection:
    def __init__(self, owner: SlowChroma, real: Any) -> None:
        self._owner = owner
        self._real = real

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._real, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            self._owner.operations.append(name)
            if name in self._owner.hang_on:
                self._owner.release.wait(10)
            if self._owner.delay:
                time.sleep(self._owner.delay)
            return method(*args, **kwargs)

        return call


@pytest_asyncio.fixture
async def slow(chroma_factory: Callable[[], Any]) -> AsyncIterator[SlowChroma]:
    """A Chroma that starts healthy, and lets every blocked thread go at teardown.

    Async, although nothing here awaits, so that its teardown runs while the event
    loop is still alive. The loop's own shutdown waits for every worker thread, so
    a plain fixture would release them only after that wait had run its full ten
    seconds, once per test.
    """
    release = threading.Event()
    yield SlowChroma(chroma_factory, release)
    release.set()


@pytest.fixture
def cache(
    slow: SlowChroma,
    mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock,
    collection_name: str,
) -> SemanticCache:
    mapping_embedder.register(PROMPT, base_vector())
    return SemanticCache(
        slow,
        mapping_embedder,
        collection_name=collection_name,
        ttl_seconds=60,
        threshold=0.90,
        reconnect_seconds=COOLDOWN,
        timeout_seconds=TIMEOUT,
        clock=fake_clock,
    )


async def timed(awaitable: Any) -> tuple[Any, float]:
    """The result of an awaitable and how long it took, in seconds."""
    started = time.monotonic()
    result = await awaitable
    return result, time.monotonic() - started


# --- a hang is an outage ---------------------------------------------------------


async def test_a_lookup_that_hangs_is_a_miss_after_the_timeout(
    cache: SemanticCache, slow: SlowChroma
) -> None:
    await cache.add(MODEL, user(PROMPT), "e1", BODY)
    slow.hang_on = {"query"}

    result, elapsed = await timed(cache.search(MODEL, user(PROMPT)))

    assert result is None
    assert TIMEOUT * 0.9 <= elapsed < PROMPTLY


async def test_a_write_that_hangs_does_not_hang_the_request(
    cache: SemanticCache, slow: SlowChroma
) -> None:
    """A provider answer that is already in hand must not wait on the index."""
    slow.hang_on = {"upsert"}

    _, elapsed = await timed(cache.add(MODEL, user(PROMPT), "e1", BODY))

    assert elapsed < PROMPTLY


async def test_a_connection_that_hangs_is_treated_as_an_outage(
    cache: SemanticCache, slow: SlowChroma
) -> None:
    slow.hang_connect = True

    result, elapsed = await timed(cache.search(MODEL, user(PROMPT)))

    assert result is None
    assert elapsed < PROMPTLY


async def test_a_connection_that_hangs_starts_the_cooldown_too(
    cache: SemanticCache, slow: SlowChroma
) -> None:
    """The second request must not try the hung connection again, or wait on it."""
    slow.hang_connect = True
    await cache.search(MODEL, user(PROMPT))

    result, elapsed = await timed(cache.search(MODEL, user(PROMPT)))

    assert result is None
    assert elapsed < TIMEOUT / 2
    assert slow.connects == 1


async def test_the_health_probe_reports_a_hung_chroma_instead_of_hanging(
    cache: SemanticCache, slow: SlowChroma
) -> None:
    slow.hang_on = {"count"}

    available, elapsed = await timed(cache.is_available())

    assert available is False
    assert elapsed < PROMPTLY


# --- the cooldown follows a timeout ------------------------------------------------


async def test_a_timeout_starts_the_cooldown_so_the_next_request_does_not_wait(
    cache: SemanticCache, slow: SlowChroma
) -> None:
    """The point of bounding the call: one request pays for the outage, not all of them."""
    slow.hang_on = {"query"}
    await cache.search(MODEL, user(PROMPT))

    result, elapsed = await timed(cache.search(MODEL, user(PROMPT)))

    assert result is None
    assert elapsed < TIMEOUT / 2
    assert slow.operations.count("query") == 1


async def test_a_request_during_the_cooldown_does_not_take_a_worker_thread(
    cache: SemanticCache, slow: SlowChroma, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hung calls hold threads, so the cooldown must be answered without one."""
    slow.hang_on = {"query"}
    await cache.search(MODEL, user(PROMPT))

    dispatched: list[str] = []
    real_to_thread = asyncio.to_thread

    async def counting(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        dispatched.append(getattr(func, "__name__", repr(func)))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(semantic_cache.asyncio, "to_thread", counting)
    await cache.search(MODEL, user(PROMPT))

    assert dispatched == []


async def test_the_cache_reconnects_once_chroma_recovers_and_the_cooldown_ends(
    cache: SemanticCache, slow: SlowChroma, fake_clock: FakeClock
) -> None:
    await cache.add(MODEL, user(PROMPT), "e1", BODY)
    slow.hang_on = {"query"}
    assert await cache.search(MODEL, user(PROMPT)) is None

    slow.hang_on = set()
    slow.release.set()
    fake_clock.advance(COOLDOWN + 1)

    hit = await cache.search(MODEL, user(PROMPT))

    assert hit is not None
    assert hit.entry_id == "e1"


# --- the bound must not cost anything when Chroma is healthy -----------------------------


async def test_a_call_that_is_slow_but_within_the_timeout_is_served(
    cache: SemanticCache, slow: SlowChroma
) -> None:
    await cache.add(MODEL, user(PROMPT), "e1", BODY)
    slow.delay = TIMEOUT / 5

    hit = await cache.search(MODEL, user(PROMPT))

    assert hit is not None


async def test_a_healthy_chroma_is_unaffected(cache: SemanticCache, slow: SlowChroma) -> None:
    await cache.add(MODEL, user(PROMPT), "e1", BODY)

    hit = await cache.search(MODEL, user(PROMPT))

    assert hit is not None
    assert await cache.is_available() is True


# --- the setting ----------------------------------------------------------------------------


def test_the_default_timeout_is_two_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEMCACHE_CHROMA_TIMEOUT_SECONDS", raising=False)
    assert Settings(_env_file=None).chroma_timeout_seconds == 2.0  # type: ignore[call-arg]


@pytest.mark.parametrize("value", [0, -1.5])
def test_a_timeout_that_allows_no_time_is_rejected_at_startup(value: float) -> None:
    """Zero would time out every call, so Tier 2 would never serve anything."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, chroma_timeout_seconds=value)  # type: ignore[call-arg]


async def test_the_configured_timeout_reaches_the_cache(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import main

    class FakeEmbedder:
        def __init__(self, model_name: str, *, cache_size: int = 0, threads: int = 0) -> None:
            pass

        def load(self) -> None:
            pass

    monkeypatch.setattr(main, "OnnxEmbedder", FakeEmbedder)
    configured = settings.model_copy(update={"chroma_timeout_seconds": 0.75})

    built = await main.build_semantic_cache(configured)

    assert built._timeout_seconds == 0.75
