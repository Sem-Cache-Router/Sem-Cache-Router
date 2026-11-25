# Tier 2 eviction scheduling tests.
#
# SemanticCache.evict_expired already has its own tests. What these cover is the
# part that was missing: that something calls it, on a schedule, and that the
# schedule survives everything a long running gateway will throw at it.
#
# The loop is stepped with a fake sleep, so no test waits in real time. The
# lifespan tests are the exception, because they prove the task really runs on a
# real event loop and really stops with it; they poll with a ceiling instead of
# sleeping a fixed time.

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import pytest
from fakeredis import aioredis
from fastapi.testclient import TestClient

import app.main as main
from app.cache.eviction import evict_once, run_eviction_loop, start_eviction, stop_eviction
from app.cache.semantic_cache import SemanticCache
from app.config import Settings
from app.models import Message
from app.providers.stub import StubProvider
from tests.conftest import FakeClock, MappingEmbedder, base_vector

MODEL = "gpt-4o-mini"
BODY = {"choices": [{"message": {"role": "assistant", "content": "Paris."}}]}
INTERVAL = 60.0


def user(content: str) -> list[Message]:
    return [Message(role="user", content=content)]


class StopLoop(Exception):
    """Raised by the fake sleep to end a loop that would otherwise run forever."""


class StepSleep:
    """A sleep that records its argument and ends the loop after a set number of passes."""

    def __init__(self, passes: int, on_wake: Callable[[float], None] | None = None) -> None:
        self.calls: list[float] = []
        self._passes = passes
        self._on_wake = on_wake

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if len(self.calls) > self._passes:
            raise StopLoop
        if self._on_wake:
            self._on_wake(seconds)


class ScriptedCache:
    """A cache whose evict_expired returns, or raises, a scripted sequence."""

    def __init__(self, *results: int | Exception) -> None:
        self._results = list(results)
        self.passes = 0

    async def evict_expired(self) -> int:
        self.passes += 1
        result = self._results.pop(0) if self._results else 0
        if isinstance(result, Exception):
            raise result
        return result

    async def evict_over_capacity(self) -> int:
        return 0


async def until(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    """Poll for a condition, so a test waits only as long as it needs to."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


# --- one pass ------------------------------------------------------------------


async def test_a_pass_reports_how_many_entries_it_removed() -> None:
    reported: list[int] = []
    await evict_once(ScriptedCache(3), reported.append)  # type: ignore[arg-type]
    assert reported == [3]


async def test_a_pass_that_removed_nothing_reports_nothing() -> None:
    reported: list[int] = []
    await evict_once(ScriptedCache(0), reported.append)  # type: ignore[arg-type]
    assert reported == []


async def test_a_failing_pass_is_logged_and_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="app.cache.eviction"):
        await evict_once(ScriptedCache(RuntimeError("chroma exploded")), lambda n: None)  # type: ignore[arg-type]

    assert "eviction pass failed" in caplog.text


# --- the loop -----------------------------------------------------------------------


async def test_the_loop_runs_a_pass_every_interval() -> None:
    cache = ScriptedCache(3, 0, 2)
    reported: list[int] = []
    sleep = StepSleep(passes=3)

    with pytest.raises(StopLoop):
        await run_eviction_loop(
            cache, interval_seconds=INTERVAL, on_evicted=reported.append, sleep=sleep  # type: ignore[arg-type]
        )

    assert cache.passes == 3
    assert reported == [3, 2]
    assert sleep.calls == [INTERVAL] * 4


async def test_the_loop_waits_before_its_first_pass() -> None:
    """A gateway that has just started is not asked to do housekeeping at once."""
    cache = ScriptedCache(5)

    with pytest.raises(StopLoop):
        await run_eviction_loop(
            cache, interval_seconds=INTERVAL, on_evicted=lambda n: None, sleep=StepSleep(passes=0)  # type: ignore[arg-type]
        )

    assert cache.passes == 0


async def test_one_failed_pass_does_not_stop_the_schedule() -> None:
    cache = ScriptedCache(RuntimeError("chroma exploded"), 4)
    reported: list[int] = []

    with pytest.raises(StopLoop):
        await run_eviction_loop(
            cache, interval_seconds=INTERVAL, on_evicted=reported.append, sleep=StepSleep(passes=2)  # type: ignore[arg-type]
        )

    assert cache.passes == 2
    assert reported == [4]


async def test_a_failing_callback_does_not_stop_the_schedule() -> None:
    cache = ScriptedCache(1, 1)
    seen: list[int] = []

    def broken(count: int) -> None:
        seen.append(count)
        raise RuntimeError("metrics are broken")

    with pytest.raises(StopLoop):
        await run_eviction_loop(
            cache, interval_seconds=INTERVAL, on_evicted=broken, sleep=StepSleep(passes=2)  # type: ignore[arg-type]
        )

    assert seen == [1, 1]


# --- against the real cache --------------------------------------------------------------


@pytest.fixture
def real_cache(
    chroma_factory: Callable[[], Any],
    mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock,
    collection_name: str,
) -> SemanticCache:
    mapping_embedder.register("older prompt", base_vector())
    mapping_embedder.register("newer prompt", base_vector())
    return SemanticCache(
        chroma_factory,
        mapping_embedder,
        collection_name=collection_name,
        ttl_seconds=60,
        threshold=0.90,
        clock=fake_clock,
    )


async def test_the_loop_physically_removes_only_the_expired_entries(
    real_cache: SemanticCache,
    chroma_factory: Callable[[], Any],
    collection_name: str,
    fake_clock: FakeClock,
) -> None:
    """Reads filter expired entries regardless; this is about reclaiming the space."""
    await real_cache.add(MODEL, user("older prompt"), "old", BODY)
    fake_clock.advance(30)
    await real_cache.add(MODEL, user("newer prompt"), "new", BODY)
    stored = chroma_factory().get_collection(collection_name)
    assert stored.count() == 2

    # 40 seconds on, "old" (written at 0, ttl 60) is 70 old and "new" (written at 30) is 40.
    reported: list[int] = []
    with pytest.raises(StopLoop):
        await run_eviction_loop(
            real_cache,
            interval_seconds=40,
            on_evicted=reported.append,
            sleep=StepSleep(passes=1, on_wake=lambda seconds: fake_clock.advance(seconds)),
        )

    assert stored.get()["ids"] == ["new"]
    assert reported == [1]


async def test_a_vector_store_outage_costs_the_loop_nothing(
    mapping_embedder: MappingEmbedder, fake_clock: FakeClock, collection_name: str
) -> None:
    """Passes during an outage are no-ops, and they do not each try to reconnect."""
    attempts: list[int] = []

    def down() -> Any:
        attempts.append(1)
        raise ValueError("Could not connect to a Chroma server.")

    cache = SemanticCache(
        down,
        mapping_embedder,
        collection_name=collection_name,
        ttl_seconds=60,
        threshold=0.90,
        reconnect_seconds=30,
        clock=fake_clock,
    )
    reported: list[int] = []

    with pytest.raises(StopLoop):
        await run_eviction_loop(
            cache, interval_seconds=INTERVAL, on_evicted=reported.append, sleep=StepSleep(passes=3)
        )

    assert reported == []
    assert len(attempts) == 1


# --- starting and stopping ----------------------------------------------------------------


async def test_there_is_nothing_to_start_without_a_semantic_cache() -> None:
    assert start_eviction(None, interval_seconds=INTERVAL, on_evicted=lambda n: None) is None


@pytest.mark.parametrize("interval", [0, -1])
async def test_an_interval_of_zero_switches_eviction_off(interval: float) -> None:
    task = start_eviction(
        ScriptedCache(),  # type: ignore[arg-type]
        interval_seconds=interval,
        on_evicted=lambda n: None,
    )
    assert task is None


async def test_the_started_task_runs_in_real_time_and_stops_when_told() -> None:
    cache = ScriptedCache(*[2] * 100)
    reported: list[int] = []

    task = start_eviction(cache, interval_seconds=0.005, on_evicted=reported.append)  # type: ignore[arg-type]
    assert task is not None
    assert await until(lambda: len(reported) >= 2)

    await stop_eviction(task)
    assert task.done()
    passes = cache.passes
    await asyncio.sleep(0.05)
    assert cache.passes == passes


async def test_stopping_nothing_is_fine() -> None:
    await stop_eviction(None)


async def test_stopping_twice_is_fine() -> None:
    task = start_eviction(ScriptedCache(), interval_seconds=10, on_evicted=lambda n: None)  # type: ignore[arg-type]
    await stop_eviction(task)
    await stop_eviction(task)


# --- the lifespan -------------------------------------------------------------------------------


def build_app(settings: Settings, fake_redis: aioredis.FakeRedis, **state: Any) -> Any:
    app = main.create_app()
    app.state.settings = settings
    app.state.redis = fake_redis
    app.state.provider = StubProvider()
    for name, value in state.items():
        setattr(app.state, name, value)
    return app


def wait_for(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_the_lifespan_runs_eviction_for_a_cache_it_built_and_feeds_the_metrics(
    settings: Settings, fake_redis: aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = ScriptedCache(*[2] * 1000)

    async def build(received: Settings) -> Any:
        return cache

    # Watching the task itself, not the loop: the test client closes its event loop
    # on exit and that would end the task whether or not the lifespan stopped it, so
    # a check that the passes ceased proves nothing.
    stopped: list[bool] = []
    real_stop = main.stop_eviction

    async def spy(task: Any) -> None:
        await real_stop(task)
        stopped.append(task is not None and task.done())

    monkeypatch.setattr(main, "build_semantic_cache", build)
    monkeypatch.setattr(main, "stop_eviction", spy)
    configured = settings.model_copy(
        update={"semantic_cache_enabled": True, "eviction_interval_seconds": 0.005}
    )

    with TestClient(build_app(configured, fake_redis)) as client:
        assert wait_for(lambda: client.get("/metrics").json()["tier2_evicted_total"] >= 4)

    assert stopped == [True], "the lifespan did not stop the task it started"


def test_the_lifespan_does_not_run_jobs_on_a_cache_a_test_supplied(
    settings: Settings, fake_redis: aioredis.FakeRedis
) -> None:
    supplied = ScriptedCache(*[1] * 1000)
    configured = settings.model_copy(
        update={"semantic_cache_enabled": True, "eviction_interval_seconds": 0.005}
    )

    with TestClient(build_app(configured, fake_redis, semantic_cache=supplied)):
        time.sleep(0.1)

    assert supplied.passes == 0


def test_an_interval_of_zero_means_the_lifespan_starts_no_task(
    settings: Settings, fake_redis: aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = ScriptedCache(*[1] * 1000)

    async def build(received: Settings) -> Any:
        return cache

    monkeypatch.setattr(main, "build_semantic_cache", build)
    configured = settings.model_copy(
        update={"semantic_cache_enabled": True, "eviction_interval_seconds": 0}
    )

    with TestClient(build_app(configured, fake_redis)):
        time.sleep(0.1)

    assert cache.passes == 0


# --- the setting ---------------------------------------------------------------------------------


def test_the_default_interval_is_five_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEMCACHE_EVICTION_INTERVAL_SECONDS", raising=False)
    assert Settings(_env_file=None).eviction_interval_seconds == 300.0  # type: ignore[call-arg]


def test_zero_is_allowed_because_it_means_never() -> None:
    assert Settings(_env_file=None, eviction_interval_seconds=0).eviction_interval_seconds == 0  # type: ignore[call-arg]


def test_a_negative_interval_is_rejected_at_startup() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, eviction_interval_seconds=-1)  # type: ignore[call-arg]
