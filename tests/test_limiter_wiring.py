"""The rate limiter in the request path: admission, refusals and settling, end to end.

The real AdmissionController and TokenBucket run over fakeredis executing the
real Lua scripts, behind the real app, so these tests see exactly what a caller
would: status codes, headers and the counters.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fakeredis import aioredis
from fastapi.testclient import TestClient

from app.cache.exact_cache import ExactCache
from app.config import Settings
from app.limiter.admission import Admission, AdmissionController, LimiterUnavailable, Reservation
from app.limiter.bucket import ReserveStatus, TokenBucket
from app.main import build_limiter, create_app
from app.providers.stub import StubProvider
from tests.conftest import FakeClock, router_around

AUTH = {"Authorization": "Bearer caller-key"}


def body(content: str, max_tokens: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": "gpt-4o-mini",
                               "messages": [{"role": "user", "content": content}]}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def app_with(settings: Settings, redis: aioredis.FakeRedis, limiter: Any) -> Any:
    app = create_app()
    app.state.settings = settings
    app.state.redis = redis
    app.state.provider = StubProvider()
    app.state.cache = ExactCache(redis, settings.cache_ttl_seconds)
    app.state.limiter = limiter
    return app


def real_limiter(
    redis: aioredis.FakeRedis, clock: FakeClock, capacity: int
) -> AdmissionController:
    # A refill of one token a second, so a refusal's wait is long enough to see.
    return AdmissionController(TokenBucket(redis, capacity, 1.0, clock=clock))


REPLY = 10  # the reply allowance the tests configure, so budgets stay small and exact


@pytest.fixture
def small(settings: Settings) -> Settings:
    return settings.model_copy(update={"limiter_default_completion_tokens": REPLY})


@pytest.fixture
def small_budget(
    small: Settings, fake_redis: aioredis.FakeRedis, fake_clock: FakeClock
) -> Iterator[TestClient]:
    with TestClient(app_with(small, fake_redis, real_limiter(fake_redis, fake_clock, 40))) as c:
        yield c


def test_an_admitted_request_reports_the_tokens_left(small_budget: TestClient) -> None:
    response = small_budget.post("/v1/chat/completions", json=body("hi"), headers=AUTH)
    assert response.status_code == 200
    remaining = int(response.headers["x-ratelimit-remaining-tokens"])
    assert 0 <= remaining < 40


def test_a_caller_over_budget_gets_429_with_a_retry_after(small_budget: TestClient) -> None:
    statuses = []
    for n in range(20):
        response = small_budget.post("/v1/chat/completions",
                                     json=body(f"a new question number {n}"), headers=AUTH)
        statuses.append(response.status_code)
        if response.status_code == 429:
            break
    assert statuses[0] == 200 and statuses[-1] == 429
    assert response.json()["error"]["code"] == "TOKEN_BUDGET_EXCEEDED"
    assert int(response.headers["retry-after"]) >= 1
    assert small_budget.get("/metrics").json()["throttled_requests"] == 1


def test_a_request_bigger_than_the_whole_budget_is_413_without_a_retry_hint(
    small_budget: TestClient,
) -> None:
    response = small_budget.post("/v1/chat/completions", json=body("hi", max_tokens=500),
                                 headers=AUTH)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    assert "retry-after" not in response.headers


def test_a_refused_request_never_reaches_the_provider(
    small: Settings, fake_redis: aioredis.FakeRedis, fake_clock: FakeClock
) -> None:
    provider = StubProvider()
    app = app_with(small, fake_redis, real_limiter(fake_redis, fake_clock, 40))
    with TestClient(app) as client:
        app.state.router = router_around(provider)
        client.post("/v1/chat/completions", json=body("hi", max_tokens=500), headers=AUTH)
    assert provider.call_count == 0


def test_callers_have_separate_budgets(small_budget: TestClient) -> None:
    for n in range(20):
        if small_budget.post("/v1/chat/completions", json=body(f"q {n} for one"),
                             headers=AUTH).status_code == 429:
            break
    other = {"Authorization": "Bearer another-caller"}
    response = small_budget.post("/v1/chat/completions", json=body("hi"), headers=other)
    assert response.status_code == 200


class RecordingLimiter:
    """Admits everything and records what the route does with each reservation."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.estimates: list[int] = []
        self.released: list[Reservation] = []
        self.reconciled: list[tuple[Reservation, int]] = []

    async def reserve(self, api_key: str, estimate: int) -> Admission:
        if self.fail:
            raise LimiterUnavailable("redis is down")
        self.estimates.append(estimate)
        return Admission(ReserveStatus.ADMITTED, Reservation(api_key, estimate), 100, 0)

    async def release(self, reservation: Reservation) -> None:
        self.released.append(reservation)

    async def reconcile(self, reservation: Reservation, actual: int) -> None:
        self.reconciled.append((reservation, actual))


@pytest.fixture
def recorded(
    settings: Settings, fake_redis: aioredis.FakeRedis
) -> Iterator[tuple[TestClient, RecordingLimiter]]:
    limiter = RecordingLimiter()
    with TestClient(app_with(settings, fake_redis, limiter)) as client:
        yield client, limiter


def test_a_miss_settles_against_the_tokens_the_provider_reported(
    recorded: tuple[TestClient, RecordingLimiter],
) -> None:
    client, limiter = recorded
    client.post("/v1/chat/completions", json=body("a question"), headers=AUTH)
    assert len(limiter.reconciled) == 1 and limiter.released == []
    assert limiter.reconciled[0][1] > 0


def test_a_cache_hit_gives_the_whole_reservation_back(
    recorded: tuple[TestClient, RecordingLimiter],
) -> None:
    client, limiter = recorded
    client.post("/v1/chat/completions", json=body("a question"), headers=AUTH)
    client.post("/v1/chat/completions", json=body("a question"), headers=AUTH)
    assert len(limiter.reconciled) == 1 and len(limiter.released) == 1


def test_a_failed_provider_call_gives_the_reservation_back(
    recorded: tuple[TestClient, RecordingLimiter],
) -> None:
    client, limiter = recorded
    client.app.state.router = router_around(StubProvider(fail_times=1))  # type: ignore[attr-defined]
    response = client.post("/v1/chat/completions", json=body("a question"), headers=AUTH)
    assert response.status_code == 502
    assert len(limiter.released) == 1 and limiter.reconciled == []


def test_an_unreachable_cache_gives_the_reservation_back(
    recorded: tuple[TestClient, RecordingLimiter], monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, limiter = recorded

    async def down(*args: object) -> None:
        raise ConnectionError("redis is down")

    monkeypatch.setattr(client.app.state.cache, "get", down)  # type: ignore[attr-defined]
    response = client.post("/v1/chat/completions", json=body("a question"), headers=AUTH)
    assert response.status_code == 503
    assert len(limiter.released) == 1


def test_an_unreachable_limiter_refuses_the_request(
    settings: Settings, fake_redis: aioredis.FakeRedis,
) -> None:
    provider = StubProvider()
    app = app_with(settings, fake_redis, RecordingLimiter(fail=True))
    with TestClient(app) as client:
        app.state.router = router_around(provider)
        response = client.post("/v1/chat/completions", json=body("hi"), headers=AUTH)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DEPENDENCY_UNAVAILABLE"
    assert provider.call_count == 0


def test_the_estimator_sets_the_reservation_when_there_is_one(
    recorded: tuple[TestClient, RecordingLimiter],
) -> None:
    client, limiter = recorded

    class FixedEstimator:
        async def estimate(self, messages: Any, *, max_tokens: int | None,
                           default_completion_tokens: int) -> int:
            return 1234 + (max_tokens or default_completion_tokens)

    client.app.state.estimator = FixedEstimator()  # type: ignore[attr-defined]
    client.post("/v1/chat/completions", json=body("hi", max_tokens=6), headers=AUTH)
    assert limiter.estimates == [1240]


async def test_concurrent_callers_never_spend_more_than_the_budget(
    small: Settings, fake_redis: aioredis.FakeRedis, fake_clock: FakeClock,
) -> None:
    # Twenty fresh questions of the same length reserve the same estimate each, so
    # exactly five fit a budget of five estimates, however the requests interleave.
    # The provider is slow, so every reservation is made before any is settled.
    estimate = len("concurrent question 00") // 4 + REPLY
    app = app_with(small, fake_redis, real_limiter(fake_redis, fake_clock, estimate * 5))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://gw") as client:
        with TestClient(app):  # runs the lifespan, which fills in the rest of app.state
            app.state.router = router_around(StubProvider(latency_ms=50))
            responses = await asyncio.gather(*(
                client.post("/v1/chat/completions",
                            json=body(f"concurrent question {n:02d}"), headers=AUTH)
                for n in range(20)
            ))
    statuses = [r.status_code for r in responses]
    assert statuses.count(200) == 5
    assert statuses.count(429) == 15


async def test_the_limiter_is_built_only_when_switched_on(
    settings: Settings, fake_redis: aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main

    class NoDownloadEstimator:
        loaded = False

        def load(self) -> None:
            NoDownloadEstimator.loaded = True

    monkeypatch.setattr(main, "TokenEstimator", NoDownloadEstimator)
    state = type("State", (), {"redis": fake_redis})()
    assert await build_limiter(state, settings) is None
    assert NoDownloadEstimator.loaded is False

    on = settings.model_copy(update={"limiter_enabled": True, "bucket_capacity": 500})
    limiter = await build_limiter(state, on)
    assert isinstance(limiter, AdmissionController)
    assert NoDownloadEstimator.loaded is True
    assert isinstance(state.estimator, NoDownloadEstimator)
