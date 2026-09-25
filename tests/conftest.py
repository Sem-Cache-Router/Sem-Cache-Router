# Shared fixtures.
#
# Why fakes rather than a live stack: the suite has to run with no
# infrastructure, or it will not run often enough to be useful.
#
# Grows as the slice grows. Today it carries what the cache tests need; the app
# level fixtures arrive with the routes.

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fakeredis import aioredis
from fastapi.testclient import TestClient

from app.cache.exact_cache import ExactCache
from app.config import Settings, get_settings
from app.main import create_app
from app.providers.stub import StubProvider


@pytest.fixture
def settings() -> Settings:
    """Test settings, constructed explicitly rather than resolved from the environment.

    Explicit because a developer's local .env would otherwise leak in and make
    a test pass on one machine and fail on another. Short TTL so expiry can be
    exercised without a long sleep.
    """
    return Settings(
        redis_url="redis://localhost:6379/0",
        cache_ttl_seconds=2,
        primary_provider="stub",
    )


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> AsyncIterator[None]:
    """Drop the cached Settings around every test.

    get_settings holds an lru_cache, which is process global. Without this, one
    test that touches the environment silently changes the next one.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def fake_redis() -> AsyncIterator[aioredis.FakeRedis]:
    """An in-memory Redis that speaks the async client interface.

    Function scoped, not session scoped. An async client binds to the event loop
    that was alive when it was created, and a session scoped one under function
    scoped loops produces intermittent "attached to a different loop" failures
    that look like flaky tests rather than a fixture bug.

    decode_responses=True throughout, so nothing in the codebase has to think
    about whether a key is bytes or str.
    """
    client = aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def stub_provider() -> StubProvider:
    """A provider that answers deterministically and never fails, unless asked."""
    return StubProvider()


@pytest_asyncio.fixture
async def client(
    fake_redis: aioredis.FakeRedis,
    stub_provider: StubProvider,
    settings: Settings,
) -> AsyncIterator[TestClient]:
    """A test client wired to the fakes, through the real app factory.

    State is seeded before the context is entered, so the lifespan finds every
    dependency already present and leaves the fakes alone. That exercises the
    real wiring rather than bypassing it, which is what makes these tests worth
    anything: a feature that works only when called directly from a test is not
    finished.

    The `with` is mandatory. TestClient runs the lifespan only inside a context
    manager, and without it app.state is never populated and every route fails
    with an attribute error that looks like a routing bug.
    """
    app = create_app()
    app.state.settings = settings
    app.state.redis = fake_redis
    app.state.provider = stub_provider
    app.state.cache = ExactCache(fake_redis, settings.cache_ttl_seconds)
    app.state.limiter = None

    with TestClient(app) as test_client:
        yield test_client
