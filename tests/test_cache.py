# Tier 1 cache tests.
#
# The TTL preservation case is the most important test in the slice. Without it,
# a plain SET on the hit path would silently resurrect every entry for a full
# fresh lifetime, producing the sliding expiry the design explicitly rejects,
# with no error and no other failing test.

from __future__ import annotations

import asyncio

import pytest
from fakeredis import aioredis

from app.cache.exact_cache import ExactCache
from app.cache.keys import make_key
from app.models import Message

BODY = {"choices": [{"message": {"role": "assistant", "content": "Paris."}}]}


def user(content: str) -> list[Message]:
    """One user message."""
    return [Message(role="user", content=content)]


async def write(cache: ExactCache, messages: list[Message], model: str = "gpt-4o-mini") -> None:
    """Seed one entry, with the argument shape the route will use."""
    await cache.set(model, messages, BODY, prompt_tokens=7, completion_tokens=3)


@pytest.fixture
def cache(fake_redis: aioredis.FakeRedis) -> ExactCache:
    """A cache with a short TTL, so expiry is testable without a long sleep."""
    return ExactCache(fake_redis, ttl_seconds=2)


# --- hit and miss ---------------------------------------------------------


async def test_miss_on_empty_cache(cache: ExactCache) -> None:
    """Nothing stored means nothing returned."""
    assert await cache.get("gpt-4o-mini", user("hello")) is None


async def test_identical_prompt_hits(cache: ExactCache) -> None:
    """TEST-001. The second identical request is served without a provider."""
    await write(cache, user("What is the capital of France?"))
    entry = await cache.get("gpt-4o-mini", user("What is the capital of France?"))
    assert entry is not None
    assert entry.response_body == BODY


async def test_whitespace_difference_still_hits(cache: ExactCache) -> None:
    """TEST-003. Formatting noise must not cost a provider call."""
    await write(cache, user("What is the capital of France?"))
    entry = await cache.get("gpt-4o-mini", user("  What is   the\tcapital\nof France?  "))
    assert entry is not None


async def test_case_difference_misses(cache: ExactCache) -> None:
    """Case is preserved at Tier 1 by design, so this falls through to Tier 2."""
    await write(cache, user("Hello"))
    assert await cache.get("gpt-4o-mini", user("hello")) is None


async def test_different_model_misses(cache: ExactCache) -> None:
    """TEST-002. The model is part of the key, so this is a separate entry."""
    await write(cache, user("hello"), model="gpt-4o-mini")
    assert await cache.get("gpt-4o", user("hello")) is None


# --- the TTL contract -----------------------------------------------------


async def test_ttl_is_applied_at_write(cache: ExactCache, fake_redis: aioredis.FakeRedis) -> None:
    """An entry is written with a bounded life, not forever."""
    await write(cache, user("hello"))
    ttl = await fake_redis.pttl(make_key("gpt-4o-mini", user("hello")))
    assert 0 < ttl <= 2000


async def test_hit_does_not_reset_ttl(cache: ExactCache, fake_redis: aioredis.FakeRedis) -> None:
    """The load bearing test of this module.

    TTL represents freshness, not popularity. A sliding expiry would let a
    frequently requested answer live indefinitely, which is precisely where a
    stale answer does the most damage. Serving a hit must therefore leave the
    remaining life strictly no longer than it was.
    """
    messages = user("hello")
    key = make_key("gpt-4o-mini", messages)
    await write(cache, messages)

    # Let real time pass first, so the remaining life is measurably below the
    # full TTL. Measuring before any time has elapsed would make the assertion
    # vacuous: a reset would land on the same number it started at and pass.
    await asyncio.sleep(0.25)
    before = await fake_redis.pttl(key)
    assert before < 2000, "the clock did not move, so this test proves nothing"

    assert await cache.get("gpt-4o-mini", messages) is not None
    after = await fake_redis.pttl(key)

    assert after <= before, f"TTL grew on a hit: {before} then {after}"
    assert after > 0


async def test_repeated_hits_never_extend_ttl(
    cache: ExactCache, fake_redis: aioredis.FakeRedis
) -> None:
    """The hot entry case, which is where sliding expiry would do real harm."""
    messages = user("hello")
    key = make_key("gpt-4o-mini", messages)
    await write(cache, messages)

    await asyncio.sleep(0.1)
    previous = await fake_redis.pttl(key)
    assert previous < 2000

    for _ in range(5):
        await asyncio.sleep(0.1)
        await cache.get("gpt-4o-mini", messages)
        current = await fake_redis.pttl(key)
        assert current <= previous, f"TTL grew on a hit: {previous} then {current}"
        previous = current


@pytest.mark.slow
async def test_entry_past_ttl_is_a_miss(fake_redis: aioredis.FakeRedis) -> None:
    """TEST-009. One real sleep, which is honest: expiry is a time based property."""
    cache = ExactCache(fake_redis, ttl_seconds=1)
    await write(cache, user("hello"))
    await asyncio.sleep(1.1)
    assert await cache.get("gpt-4o-mini", user("hello")) is None


# --- hit accounting -------------------------------------------------------


async def test_hit_count_starts_at_zero(cache: ExactCache) -> None:
    """A freshly written entry has not been hit."""
    entry = await cache.set("gpt-4o-mini", user("hello"), BODY, 7, 3)
    assert entry.hit_count == 0
    assert entry.last_hit_at is None


async def test_hit_count_increments(cache: ExactCache) -> None:
    """Every hit is counted, which is what the cache stats endpoint will report."""
    messages = user("hello")
    await write(cache, messages)
    for expected in (1, 2, 3):
        entry = await cache.get("gpt-4o-mini", messages)
        assert entry is not None
        assert entry.hit_count == expected


async def test_last_hit_at_is_recorded(cache: ExactCache) -> None:
    """Set on a hit, so a stale popular entry is identifiable during evaluation."""
    messages = user("hello")
    await write(cache, messages)
    entry = await cache.get("gpt-4o-mini", messages)
    assert entry is not None
    assert entry.last_hit_at is not None


async def test_many_hits_are_all_counted(cache: ExactCache) -> None:
    """Twenty hits produce the counts one through twenty, none lost or repeated.

    Note what this does NOT prove. It cannot demonstrate that HINCRBY is atomic:
    fakeredis runs in the same event loop as the test, so gathered coroutines
    never genuinely interleave inside the client, and a non-atomic read, modify,
    write passes this too. Verified by mutation.

    Atomicity is a property of HINCRBY itself and is the reason the entry is a
    hash rather than a JSON string, but demonstrating it needs a real Redis and
    real concurrency. That belongs with the load test, not here. This case is a
    regression guard on counting, and is documented as no more than that.
    """
    messages = user("hello")
    await write(cache, messages)
    results = await asyncio.gather(*(cache.get("gpt-4o-mini", messages) for _ in range(20)))
    assert all(entry is not None for entry in results)
    counts = sorted(entry.hit_count for entry in results if entry is not None)
    assert counts == list(range(1, 21))


# --- removal --------------------------------------------------------------


async def test_delete_removes_the_entry(cache: ExactCache) -> None:
    """Needed by the flush endpoint and to force a cold start between runs."""
    messages = user("hello")
    await write(cache, messages)
    assert await cache.delete("gpt-4o-mini", messages) is True
    assert await cache.get("gpt-4o-mini", messages) is None


async def test_delete_on_a_miss_reports_nothing_removed(cache: ExactCache) -> None:
    """Deleting what was never there is not an error, but it is not a removal."""
    assert await cache.delete("gpt-4o-mini", user("never stored")) is False
