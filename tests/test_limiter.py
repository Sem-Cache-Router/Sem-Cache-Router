# Token bucket tests.
#
# The scripts have their own tests, on the arithmetic. These cover the layer over
# them: that the bucket names its keys safely, hands the scripts the right
# arguments, reads their answers into the right types, and settles a reservation
# in both directions. The two tests that used to be skipped here, for want of a
# fake Redis that ran Lua, are the first two below, and they run now.

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
from fakeredis import aioredis

from app.limiter.bucket import Reserve, ReserveStatus, TokenBucket
from tests.conftest import FakeClock

CAPACITY = 100
RATE = 10
CALLER = "caller-key"
WINDOW = "tokens"


@pytest.fixture
def bucket(fake_redis: aioredis.FakeRedis, fake_clock: FakeClock) -> TokenBucket:
    return TokenBucket(fake_redis, capacity=CAPACITY, refill_rate=RATE, clock=fake_clock)


# --- reserving ------------------------------------------------------------------------


async def test_a_request_within_the_budget_is_admitted(bucket: TokenBucket) -> None:
    assert await bucket.reserve(CALLER, WINDOW, 60) == Reserve(ReserveStatus.ADMITTED, 40, 0)


async def test_a_request_over_the_budget_is_refused_with_a_retry_hint(bucket: TokenBucket) -> None:
    await bucket.reserve(CALLER, WINDOW, 60)

    refused = await bucket.reserve(CALLER, WINDOW, 50)

    assert refused.status is ReserveStatus.DENIED
    assert refused.admitted is False
    assert refused.remaining == 40
    assert refused.retry_after_seconds == 1, "10 short, at 10 a second"


async def test_tokens_come_back_as_time_passes(bucket: TokenBucket, fake_clock: FakeClock) -> None:
    await bucket.reserve(CALLER, WINDOW, CAPACITY)
    assert not (await bucket.reserve(CALLER, WINDOW, 30)).admitted

    fake_clock.advance(3)

    assert (await bucket.reserve(CALLER, WINDOW, 30)).admitted


async def test_a_request_larger_than_the_bucket_can_ever_hold_is_flagged_as_such(
    bucket: TokenBucket,
) -> None:
    """Not a denial with a retry hint: no amount of waiting would make it fit."""
    result = await bucket.reserve(CALLER, WINDOW, CAPACITY + 1)

    assert result.status is ReserveStatus.TOO_LARGE
    assert result.admitted is False


async def test_callers_have_separate_buckets(bucket: TokenBucket) -> None:
    await bucket.reserve("one", WINDOW, CAPACITY)

    assert not (await bucket.reserve("one", WINDOW, 1)).admitted
    assert (await bucket.reserve("two", WINDOW, CAPACITY)).admitted


async def test_windows_are_separate_buckets_for_the_same_caller(bucket: TokenBucket) -> None:
    await bucket.reserve(CALLER, "minute", CAPACITY)

    assert (await bucket.reserve(CALLER, "day", CAPACITY)).admitted


async def test_concurrent_requests_cannot_overspend_one_bucket(bucket: TokenBucket) -> None:
    """TEST-011, through the bucket: twenty at once, room for ten."""
    results = await asyncio.gather(*[bucket.reserve(CALLER, WINDOW, 10) for _ in range(20)])

    assert sum(result.admitted for result in results) == 10


async def test_without_a_clock_the_scripts_use_redis_time(
    fake_redis: aioredis.FakeRedis,
) -> None:
    """Production passes no clock. A trickle refill means nothing is earned while this runs."""
    production = TokenBucket(fake_redis, capacity=CAPACITY, refill_rate=0.001)

    assert (await production.reserve(CALLER, WINDOW, 60)).admitted
    refused = await production.reserve(CALLER, WINDOW, 50)

    assert not refused.admitted and refused.remaining == 40


# --- how the bucket is named ----------------------------------------------------------


async def test_the_callers_key_never_appears_in_redis(
    bucket: TokenBucket, fake_redis: aioredis.FakeRedis
) -> None:
    """The key is a credential, and Redis keys are visible to anyone who can list them."""
    secret = "sk-live-this-must-not-be-readable"
    await bucket.reserve(secret, WINDOW, 10)

    keys = [key.decode() if isinstance(key, bytes) else key for key in await fake_redis.keys("*")]

    assert len(keys) == 1 and keys[0].startswith("rl:")
    assert all(secret not in key for key in keys)


async def test_the_same_caller_always_reaches_the_same_bucket(bucket: TokenBucket) -> None:
    await bucket.reserve(CALLER, WINDOW, 60)

    assert (await bucket.reserve(CALLER, WINDOW, 10)).remaining == 30


def test_different_callers_get_different_keys() -> None:
    assert TokenBucket.key("one", WINDOW) != TokenBucket.key("two", WINDOW)


def test_a_key_with_awkward_characters_is_still_one_well_formed_key() -> None:
    key = TokenBucket.key("a:b c\nd" + "x" * 5000, WINDOW)

    assert key.startswith("rl:") and key.endswith(f":{WINDOW}")
    assert len(key) < 80 and "\n" not in key and " " not in key


# --- settling a reservation -----------------------------------------------------------


async def test_a_released_reservation_is_returned_in_full(bucket: TokenBucket) -> None:
    await bucket.reserve(CALLER, WINDOW, CAPACITY)

    await bucket.release(CALLER, WINDOW, CAPACITY)

    assert (await bucket.reserve(CALLER, WINDOW, CAPACITY)).admitted


async def test_a_credit_is_capped_at_the_buckets_own_capacity(bucket: TokenBucket) -> None:
    """The bucket passes its capacity to the script, which is what the cap is measured against."""
    await bucket.reserve(CALLER, WINDOW, 10)

    await bucket.release(CALLER, WINDOW, 10_000)

    assert await bucket.adjust(CALLER, WINDOW, 0) == CAPACITY


async def test_reconciling_a_cheaper_reply_returns_the_difference(bucket: TokenBucket) -> None:
    """TEST-012."""
    await bucket.reserve(CALLER, WINDOW, 60)

    await bucket.reconcile(CALLER, WINDOW, estimate=60, actual=30)

    assert await bucket.adjust(CALLER, WINDOW, 0) == 70


async def test_reconciling_a_dearer_reply_takes_the_difference(bucket: TokenBucket) -> None:
    """The old reconcile only ever credited, so a caller could spend past their estimate free."""
    await bucket.reserve(CALLER, WINDOW, 60)

    await bucket.reconcile(CALLER, WINDOW, estimate=60, actual=90)

    assert await bucket.adjust(CALLER, WINDOW, 0) == 10


async def test_reconciling_an_exact_estimate_changes_nothing(bucket: TokenBucket) -> None:
    await bucket.reserve(CALLER, WINDOW, 60)

    await bucket.reconcile(CALLER, WINDOW, estimate=60, actual=60)

    assert await bucket.adjust(CALLER, WINDOW, 0) == 40


async def test_settling_a_caller_with_no_bucket_reports_that_there_is_none(
    bucket: TokenBucket,
) -> None:
    assert await bucket.adjust("never-seen", WINDOW, 50) == -1


# --- construction ---------------------------------------------------------------------


@pytest.mark.parametrize(("capacity", "rate"), [(0, 10), (-1, 10), (100, 0), (100, -1)])
def test_a_bucket_that_could_never_work_is_refused(
    fake_redis: aioredis.FakeRedis, capacity: int, rate: float
) -> None:
    with pytest.raises(ValueError, match="positive"):
        TokenBucket(fake_redis, capacity=capacity, refill_rate=rate)


def test_the_scripts_load_whatever_directory_the_process_starts_in(tmp_path: Path) -> None:
    """The scripts used to be opened by a path relative to the working directory."""
    root = Path(__file__).resolve().parents[1]

    done = subprocess.run(
        [sys.executable, "-c", "import app.limiter.bucket"],
        cwd=tmp_path,
        env={"PYTHONPATH": str(root), "PATH": ""},
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stderr
