# Admission controller tests.
#
# The bucket and its scripts have their own tests. These cover the conversation the
# request path has with them: what an admission carries, how a reservation is given
# back or settled, and above all the two opposite rules for failure. Asking to be
# admitted while the limiter is down must fail, closed. Settling afterwards while it
# is down must not, because the caller already has their answer.

from __future__ import annotations

import logging
from typing import Any

import pytest
import redis.exceptions
from fakeredis import aioredis

from app.limiter.admission import (
    DEFAULT_WINDOW,
    Admission,
    AdmissionController,
    LimiterUnavailable,
    Reservation,
)
from app.limiter.bucket import ReserveStatus, TokenBucket
from tests.conftest import FakeClock

CAPACITY = 100
RATE = 10
CALLER = "caller-key"


@pytest.fixture
def controller(fake_redis: aioredis.FakeRedis, fake_clock: FakeClock) -> AdmissionController:
    bucket = TokenBucket(fake_redis, capacity=CAPACITY, refill_rate=RATE, clock=fake_clock)
    return AdmissionController(bucket)


class DownRedis:
    """A Redis whose every script call fails the way a lost connection does."""

    def __init__(self, failure: BaseException) -> None:
        self._failure = failure

    def register_script(self, script: str) -> Any:
        async def call(*args: Any, **kwargs: Any) -> Any:
            raise self._failure

        return call


def controller_over(failure: BaseException) -> AdmissionController:
    return AdmissionController(TokenBucket(DownRedis(failure), CAPACITY, RATE))  # type: ignore[arg-type]


LOST_CONNECTION = [
    redis.exceptions.ConnectionError("connection refused"),
    redis.exceptions.TimeoutError("no reply"),
    ConnectionResetError("reset by peer"),
]


# --- the decision ----------------------------------------------------------------------------


async def test_a_request_within_the_budget_is_admitted_with_a_reservation(
    controller: AdmissionController,
) -> None:
    admission = await controller.reserve(CALLER, 60)

    assert admission == Admission(ReserveStatus.ADMITTED, Reservation(CALLER, 60), 40, 0)
    assert admission.admitted is True


async def test_a_request_over_the_budget_is_refused_with_no_reservation_and_a_retry_hint(
    controller: AdmissionController,
) -> None:
    await controller.reserve(CALLER, 60)

    refused = await controller.reserve(CALLER, 50)

    assert refused.admitted is False
    assert refused.reservation is None
    assert refused.status is ReserveStatus.DENIED
    assert refused.retry_after_seconds == 1


async def test_a_refusal_spends_nothing(controller: AdmissionController) -> None:
    await controller.reserve(CALLER, 60)
    await controller.reserve(CALLER, 50)

    assert (await controller.reserve(CALLER, 40)).admitted


async def test_a_request_the_bucket_could_never_hold_is_flagged_not_told_to_wait(
    controller: AdmissionController,
) -> None:
    admission = await controller.reserve(CALLER, CAPACITY + 1)

    assert admission.status is ReserveStatus.TOO_LARGE
    assert admission.admitted is False
    assert admission.retry_after_seconds == 0


async def test_callers_are_budgeted_separately(controller: AdmissionController) -> None:
    await controller.reserve("one", CAPACITY)

    assert not (await controller.reserve("one", 1)).admitted
    assert (await controller.reserve("two", CAPACITY)).admitted


async def test_windows_are_budgeted_separately(
    fake_redis: aioredis.FakeRedis, fake_clock: FakeClock
) -> None:
    bucket = TokenBucket(fake_redis, capacity=CAPACITY, refill_rate=RATE, clock=fake_clock)
    minute, day = AdmissionController(bucket, "minute"), AdmissionController(bucket, "day")
    await minute.reserve(CALLER, CAPACITY)

    assert (await day.reserve(CALLER, CAPACITY)).admitted


def test_the_default_window_is_a_single_named_budget() -> None:
    assert DEFAULT_WINDOW == "tokens"


# --- settling ------------------------------------------------------------------------------------


async def test_a_released_reservation_is_given_back_in_full(
    controller: AdmissionController,
) -> None:
    admission = await controller.reserve(CALLER, CAPACITY)
    assert admission.reservation is not None

    await controller.release(admission.reservation)

    assert (await controller.reserve(CALLER, CAPACITY)).admitted


async def test_a_cheaper_reply_gives_back_the_difference(controller: AdmissionController) -> None:
    """TEST-012: reserved 60, used 30, so 70 is left."""
    admission = await controller.reserve(CALLER, 60)
    assert admission.reservation is not None

    await controller.reconcile(admission.reservation, actual_tokens=30)

    assert (await controller.reserve(CALLER, 70)).admitted
    assert not (await controller.reserve(CALLER, 1)).admitted


async def test_a_dearer_reply_takes_the_difference(controller: AdmissionController) -> None:
    admission = await controller.reserve(CALLER, 60)
    assert admission.reservation is not None

    await controller.reconcile(admission.reservation, actual_tokens=90)

    assert (await controller.reserve(CALLER, 10)).admitted
    assert not (await controller.reserve(CALLER, 1)).admitted


# --- when the limiter is down ---------------------------------------------------------------------


@pytest.mark.parametrize("failure", LOST_CONNECTION)
async def test_admission_fails_closed_when_the_limiter_cannot_be_reached(
    failure: BaseException,
) -> None:
    """TEST-016. Admitting unmetered during an outage would make the budget escapable."""
    with pytest.raises(LimiterUnavailable) as raised:
        await controller_over(failure).reserve(CALLER, 10)

    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("failure", LOST_CONNECTION)
async def test_a_failed_release_does_not_fail_the_request_and_says_so(
    failure: BaseException, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="app.limiter.admission"):
        await controller_over(failure).release(Reservation(CALLER, 40))

    assert "could not release 40 tokens" in caplog.text


@pytest.mark.parametrize("failure", LOST_CONNECTION)
async def test_a_failed_settlement_does_not_fail_the_request_and_says_so(
    failure: BaseException, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="app.limiter.admission"):
        await controller_over(failure).reconcile(Reservation(CALLER, 40), actual_tokens=25)

    assert "could not settle a reservation of 40 tokens against 25 used" in caplog.text


async def test_a_programming_error_is_not_mistaken_for_an_outage() -> None:
    """Only a lost connection is the limiter being unavailable; a bug must be seen as one."""
    with pytest.raises(ValueError, match="a bug"):
        await controller_over(ValueError("a bug")).reserve(CALLER, 10)


async def test_cancellation_is_never_swallowed() -> None:
    import asyncio

    with pytest.raises(asyncio.CancelledError):
        await controller_over(asyncio.CancelledError()).release(Reservation(CALLER, 10))
