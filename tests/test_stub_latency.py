"""The stub's latency: off by default, and when set, the stub really takes that long."""

from __future__ import annotations

import time

import pytest

from app.config import Settings
from app.models import ChatCompletionRequest, Message
from app.providers.factory import build_provider
from app.providers.stub import StubProvider

REQUEST = ChatCompletionRequest(model="gpt-4o-mini", messages=[Message(role="user", content="hi")])


async def _timed(provider: StubProvider) -> float:
    started = time.perf_counter()
    await provider.complete(REQUEST)
    return (time.perf_counter() - started) * 1000


async def test_by_default_the_stub_answers_at_once() -> None:
    assert await _timed(StubProvider()) < 20


async def test_with_a_latency_the_stub_takes_that_long() -> None:
    assert await _timed(StubProvider(latency_ms=120)) >= 115


def test_the_setting_is_off_by_default_and_cannot_be_negative() -> None:
    assert Settings(_env_file=None).stub_latency_ms == 0  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Settings(_env_file=None, stub_latency_ms=-1)  # type: ignore[call-arg]


async def test_the_gateway_builds_the_stub_with_the_configured_latency() -> None:
    settings = Settings(_env_file=None, stub_latency_ms=120)  # type: ignore[call-arg]
    provider = build_provider("stub", settings, None)
    assert isinstance(provider, StubProvider)
    assert await _timed(provider) >= 115
