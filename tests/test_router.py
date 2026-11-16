# Tests for the pure routing function.
#
# choose_model has no I/O and no state, so every case is table-driven.
# The Settings object is constructed directly rather than loaded from the
# environment, keeping these tests hermetic and fast.

from __future__ import annotations

import pytest

from app.config import Settings
from app.models import Message
from app.router import RoutingResult, choose_model

_BASE = Settings(
    redis_url="redis://localhost:6379/0",
    cache_ttl_seconds=2,
    primary_provider="stub",
)
_CHEAP = "gpt-4o-mini"
_FRONTIER = "gpt-4o"


def _settings(*, enabled: bool, cheap: str | None = None, max_tokens: int = 512) -> Settings:
    return _BASE.model_copy(
        update={
            "routing_enabled": enabled,
            "routing_cheap_model": cheap,
            "routing_simple_max_tokens": max_tokens,
        }
    )


def _msg(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


# --- routing disabled ---------------------------------------------------------


def test_routing_disabled_returns_requested_model() -> None:
    """When routing is off, the requested model is always returned unchanged."""
    settings = _settings(enabled=False)
    result = choose_model(_FRONTIER, [_msg("hello")], settings)
    assert result == RoutingResult(model=_FRONTIER, reason="unchanged")


def test_routing_disabled_regardless_of_prompt_length() -> None:
    """A very short prompt still keeps the requested model when routing is off."""
    settings = _settings(enabled=False)
    result = choose_model(_FRONTIER, [_msg("hi")], settings)
    assert result.reason == "unchanged"
    assert result.model == _FRONTIER


# --- routing enabled, simple prompts -----------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "What is the capital of France?",
        "hi",
        "ping",
    ],
)
def test_simple_single_user_message_routes_to_cheap(content: str) -> None:
    """A single short user message with no code routes to the cheap model."""
    settings = _settings(enabled=True, cheap=_CHEAP)
    result = choose_model(_FRONTIER, [_msg(content)], settings)
    assert result == RoutingResult(model=_CHEAP, reason="routed_cheap")


def test_routing_uses_requested_model_as_key_identity() -> None:
    """The returned model is the cheap model, but the requested model is not lost.

    The caller is expected to use the original requested model as the cache
    key; this test just confirms choose_model carries the routing reason so
    the caller can record both.
    """
    settings = _settings(enabled=True, cheap=_CHEAP)
    result = choose_model(_FRONTIER, [_msg("short")], settings)
    assert result.model == _CHEAP
    assert result.reason == "routed_cheap"


# --- routing enabled, non-simple prompts -------------------------------------


def test_multi_turn_keeps_requested_model() -> None:
    """Multi-turn conversations stay on the requested model."""
    settings = _settings(enabled=True, cheap=_CHEAP)
    messages = [_msg("Question one"), _msg("Answer one", "assistant"), _msg("Follow up")]
    result = choose_model(_FRONTIER, messages, settings)
    assert result == RoutingResult(model=_FRONTIER, reason="unchanged")


def test_code_fence_keeps_requested_model() -> None:
    """A prompt with a code fence is not simple, regardless of length."""
    settings = _settings(enabled=True, cheap=_CHEAP)
    result = choose_model(_FRONTIER, [_msg("Fix this:\n```python\nprint('x')\n```")], settings)
    assert result == RoutingResult(model=_FRONTIER, reason="unchanged")


def test_long_prompt_keeps_requested_model() -> None:
    """A prompt over the token ceiling stays on the requested model."""
    settings = _settings(enabled=True, cheap=_CHEAP, max_tokens=5)
    # 100 chars / 4 = 25 tokens, well over the ceiling of 5
    long = "a" * 100
    result = choose_model(_FRONTIER, [_msg(long)], settings)
    assert result == RoutingResult(model=_FRONTIER, reason="unchanged")


def test_assistant_only_message_keeps_requested_model() -> None:
    """A single assistant message (no user turn) is not simple."""
    settings = _settings(enabled=True, cheap=_CHEAP)
    result = choose_model(_FRONTIER, [_msg("Sure.", "assistant")], settings)
    assert result == RoutingResult(model=_FRONTIER, reason="unchanged")


def test_empty_message_list_keeps_requested_model() -> None:
    """Zero messages is not a single user message, so routing does not fire."""
    settings = _settings(enabled=True, cheap=_CHEAP)
    result = choose_model(_FRONTIER, [], settings)
    assert result == RoutingResult(model=_FRONTIER, reason="unchanged")


# --- token estimation ---------------------------------------------------------


def test_token_ceiling_is_exclusive() -> None:
    """A prompt whose estimated tokens equal the ceiling is NOT simple."""
    # 4 chars per token; ceiling = 4 means exactly 4 chars = 1 token < 4, routes cheap
    # but 16 chars = 4 tokens, which equals ceiling → NOT simple
    settings = _settings(enabled=True, cheap=_CHEAP, max_tokens=4)
    exact = "a" * 16  # 16 / 4 = 4 == max_tokens, not strictly less
    result = choose_model(_FRONTIER, [_msg(exact)], settings)
    assert result == RoutingResult(model=_FRONTIER, reason="unchanged")


def test_token_just_under_ceiling_routes_cheap() -> None:
    """A prompt one token under the ceiling is routed to the cheap model."""
    settings = _settings(enabled=True, cheap=_CHEAP, max_tokens=4)
    just_under = "a" * 12  # 12 / 4 = 3 < 4
    result = choose_model(_FRONTIER, [_msg(just_under)], settings)
    assert result == RoutingResult(model=_CHEAP, reason="routed_cheap")
