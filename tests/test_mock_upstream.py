# Mock upstream tests.
#
# The mock is only useful if the real adapters accept what it sends, so the main
# test runs them against it, unmodified, over an in process transport. The rest
# pin the controls a demo and a benchmark depend on.

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from app.config import Settings
from app.models import ChatCompletionRequest, Message
from app.providers.anthropic_client import AnthropicClient
from app.providers.base import ProviderError
from app.providers.openai_client import OpenAIClient
from benchmarks.mock_upstream import create_app, reply_text, token_count


@pytest.fixture
def mock() -> TestClient:
    return TestClient(create_app())


@pytest_asyncio.fixture
async def adapters() -> AsyncIterator[tuple[OpenAIClient, AnthropicClient, TestClient]]:
    """The real adapters, pointed at a fresh mock over an in process transport."""
    upstream = create_app()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream), base_url="http://mock")
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        OPENAI_API_KEY="k",
        ANTHROPIC_API_KEY="k",
        openai_base_url="http://mock/v1",
        anthropic_base_url="http://mock/v1",
    )
    with TestClient(upstream) as control:
        yield OpenAIClient(client, settings), AnthropicClient(client, settings), control
    await client.aclose()


def ask(model: str = "gpt-4o-mini", content: str = "What is the capital of France?") -> Any:
    return ChatCompletionRequest(model=model, messages=[Message(role="user", content=content)])


# --- the real adapters accept what it sends ---------------------------------------------------


async def test_the_openai_adapter_reads_the_mocks_reply(
    adapters: tuple[OpenAIClient, AnthropicClient, TestClient],
) -> None:
    openai, _, _ = adapters

    reply = await openai.complete(ask())

    assert reply.message.content == reply_text("What is the capital of France?")
    assert reply.usage.prompt_tokens == token_count("What is the capital of France?")
    assert reply.cost_usd > 0, "a priced call, with no key and no spend"


async def test_the_anthropic_adapter_reads_the_mocks_reply(
    adapters: tuple[OpenAIClient, AnthropicClient, TestClient],
) -> None:
    _, anthropic, _ = adapters

    reply = await anthropic.complete(ask("claude-3-haiku-20240307"))

    assert reply.message.content == reply_text("What is the capital of France?")
    assert reply.finish_reason == "stop"
    assert reply.cost_usd > 0


async def test_the_same_prompt_always_gets_the_same_answer(
    adapters: tuple[OpenAIClient, AnthropicClient, TestClient],
) -> None:
    openai, _, _ = adapters
    first = await openai.complete(ask())
    second = await openai.complete(ask())
    other = await openai.complete(ask(content="something else entirely"))

    assert first.message.content == second.message.content != other.message.content


# --- the controls ---------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_a_failing_vendor_answers_with_its_status_and_the_adapter_sees_it(
    adapters: tuple[OpenAIClient, AnthropicClient, TestClient], status: int
) -> None:
    openai, _, control = adapters
    control.put("/control/openai", json={"status": status})

    with pytest.raises(ProviderError) as raised:
        await openai.complete(ask())

    assert raised.value.status_code == status


async def test_each_vendor_is_controlled_separately(
    adapters: tuple[OpenAIClient, AnthropicClient, TestClient],
) -> None:
    openai, anthropic, control = adapters
    control.put("/control/openai", json={"status": 503})

    with pytest.raises(ProviderError):
        await openai.complete(ask())

    assert (await anthropic.complete(ask("claude-3-haiku-20240307"))).provider == "anthropic"


def test_a_delay_slows_the_reply(mock: TestClient) -> None:
    mock.put("/control/openai", json={"delay_ms": 120})

    started = time.monotonic()
    mock.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert time.monotonic() - started >= 0.12


def test_calls_are_counted_per_vendor_including_failed_ones(mock: TestClient) -> None:
    body = {"messages": [{"role": "user", "content": "hi"}]}
    mock.post("/v1/chat/completions", json=body)
    mock.put("/control/openai", json={"status": 503})
    mock.post("/v1/chat/completions", json=body)
    mock.post("/v1/messages", json=body)

    stats = mock.get("/stats").json()

    assert stats["openai"]["calls"] == 2 and stats["openai"]["status"] == 503
    assert stats["anthropic"]["calls"] == 1 and stats["anthropic"]["status"] == 200


def test_reset_returns_both_vendors_to_healthy_and_zeroes_the_counters(mock: TestClient) -> None:
    mock.put("/control/openai", json={"status": 503})
    mock.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    mock.post("/reset")

    stats = mock.get("/stats").json()
    assert stats["openai"] == {"calls": 0, "status": 200, "delay_ms": 0.0}


@pytest.mark.parametrize("status", [199, 600])
def test_a_status_that_is_not_http_is_refused(mock: TestClient, status: int) -> None:
    assert mock.put("/control/openai", json={"status": status}).status_code == 422


def test_an_unknown_vendor_is_refused(mock: TestClient) -> None:
    assert mock.put("/control/mistral", json={"status": 503}).status_code == 422


@pytest.mark.parametrize(
    "body", [{"messages": []}, {"nothing": 1}, {"messages": [{"role": "user"}]}]
)
def test_a_body_that_is_not_a_chat_request_is_a_400(
    mock: TestClient, body: dict[str, Any]
) -> None:
    assert mock.post("/v1/chat/completions", json=body).status_code == 400
