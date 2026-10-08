import httpx
import pytest
from httpx import Response

from app.models import ChatCompletionRequest, Message
from app.providers.anthropic_client import AnthropicClient
from app.providers.base import ProviderError


class MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, response: Response):
        self._response = response

    async def handle_async_request(self, request: httpx.Request) -> Response:
        self._response.request = request
        return self._response


@pytest.mark.asyncio
async def test_anthropic_client_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SEMCACHE_ANTHROPIC_API_KEY", "test_key")

    mock_response = httpx.Response(
        200,
        json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello, world!"}],
            "model": "claude-3-haiku-20240307",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
    )

    client = httpx.AsyncClient(transport=MockTransport(mock_response))
    adapter = AnthropicClient(http_client=client)

    req = ChatCompletionRequest(
        model="claude-3-haiku-20240307",
        messages=[
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hi!")
        ]
    )

    res = await adapter.complete(req)
    assert res.provider == "anthropic"
    assert res.message.content == "Hello, world!"
    assert res.usage.prompt_tokens == 10
    assert res.usage.completion_tokens == 5
    assert res.finish_reason == "stop"


@pytest.mark.asyncio
async def test_anthropic_client_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SEMCACHE_ANTHROPIC_API_KEY", "test_key")

    mock_response = httpx.Response(
        400,
        json={"error": {"type": "invalid_request_error", "message": "Bad request"}},
    )

    client = httpx.AsyncClient(transport=MockTransport(mock_response))
    adapter = AnthropicClient(http_client=client)

    req = ChatCompletionRequest(
        model="claude-3-haiku-20240307",
        messages=[Message(role="user", content="Hi!")]
    )

    with pytest.raises(ProviderError) as exc:
        await adapter.complete(req)

    assert exc.value.status_code == 400
    assert "Anthropic HTTP error" in str(exc.value)
