from __future__ import annotations

import time
from typing import Any

import httpx

from app.config import get_settings
from app.models import ChatCompletionRequest, Message, ProviderResponse, Usage
from app.providers.base import LLMProvider, ProviderError


class AnthropicClient(LLMProvider):
    name = "anthropic"
    base_url = "https://api.anthropic.com/v1"

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self.http_client = http_client
        self.settings = get_settings()
        if not self.settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.api_key: str = self.settings.anthropic_api_key

        self.pricing = {
            "claude-3-haiku-20240307": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
            "claude-3-5-sonnet-20240620": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        }

    async def complete(self, request: ChatCompletionRequest) -> ProviderResponse:
        start_time = time.monotonic()

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        system = None
        anthropic_messages: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                if system is None:
                    system = msg.content
                else:
                    system += "\n" + msg.content
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": anthropic_messages,
            "max_tokens": request.max_tokens if request.max_tokens is not None else 1024,
            "temperature": request.temperature,
        }
        if system is not None:
            payload["system"] = system

        try:
            response = await self.http_client.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            raise ProviderError(
                f"Anthropic HTTP error: {e.response.text}",
                provider=self.name,
                status_code=status_code,
            ) from e
        except httpx.RequestError as e:
            raise ProviderError(
                f"Anthropic Network error: {str(e)}",
                provider=self.name,
                status_code=None,
            ) from e

        latency_ms = (time.monotonic() - start_time) * 1000.0

        text_content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_content += block.get("text", "")

        raw_usage = data.get("usage", {})
        input_tokens = raw_usage.get("input_tokens", 0)
        output_tokens = raw_usage.get("output_tokens", 0)

        provider_usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )

        cost = self.price_of(provider_usage, model=request.model)

        finish_reason = data.get("stop_reason")
        if finish_reason == "end_turn":
            finish_reason = "stop"

        return ProviderResponse(
            provider=self.name,
            model=request.model,
            message=Message(role="assistant", content=text_content),
            usage=provider_usage,
            cost_usd=cost,
            finish_reason=finish_reason,
            latency_ms=latency_ms
        )

    def estimate_tokens(self, messages: list[Message]) -> int:
        characters = sum(len(message.content) for message in messages)
        return max(1, characters // 4)

    def price_of(self, usage: Usage, model: str = "claude-3-haiku-20240307") -> float:
        rates = self.pricing.get(model, self.pricing["claude-3-haiku-20240307"])
        return (usage.prompt_tokens * rates["input"]) + (usage.completion_tokens * rates["output"])
