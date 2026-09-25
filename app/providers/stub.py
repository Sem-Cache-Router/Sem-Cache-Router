# A provider that answers without a network or a credential.
#
# Ships in app/ rather than tests/ deliberately. The lifespan selects a provider
# by settings.primary_provider, so making "stub" selectable means three things:
# the gateway starts and serves with no credentials, which is what makes a local
# demo and the test suite runnable offline; the dependency wiring is exercised
# for real rather than bypassed by a fixture; and the load test later has
# something to hammer without spending money. The cost is one non-production
# module in the image, which is a fair trade for all three.

from __future__ import annotations

import time
import uuid

from app.models import ChatCompletionRequest, Choice, Message, ProviderResponse, Usage
from app.providers.base import LLMProvider, ProviderError


class StubProvider(LLMProvider):
    """Returns a deterministic canned answer, and fails on demand.

    Deterministic because the cache tests assert that a second request returns
    the same content as the first. If the answer varied, an exact hit would be
    indistinguishable from a second provider call.
    """

    name = "stub"

    def __init__(
        self,
        *,
        fail_times: int = 0,
        status_code: int | None = 503,
        latency_ms: float = 0.0,
    ) -> None:
        """Configure how many calls fail, with what status, and how slowly it answers.

        fail_times drives the 502 path now and the breaker tests later.
        status_code is what the breaker classifies on, so a test can distinguish
        a 400, which must not count as a failure, from a 503, which must.
        """
        self._fail_times = fail_times
        self._status_code = status_code
        self._latency_ms = latency_ms
        self.call_count = 0

    async def complete(self, request: ChatCompletionRequest) -> ProviderResponse:
        """Return a canned answer, or raise for the first fail_times calls.

        Counts every call including the failures, so a test can assert that a
        cache hit did not reach the provider at all.
        """
        self.call_count += 1

        if self.call_count <= self._fail_times:
            raise ProviderError(
                f"stub failure {self.call_count} of {self._fail_times}",
                provider=self.name,
                status_code=self._status_code,
            )

        prompt = request.messages[-1].content
        content = f"Stub answer to: {prompt}"

        prompt_tokens = self.estimate_tokens(request.messages)
        completion_tokens = max(1, len(content) // 4)

        return ProviderResponse(
            provider=self.name,
            model=request.model,
            message=Message(role="assistant", content=content),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            cost_usd=0.0,
            finish_reason="stop",
            latency_ms=self._latency_ms,
        )

    def estimate_tokens(self, messages: list[Message]) -> int:
        """Approximate four characters per token.

        A placeholder, not an estimator. tiktoken replaces this when the rate
        limiter lands, and until then nothing depends on the number being
        accurate, only on it being deterministic.
        """
        characters = sum(len(message.content) for message in messages)
        return max(1, characters // 4)

    def price_of(self, usage: Usage) -> float:
        """Zero. The stub makes no network call, so it genuinely costs nothing.

        Not a placeholder value: a benchmark run against the stub should report
        zero spend, because zero is the true figure.
        """
        return 0.0

    def to_choices(self, response: ProviderResponse) -> list[Choice]:
        """Wrap a provider reply in the OpenAI choices list.

        Lives here rather than in the route so the route does not need to know
        how any one provider shapes its reply.
        """
        return [Choice(index=0, message=response.message, finish_reason=response.finish_reason)]

    @staticmethod
    def new_response_id() -> str:
        """Generate an OpenAI style completion id."""
        return f"chatcmpl-{uuid.uuid4().hex[:24]}"

    @staticmethod
    def now_unix() -> int:
        """Current time as the OpenAI created field expects it."""
        return int(time.time())
