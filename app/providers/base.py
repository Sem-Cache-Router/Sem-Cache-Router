# The provider interface.
#
# Why a common interface: it makes adding a second provider a configuration
# change rather than new code, and it keeps the breaker from being coupled to
# one vendor SDK.
#
# Why price comes from a committed table rather than a constant: a provider
# price change would otherwise silently invalidate every recorded benchmark
# result.

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import ChatCompletionRequest, Message, ProviderResponse, Usage


class ProviderError(Exception):
    """A provider call failed in a way the caller cannot fix.

    Carries the upstream status code where there was one, because the circuit
    breaker classifies failures by it: timeouts, 5xx and provider 429s count
    toward tripping, while a 400 does not, since a malformed request is a caller
    defect and counting it would let one bad client trip failover for everybody.

    status_code is None for failures that never reached the provider, such as a
    connection error or a timeout, which do count.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMProvider(ABC):
    """A provider the gateway can call. OpenAIClient and AnthropicClient implement this.

    The interface is the deliverable, not an implementation detail. All three
    methods are defined now, before any of them is needed, because a two method
    interface that later grows a third is an interface change, and avoiding
    interface changes is the entire reason this abstraction exists.
    """

    name: str

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> ProviderResponse:
        """Issue a completion and return it in the normalised shape.

        Async because every real implementation is network bound. Defining it
        sync now, because a stub does not need to await anything, would force a
        rewrite of the whole call chain when the first real adapter lands.

        Raises ProviderError when the call fails.
        """

    @abstractmethod
    def estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate prompt cost before the call, since admission cannot wait for the reply.

        The rate limiter has to decide whether to admit a request before the
        response, and therefore its true cost, exists. The estimate is reserved
        at admission and reconciled against reported usage afterwards.
        """

    @abstractmethod
    def price_of(self, usage: Usage) -> float:
        """Convert reported usage to dollars using a committed price table.

        A table rather than a constant, so that a provider price change does not
        silently invalidate results recorded before it.
        """
