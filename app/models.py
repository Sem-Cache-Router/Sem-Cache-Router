# Wire and internal schemas.
#
# Why the OpenAI shape: the gateway earns nothing by inventing its own
# contract, and matching the existing one means a caller changes a base URL and
# nothing else. The two extra fields are additive, so a strict client that
# ignores unknown keys still parses the body.

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CacheStatus(StrEnum):
    """How a response was produced.

    Recorded on every response and every trace span, so a wrong answer can be
    traced back to the decision that produced it. SEMANTIC_HIT is defined now
    although nothing emits it yet, because it is part of the wire contract and
    a client may already branch on it.
    """

    EXACT_HIT = "exact_hit"
    SEMANTIC_HIT = "semantic_hit"
    MISS = "miss"


class ErrorCode(StrEnum):
    """The closed set of error codes the gateway emits.

    An enum rather than string literals at each raise site, so a typo is a
    startup failure rather than an error body no client can match on.
    """

    INVALID_REQUEST = "INVALID_REQUEST"
    MISSING_API_KEY = "MISSING_API_KEY"
    INVALID_API_KEY = "INVALID_API_KEY"
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"
    TOKEN_BUDGET_EXCEEDED = "TOKEN_BUDGET_EXCEEDED"
    ALL_PROVIDERS_UNAVAILABLE = "ALL_PROVIDERS_UNAVAILABLE"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"


class Message(BaseModel):
    """One chat message, in the OpenAI role and content shape."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """An inbound chat completion request, validated before anything else runs."""

    # extra="ignore", deliberately, not "forbid". Real OpenAI clients send n,
    # user, stream_options, response_format and tool_choice among others.
    # Rejecting them would break the one claim the gateway makes, that adoption
    # costs a base URL change and nothing else. Unmodelled fields are ignored
    # rather than honoured, which is a narrower promise but an honest one.
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float = 1.0
    max_tokens: int | None = None
    stream: bool = False


class Usage(BaseModel):
    """Token counts as reported by the provider, the basis for cost and reconciliation."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """One completion choice, mirroring the OpenAI response shape."""

    index: int = 0
    message: Message
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    """The OpenAI response shape plus two additive fields."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

    # The additive pair. semcache_similarity stays null on an exact hit and on
    # a miss; it carries a score only when Tier 2 decided the outcome, because
    # a similarity number on a result that similarity did not produce would be
    # misleading.
    semcache_status: CacheStatus
    semcache_similarity: float | None = None


class ErrorDetail(BaseModel):
    """The body of an error, carrying a retry hint so a caller can back off correctly."""

    code: ErrorCode
    message: str
    retry_after_seconds: int | None = None
    request_id: str


class ErrorResponse(BaseModel):
    """The error envelope.

    Nested under an "error" key to match the wire format in PRD section 19.4,
    which is also the shape OpenAI clients already parse. The flat alternative
    would collide with a successful response body on the field name "message".
    """

    error: ErrorDetail


class CacheEntry(BaseModel):
    """One logical cached response, written to both tiers under a shared identifier.

    Carries only the fields that can be populated today. The embedding lives in
    ChromaDB alone and is added with Tier 2. cost_usd is present but zero while
    the stub provider is in use, because the stub genuinely costs nothing.
    """

    entry_id: str
    prompt_hash: str
    prompt_text: str
    response_body: dict[str, Any]
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float = 0.0
    created_at: datetime
    expires_at: datetime

    # Updated on every hit. Deliberately separate from the response body so the
    # counter can be incremented without rewriting the entry, which is what
    # keeps the remaining TTL untouched.
    hit_count: int = 0
    last_hit_at: datetime | None = None


class ProviderResponse(BaseModel):
    """A provider reply normalised across vendors.

    Normalising here is the whole point of the adapter layer: OpenAI and
    Anthropic report usage in different shapes, and absorbing that difference
    at the edge leaves cost accounting with one code path rather than one per
    vendor.
    """

    provider: str
    model: str
    message: Message
    usage: Usage
    cost_usd: float = 0.0
    finish_reason: str | None = "stop"
    latency_ms: float = 0.0
