# Chat completions.
#
# This module owns the ordering the whole design rests on:
#   1. validate the body against the chat completion schema
#   2. authenticate the caller, since the key is the budget identity
#   3. reserve the estimated token cost, or return 429 and stop
#   4. check Tier 1, and on a hit release the reservation in full
#   5. on a Tier 1 miss, embed and search Tier 2            (not built)
#   6. on a Tier 2 miss, route a model and call the provider through the breaker
#   7. on provider failure, let the breaker move to the secondary   (not built)
#   8. write both tiers with a TTL, reconcile the reservation, return
#
# Why the limiter runs before any cache lookup: a budget that depends on what
# happens to be cached is not a budget anyone can reason about. The same caller
# sending the same traffic would be admitted or refused depending on what an
# unrelated caller had cached a minute earlier. The accepted cost is that a
# caller at their limit is refused an answer that was free to serve.
#
# Steps 5 and 7 are marked in place below rather than omitted, so that the
# ordering is legible now and the later work has an obvious seam to land in.

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Request, Response

from app.errors import GatewayError, request_id
from app.models import (
    CacheStatus,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ErrorCode,
    ProviderResponse,
    Usage,
)
from app.providers.base import ProviderError

router = APIRouter(tags=["chat"])


def _require_api_key(request: Request) -> str:
    """Return the caller's key, or refuse the request.

    The key is the identity a token budget is charged against, so it cannot be
    optional once the limiter exists. Until then there is nothing to authorise
    against, and any non-empty bearer token is accepted rather than checked
    against a key store that does not exist yet. That is a deliberate gap, not
    an oversight, and it closes when the limiter lands.
    """
    settings = request.app.state.settings
    header = request.headers.get("authorization", "")
    token = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""

    if settings.require_api_key and not token:
        raise GatewayError(
            ErrorCode.MISSING_API_KEY,
            "No caller API key supplied. Send an Authorization: Bearer header.",
            status_code=401,
        )
    return token or "anonymous"


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
    response: Response,
) -> ChatCompletionResponse:
    """Serve a completion from cache or a provider, enforcing the budget first."""
    state = request.app.state
    rid = request_id(request)
    response.headers["x-request-id"] = rid

    # Step 2. Authentication. Resolved once; the key is also the bucket
    # identity the limiter will charge against.
    api_key = _require_api_key(request)

    # Step 3. Admission control. Guarded rather than absent, so the position of
    # the check is fixed even though the implementation is not.
    limiter = getattr(state, "limiter", None)
    reservation = None
    if limiter is not None:
        estimate = state.provider.estimate_tokens(payload.messages)
        reservation = await limiter.reserve(api_key, estimate)
        if reservation is None:
            raise GatewayError(
                ErrorCode.TOKEN_BUDGET_EXCEEDED,
                "Token budget exhausted for this key.",
                status_code=429,
                retry_after_seconds=limiter.retry_after(api_key),
            )

    # Step 4. Tier 1.
    try:
        cached = await state.cache.get(payload.model, payload.messages)
    except Exception as exc:
        # Fail closed. Admission is a no-op today, so a pass through would be
        # possible, but the ordering contract is the invariant being protected:
        # a gateway whose failure behaviour changes once the limiter ships is
        # one nobody can reason about.
        raise GatewayError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "Cache is unreachable, so admission cannot be decided.",
            status_code=503,
        ) from exc

    if cached is not None:
        if limiter is not None and reservation is not None:
            # A hit spends no provider tokens, so the whole reservation returns.
            await limiter.release(reservation)
        return _from_cached(cached.response_body, payload.model, CacheStatus.EXACT_HIT)

    # Step 5. Tier 2 semantic lookup lands here, between the exact miss and the
    # provider call. Not implemented.

    # Step 6. Provider call. The circuit breaker wraps this once failover is
    # wired, which is step 7; today a failure is terminal.
    try:
        provider_response = await state.provider.complete(payload)
    except ProviderError as exc:
        raise GatewayError(
            ErrorCode.ALL_PROVIDERS_UNAVAILABLE,
            f"Provider {exc.provider} failed and there is no configured failover.",
            status_code=502,
        ) from exc

    # Step 8. Write through and reconcile.
    body = _to_body(provider_response, payload.model)

    try:
        await state.cache.set(
            payload.model,
            payload.messages,
            body,
            prompt_tokens=provider_response.usage.prompt_tokens,
            completion_tokens=provider_response.usage.completion_tokens,
            cost_usd=provider_response.cost_usd,
        )
    except Exception as exc:
        raise GatewayError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "Cache is unreachable, so the response could not be stored.",
            status_code=503,
        ) from exc

    if limiter is not None and reservation is not None:
        await limiter.reconcile(reservation, provider_response.usage.total_tokens)

    return _from_cached(body, payload.model, CacheStatus.MISS)


def _to_body(provider_response: ProviderResponse, model: str) -> dict[str, Any]:
    """Render a provider reply into the OpenAI response body that gets cached.

    The cached value is the wire body rather than the provider object, so a hit
    replays exactly what a miss returned. Storing the richer internal shape
    would let the two drift.
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            Choice(
                index=0,
                message=provider_response.message,
                finish_reason=provider_response.finish_reason,
            ).model_dump()
        ],
        "usage": provider_response.usage.model_dump(),
    }


def _from_cached(body: dict[str, Any], model: str, status: CacheStatus) -> ChatCompletionResponse:
    """Assemble the response, stamping how it was produced.

    semcache_similarity stays null here for both outcomes this route can
    produce. A score belongs only on a result that similarity actually decided,
    and putting one on an exact hit would be misleading.
    """
    return ChatCompletionResponse(
        id=body["id"],
        object=body.get("object", "chat.completion"),
        created=body["created"],
        model=body.get("model", model),
        choices=[Choice.model_validate(choice) for choice in body["choices"]],
        usage=Usage.model_validate(body["usage"]),
        semcache_status=status,
        semcache_similarity=None,
    )
