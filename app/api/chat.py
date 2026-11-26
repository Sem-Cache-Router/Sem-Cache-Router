# Chat completions.
#
# This module owns the ordering the whole design rests on:
#   1. validate the body against the chat completion schema
#   2. authenticate the caller, since the key is the budget identity
#   3. reserve the estimated token cost, or return 429 and stop
#   4. check Tier 1, and on a hit release the reservation in full
#   5. on a Tier 1 miss, embed and search Tier 2
#   6. on a Tier 2 miss, call the primary provider through its breaker
#   7. on provider failure, move to the secondary, in the same request
#   8. write both tiers with a TTL, reconcile the reservation, return
#
# Why the limiter runs before any cache lookup: a budget that depends on what
# happens to be cached is not a budget anyone can reason about. The same caller
# sending the same traffic would be admitted or refused depending on what an
# unrelated caller had cached a minute earlier. The accepted cost is that a
# caller at their limit is refused an answer that was free to serve.
#
# Steps 6 and 7 are one call, because failover is something the router does inside
# the call, not something the handler sequences: the handler asks for an answer and
# is told which provider gave it and whether it had to fail over to get it.
#
# Tier 2 is soft where Tier 1 is hard. A Tier 1 failure fails the request,
# because Redis also decides admission. A Tier 2 failure of any kind costs a
# cache hit and nothing else: the request carries on to the provider.
#
# A semantic hit is never copied into Tier 1. Doing so would later serve a
# paraphrase's answer as an exact hit with no similarity score, laundering an
# uncertain match into a certain one.

from __future__ import annotations

import hmac
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request, Response

from app.cache.semantic_cache import SemanticHit, Tier2Result
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
from app.providers.router import AllProvidersFailed
from app.router import choose_model
from app.telemetry.request_record import RequestOutcome, breaker_state_of, record_request
from app.telemetry.tracing import child_span, request_span

router = APIRouter(tags=["chat"])

log = logging.getLogger(__name__)


def _require_api_key(request: Request) -> str:
    """Return the caller's key, or refuse the request.

    The key is the identity a token budget is charged against. With an
    allowlist configured only those keys are accepted, compared in constant
    time, and any other is a 401 before the request does any work; without one
    any non-empty bearer token is accepted.
    """
    settings = request.app.state.settings
    header = request.headers.get("authorization", "")
    token = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""

    allowed = settings.api_key_allowlist
    # An allowlist always requires a key, whatever require_api_key says.
    if not token and (settings.require_api_key or allowed):
        raise GatewayError(
            ErrorCode.MISSING_API_KEY,
            "No caller API key supplied. Send an Authorization: Bearer header.",
            status_code=401,
        )
    if allowed and not any(
        hmac.compare_digest(token.encode(), key.encode()) for key in allowed
    ):
        raise GatewayError(
            ErrorCode.INVALID_API_KEY,
            "That API key is not recognised.",
            status_code=401,
        )
    return token or "anonymous"


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
    response: Response,
) -> ChatCompletionResponse:
    """Serve a completion from cache or a provider, enforcing the budget first.

    Opens one span for the request and records what happened on it and in the
    log, whatever the outcome, failures included. The work itself is in
    _serve, which fills in the outcome as it reaches each exit.
    """
    state = request.app.state
    rid = request_id(request)
    response.headers["x-request-id"] = rid
    outcome = RequestOutcome(
        request_id=rid,
        provider=state.provider.name,
        model=payload.model,
        breaker_state=breaker_state_of(state),
    )
    start = time.monotonic()
    tracer_provider = getattr(state, "tracer_provider", None)

    # Span named "{operation} {model}", as the gen_ai conventions specify.
    with request_span(f"chat {payload.model}", provider=tracer_provider) as span:
        try:
            return await _serve(payload, request, outcome, start)
        except GatewayError as exc:
            outcome.error_code = exc.code.value
            raise
        except Exception:
            outcome.error_code = "UNHANDLED"
            raise
        finally:
            outcome.latency_ms = round((time.monotonic() - start) * 1000, 3)
            record_request(
                span,
                outcome,
                messages=payload.messages,
                log_prompts=state.settings.log_prompts,
            )


async def _serve(
    payload: ChatCompletionRequest,
    request: Request,
    outcome: RequestOutcome,
    start: float,
) -> ChatCompletionResponse:
    """The request pipeline, steps 2 to 8, recording its outcome as it exits."""
    state = request.app.state
    metrics = state.metrics

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
            metrics.record_throttled()
            raise GatewayError(
                ErrorCode.TOKEN_BUDGET_EXCEEDED,
                "Token budget exhausted for this key.",
                status_code=429,
                retry_after_seconds=limiter.retry_after(api_key),
            )

    # Step 4. Tier 1.
    with child_span("cache tier1 lookup", state):
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
        await _release_for_hit(limiter, reservation)
        metrics.record_tier1_hit()
        metrics.record_latency(time.monotonic() - start, hit=True)
        outcome.cache_status = CacheStatus.EXACT_HIT.value
        return _from_cached(cached.response_body, payload.model, CacheStatus.EXACT_HIT)

    # Step 5. Tier 2, only after an exact miss, since embedding is the
    # expensive step and an exact repeat does not need it.
    with child_span("cache tier2 lookup", state):
        semantic, tier2_status = await _lookup_tier2(state, payload)
    outcome.tier2_status = tier2_status
    if tier2_status == "unavailable":
        metrics.record_tier2_unavailable()
    if semantic is not None:
        await _release_for_hit(limiter, reservation)
        metrics.record_tier2_hit()
        metrics.record_latency(time.monotonic() - start, hit=True)
        outcome.cache_status = CacheStatus.SEMANTIC_HIT.value
        outcome.similarity = semantic.similarity
        return _from_cached(
            semantic.response_body,
            payload.model,
            CacheStatus.SEMANTIC_HIT,
            similarity=semantic.similarity,
        )

    # Step 6. Model routing. The routing result controls which model the provider
    # is asked for; the cache key still uses the model the caller requested.
    routing = choose_model(payload.model, payload.messages, state.settings)
    outcome.routed_model = routing.model
    outcome.routing_reason = routing.reason
    if routing.reason == "routed_cheap":
        metrics.record_routed_cheap()
    else:
        metrics.record_routed_unchanged()

    # Steps 6 and 7. The provider call, through its breaker, with failover. The
    # child span covers the whole of it, so a failover shows in the trace as one
    # span that took two attempts rather than as unexplained latency.
    routed_payload = payload.model_copy(update={"model": routing.model})
    with child_span(f"provider call {routing.model}", state):
        provider_response = await _complete(state, routed_payload, outcome)

    # Step 8. Write through and reconcile.
    body = _to_body(provider_response, payload.model)

    try:
        entry = await state.cache.set(
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

    await _write_tier2(state, payload, entry.entry_id, body)

    if limiter is not None and reservation is not None:
        await limiter.reconcile(reservation, provider_response.usage.total_tokens)

    metrics.record_miss()
    metrics.add_spend(provider_response.cost_usd)
    metrics.record_latency(time.monotonic() - start, hit=False)
    outcome.cache_status = CacheStatus.MISS.value
    outcome.input_tokens = provider_response.usage.prompt_tokens
    outcome.output_tokens = provider_response.usage.completion_tokens
    outcome.cost_usd = provider_response.cost_usd
    return _from_cached(body, payload.model, CacheStatus.MISS)


async def _complete(
    state: Any, payload: ChatCompletionRequest, outcome: RequestOutcome
) -> ProviderResponse:
    """Get an answer from the first provider that can give one.

    Records on the outcome which provider actually answered and the state its
    breaker was in, since after a failover that is not the one the request
    started with, and counts the failover once per request.

    Two failures are told apart. A provider that rejects the request itself as
    malformed is reported to the caller as a 400: another provider would say the
    same, and it is the caller's to fix. Every provider failing is a 502, with the
    detail in the log rather than the response, since it names vendors and status
    codes the caller has no use for.
    """
    try:
        completion = await state.router.complete(payload)
    except ProviderError as exc:
        raise GatewayError(
            ErrorCode.INVALID_REQUEST,
            "The provider rejected the request as invalid.",
            status_code=400,
        ) from exc
    except AllProvidersFailed as exc:
        log.warning("no provider could answer: %s", exc)
        raise GatewayError(
            ErrorCode.ALL_PROVIDERS_UNAVAILABLE,
            "No provider could answer the request.",
            status_code=502,
        ) from exc

    outcome.provider = completion.provider
    outcome.breaker_state = completion.breaker_state
    if completion.failed_over:
        state.metrics.record_failover()
    return completion.response


async def _release_for_hit(limiter: Any, reservation: Any) -> None:
    """Return a reservation in full when a cache served the request.

    A hit from either tier spends no provider tokens, so nothing of the estimate
    was used. Shared by both hit paths so the rule lives in one place.
    """
    if limiter is not None and reservation is not None:
        await limiter.release(reservation)


async def _lookup_tier2(
    state: Any, payload: ChatCompletionRequest
) -> tuple[SemanticHit | None, str | None]:
    """Search the semantic cache; return the hit and a status string.

    Status is "disabled" when Tier 2 is not configured, "unavailable" when
    Chroma could not be reached during this request, or "ok" when the search
    completed normally (whether or not it was a hit). None is returned only
    when Tier 1 answered before Tier 2 was tried; that case never reaches here.

    The status is what allows the caller to record a Chroma outage as a
    separate counter rather than conflating it with a genuine cache miss.
    Tier 2 must never be the reason a request fails, so exceptions are caught
    and treated as "ok" (not an outage, some other transient problem).
    """
    semantic_cache = getattr(state, "semantic_cache", None)
    if semantic_cache is None:
        return None, "disabled"
    try:
        result: Tier2Result = await semantic_cache.search_with_status(
            payload.model, payload.messages
        )
    except Exception:
        log.warning("Tier 2 lookup failed; continuing to the provider", exc_info=True)
        return None, "ok"
    if result.unavailable:
        return None, "unavailable"
    return result.hit, "ok"


async def _write_tier2(
    state: Any, payload: ChatCompletionRequest, entry_id: str, body: dict[str, Any]
) -> None:
    """Index a fresh provider answer in the semantic cache, best effort.

    Uses Tier 1's entry id, so a later hit in either tier traces to the same
    origin request. A failure is logged and dropped: the caller already has a
    good answer and losing the index entry only costs a future hit.
    """
    semantic_cache = getattr(state, "semantic_cache", None)
    if semantic_cache is None:
        return
    try:
        await semantic_cache.add(payload.model, payload.messages, entry_id, body)
    except Exception:
        log.warning("Tier 2 write failed; the answer was still served", exc_info=True)


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


def _from_cached(
    body: dict[str, Any],
    model: str,
    status: CacheStatus,
    *,
    similarity: float | None = None,
) -> ChatCompletionResponse:
    """Assemble the response, stamping how it was produced.

    semcache_similarity is set only for a semantic hit. A score belongs only on
    a result that similarity actually decided; on an exact hit or a miss it
    would be misleading, so it stays null there.
    """
    return ChatCompletionResponse(
        id=body["id"],
        object=body.get("object", "chat.completion"),
        created=body["created"],
        model=body.get("model", model),
        choices=[Choice.model_validate(choice) for choice in body["choices"]],
        usage=Usage.model_validate(body["usage"]),
        semcache_status=status,
        semcache_similarity=similarity,
    )
