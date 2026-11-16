# What one request did, recorded once.
#
# The pipeline in app/api/chat.py fills in a RequestOutcome as it reaches each
# exit: which tier answered, at what similarity, what the provider charged.
# record_request then turns that into span attributes and a single structured
# log line. Keeping the recording here, rather than as telemetry calls threaded
# through the handler, keeps the handler reading as a sequence of cache and
# provider decisions, and keeps it under the complexity ceiling.
#
# Prompt text is caller data. It reaches the log only when log_prompts is on,
# which is off by default and meant for development.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace

from app.models import Message
from app.telemetry.tracing import (
    SEMCACHE_ROUTED_MODEL,
    SEMCACHE_ROUTING_REASON,
    SEMCACHE_TIER2_STATUS,
    set_cache_outcome,
    set_gen_ai_attributes,
)

log = logging.getLogger("semcache.request")

# Reported when no circuit breaker is wired into the app. Not "closed": a span
# claiming a breaker state for a breaker that does not exist would be the same
# lie the health check refuses to tell about an unchecked dependency.
NO_BREAKER = "not_configured"

# The OpenTelemetry convention for classifying a failure on a span.
ERROR_TYPE = "error.type"


@dataclass
class RequestOutcome:
    """Everything worth recording about one request, filled in as it runs."""

    request_id: str
    provider: str
    model: str
    breaker_state: str = NO_BREAKER
    cache_status: str | None = None
    similarity: float | None = None
    tier2_status: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float = 0.0
    error_code: str | None = None
    latency_ms: float | None = None
    routed_model: str | None = None
    routing_reason: str | None = None


def breaker_state_of(state: Any) -> str:
    """The primary provider's breaker state, or report that there is none.

    Read before the call, so it is what the request started with. A request that
    fails over replaces it with the state of the provider that answered.
    """
    router = getattr(state, "router", None)
    if router is None:
        return NO_BREAKER
    return str(router.breaker_states()[router.primary.name])


def _trace_id_of(span: trace.Span) -> str | None:
    """Extract the trace ID as a 32-character hex string, or None for no-op spans.

    A no-op span (produced when tracing is disabled) has an invalid span
    context whose trace_id is zero. Returning None instead of the zero string
    keeps the log field honest: null means "no trace", not "trace 00000...".
    """
    ctx = span.get_span_context()
    if ctx is None or not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")


def _log_fields(
    span: trace.Span,
    outcome: RequestOutcome,
    messages: list[Message],
    log_prompts: bool,
) -> dict[str, Any]:
    """The fields of the structured log line, prompt included only on request.

    Every documented field is always present, with null where it does not
    apply, so a log query never has to tell "absent" from "not recorded".
    trace_id lets a log grep land directly on the matching Jaeger trace.
    """
    fields: dict[str, Any] = {
        "request_id": outcome.request_id,
        "trace_id": _trace_id_of(span),
        "cache_status": outcome.cache_status,
        "similarity": outcome.similarity,
        "tier2_status": outcome.tier2_status,
        "provider": outcome.provider,
        "model": outcome.model,
        "input_tokens": outcome.input_tokens,
        "output_tokens": outcome.output_tokens,
        "cost_usd": outcome.cost_usd,
        "breaker_state": outcome.breaker_state,
        "error_code": outcome.error_code,
        "latency_ms": outcome.latency_ms,
        "routed_model": outcome.routed_model,
        "routing_reason": outcome.routing_reason,
    }
    if log_prompts:
        fields["prompt"] = [message.model_dump() for message in messages]
    return fields


def record_request(
    span: trace.Span,
    outcome: RequestOutcome,
    *,
    messages: list[Message],
    log_prompts: bool,
) -> None:
    """Attach the outcome to the request's span and write one log line for it.

    A request that ended in an error has no cache status, so only the gen_ai
    attributes go on its span, plus the error code under error.type, which is
    what lets a trace backend group failures by kind. The span's status needs
    no setting here: OpenTelemetry marks it failed when the exception passes
    back out through it, and setting it as well was verified to change nothing.
    """
    set_gen_ai_attributes(
        span,
        system=outcome.provider,
        model=outcome.model,
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
    )
    if outcome.cache_status is not None:
        set_cache_outcome(
            span,
            cache_status=outcome.cache_status,
            similarity_score=outcome.similarity,
            cost_usd=outcome.cost_usd,
            breaker_state=outcome.breaker_state,
        )
    if outcome.tier2_status is not None:
        span.set_attribute(SEMCACHE_TIER2_STATUS, outcome.tier2_status)
    if outcome.error_code is not None:
        span.set_attribute(ERROR_TYPE, outcome.error_code)
    if outcome.routed_model is not None:
        span.set_attribute(SEMCACHE_ROUTED_MODEL, outcome.routed_model)
    if outcome.routing_reason is not None:
        span.set_attribute(SEMCACHE_ROUTING_REASON, outcome.routing_reason)

    level = logging.WARNING if outcome.error_code else logging.INFO
    log.log(level, json.dumps(_log_fields(span, outcome, messages, log_prompts), sort_keys=True))
