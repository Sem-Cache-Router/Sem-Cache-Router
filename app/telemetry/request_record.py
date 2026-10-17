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
from app.telemetry.tracing import set_cache_outcome, set_gen_ai_attributes

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
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float = 0.0
    error_code: str | None = None
    latency_ms: float | None = None


def breaker_state_of(state: Any) -> str:
    """Read the breaker's state from app.state, or report that there is none."""
    breaker = getattr(state, "breaker", None)
    if breaker is None:
        return NO_BREAKER
    return str(breaker.state.value)


def _log_fields(
    outcome: RequestOutcome, messages: list[Message], log_prompts: bool
) -> dict[str, Any]:
    """The fields of the structured log line, prompt included only on request.

    Every documented field is always present, with null where it does not
    apply, so a log query never has to tell "absent" from "not recorded".
    """
    fields: dict[str, Any] = {
        "request_id": outcome.request_id,
        "cache_status": outcome.cache_status,
        "similarity": outcome.similarity,
        "provider": outcome.provider,
        "model": outcome.model,
        "input_tokens": outcome.input_tokens,
        "output_tokens": outcome.output_tokens,
        "cost_usd": outcome.cost_usd,
        "breaker_state": outcome.breaker_state,
        "error_code": outcome.error_code,
        "latency_ms": outcome.latency_ms,
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
    if outcome.error_code is not None:
        span.set_attribute(ERROR_TYPE, outcome.error_code)

    level = logging.WARNING if outcome.error_code else logging.INFO
    log.log(level, json.dumps(_log_fields(outcome, messages, log_prompts), sort_keys=True))
