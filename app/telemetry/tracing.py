# OpenTelemetry span setup.
#
# Provides the machinery for distributed tracing.  The request path is
# instrumented in a separate PR; this module exposes only:
#   attribute name constants
#   a function to install the tracer provider and exporter
#   a context manager that opens a span for one request
#   a helper that attaches the cache outcome to a span
#
# Attribute names follow the OpenTelemetry gen_ai semantic conventions where
# a standard name exists, so spans are readable by tooling that already
# understands LLM telemetry.  Project-specific attributes live under the
# semcache.* namespace.
#
# Why tracing rather than logging alone: the interesting questions here are per
# request and comparative. Which requests hit, at what similarity, at what cost,
# in which breaker state. A log line answers that for one request; a trace
# answers it for a distribution.
#
# Prompt text is recorded only in a development configuration, since prompts are
# caller data.

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# ---------------------------------------------------------------------------
# Attribute name constants
#
# Standard gen_ai names come from the OpenTelemetry semantic conventions for
# generative AI.  Using these rather than inventing our own means a Jaeger or
# Grafana dashboard that already understands LLM spans renders ours correctly
# without custom configuration.
# ---------------------------------------------------------------------------

# gen_ai semantic conventions
GEN_AI_SYSTEM: str = "gen_ai.system"
GEN_AI_REQUEST_MODEL: str = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS: str = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: str = "gen_ai.usage.output_tokens"

# Project-specific attributes
SEMCACHE_CACHE_STATUS: str = "semcache.cache_status"
SEMCACHE_SIMILARITY_SCORE: str = "semcache.similarity_score"
SEMCACHE_BREAKER_STATE: str = "semcache.breaker_state"
SEMCACHE_COST_USD: str = "semcache.cost_usd"

_TRACER_NAME = "semcache-router"

log = logging.getLogger(__name__)


def install_tracing(*, endpoint: str | None, enabled: bool) -> TracerProvider | None:
    """Install the OpenTelemetry tracer provider and exporter.

    Called once from the lifespan.  Returns the provider so the lifespan can
    shut it down on exit and flush any pending spans.

    When tracing is disabled or no endpoint is configured, nothing is installed
    and the global NoOp tracer provider remains in place.  That means every
    call to ``request_span`` still works — it just produces no-op spans that
    carry no cost and export nothing.  This is the design that lets the test
    suite pass with no collector, no network, and no environment variable set.
    """
    if not enabled:
        return None
    if not endpoint:
        # Enabled with nowhere to send spans is a misconfiguration rather than an
        # outage, so it is said out loud. Tracing still stays off rather than
        # stopping the gateway, as the setting's own comment promises.
        log.warning(
            "tracing is enabled but otel_exporter_endpoint is not set; no spans will be exported"
        )
        return None

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


@contextmanager
def request_span(
    name: str,
    *,
    provider: TracerProvider | None = None,
) -> Iterator[trace.Span]:
    """Open a span for one gateway request, yielding it for attribute attachment.

    Works in both sync and async code — Python's ``with`` statement is not
    async-only.  When the global tracer provider is the NoOp (tracing off),
    this yields a no-op span whose ``set_attribute`` calls are silently ignored,
    so the instrumentation in the request path carries zero cost when tracing
    is disabled.  An optional provider can be passed for test isolation.
    """
    tracer = (
        provider.get_tracer(_TRACER_NAME)
        if provider is not None
        else trace.get_tracer(_TRACER_NAME)
    )
    with tracer.start_as_current_span(name) as span:
        yield span


def set_cache_outcome(
    span: trace.Span,
    *,
    cache_status: str,
    similarity_score: float | None = None,
    cost_usd: float = 0.0,
    breaker_state: str = "closed",
) -> None:
    """Attach the cache decision attributes to a span.

    Separated from the recording calls in the handler so the handler reads as
    a sequence of domain decisions rather than a sequence of telemetry calls.
    similarity_score is set only when Tier 2 decided the outcome; omitting it
    on an exact hit or a miss avoids a misleading zero.
    """
    span.set_attribute(SEMCACHE_CACHE_STATUS, cache_status)
    if similarity_score is not None:
        span.set_attribute(SEMCACHE_SIMILARITY_SCORE, similarity_score)
    span.set_attribute(SEMCACHE_COST_USD, cost_usd)
    span.set_attribute(SEMCACHE_BREAKER_STATE, breaker_state)


def set_gen_ai_attributes(
    span: trace.Span,
    *,
    system: str,
    model: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Attach the standard gen_ai attributes to a span.

    Token counts are optional because they are not known on a cache hit that
    skips the provider entirely.  When present they come from the provider's
    reported usage, not from an estimate.
    """
    span.set_attribute(GEN_AI_SYSTEM, system)
    span.set_attribute(GEN_AI_REQUEST_MODEL, model)
    if input_tokens is not None:
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    if output_tokens is not None:
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
