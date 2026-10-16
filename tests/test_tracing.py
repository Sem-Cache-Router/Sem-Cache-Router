# Tests for the tracing layer.
#
# OTel SDK 1.34 prevents overriding the global tracer provider once set, so
# tests create their own TracerProvider with an in-memory exporter and pass it
# to request_span directly rather than going through the global API.

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.telemetry.tracing import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    SEMCACHE_BREAKER_STATE,
    SEMCACHE_CACHE_STATUS,
    SEMCACHE_COST_USD,
    SEMCACHE_SIMILARITY_SCORE,
    install_tracing,
    request_span,
    set_cache_outcome,
    set_gen_ai_attributes,
)


@pytest.fixture
def traced() -> tuple[InMemorySpanExporter, TracerProvider]:
    """Create a TracerProvider with an in-memory exporter for one test.

    SimpleSpanProcessor so spans are exported synchronously and assertions
    can run immediately without flushing.  The provider is not set globally —
    it is passed to request_span via the provider parameter so tests don't
    fight over the global singleton.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter, provider
    provider.shutdown()


# ---------------------------------------------------------------------------
# Attribute constant tests
# ---------------------------------------------------------------------------


def test_gen_ai_attribute_names_follow_convention() -> None:
    """Standard gen_ai attribute names use the documented dotted namespace."""
    assert GEN_AI_SYSTEM == "gen_ai.system"
    assert GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
    assert GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
    assert GEN_AI_USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"


def test_semcache_attribute_names() -> None:
    """Project-specific attributes live under the semcache.* namespace."""
    assert SEMCACHE_CACHE_STATUS == "semcache.cache_status"
    assert SEMCACHE_SIMILARITY_SCORE == "semcache.similarity_score"
    assert SEMCACHE_BREAKER_STATE == "semcache.breaker_state"
    assert SEMCACHE_COST_USD == "semcache.cost_usd"


# ---------------------------------------------------------------------------
# request_span tests
# ---------------------------------------------------------------------------


def test_request_span_creates_named_span(
    traced: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    """request_span produces a finished span with the given name."""
    exp, provider = traced
    with request_span("test_operation", provider=provider):
        pass
    spans = exp.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_operation"


def test_request_span_yields_live_span(
    traced: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    """The yielded span accepts attributes set during the with block."""
    exp, provider = traced
    with request_span("attr_test", provider=provider) as span:
        span.set_attribute("test.key", "test_value")
    spans = exp.get_finished_spans()
    attrs = dict(spans[0].attributes)
    assert attrs.get("test.key") == "test_value"


# ---------------------------------------------------------------------------
# set_cache_outcome tests
# ---------------------------------------------------------------------------


def test_set_cache_outcome_attaches_all_fields(
    traced: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    """set_cache_outcome writes cache_status, cost_usd and breaker_state."""
    exp, provider = traced
    with request_span("outcome_test", provider=provider) as span:
        set_cache_outcome(
            span,
            cache_status="miss",
            cost_usd=0.003,
            breaker_state="closed",
        )
    spans = exp.get_finished_spans()
    attrs = dict(spans[0].attributes)
    assert attrs[SEMCACHE_CACHE_STATUS] == "miss"
    assert attrs[SEMCACHE_COST_USD] == 0.003
    assert attrs[SEMCACHE_BREAKER_STATE] == "closed"


def test_similarity_score_omitted_when_none(
    traced: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    """When similarity_score is None, the attribute is not set on the span."""
    exp, provider = traced
    with request_span("no_sim", provider=provider) as span:
        set_cache_outcome(span, cache_status="exact_hit")
    attrs = dict(exp.get_finished_spans()[0].attributes)
    assert SEMCACHE_SIMILARITY_SCORE not in attrs


def test_similarity_score_present_when_given(
    traced: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    """When similarity_score is provided, the attribute appears on the span."""
    exp, provider = traced
    with request_span("with_sim", provider=provider) as span:
        set_cache_outcome(span, cache_status="semantic_hit", similarity_score=0.94)
    attrs = dict(exp.get_finished_spans()[0].attributes)
    assert attrs[SEMCACHE_SIMILARITY_SCORE] == 0.94


# ---------------------------------------------------------------------------
# set_gen_ai_attributes tests
# ---------------------------------------------------------------------------


def test_gen_ai_attributes_attached(
    traced: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    """set_gen_ai_attributes writes system, model, and token counts."""
    exp, provider = traced
    with request_span("gen_ai_test", provider=provider) as span:
        set_gen_ai_attributes(
            span,
            system="openai",
            model="gpt-4o-mini",
            input_tokens=14,
            output_tokens=3,
        )
    attrs = dict(exp.get_finished_spans()[0].attributes)
    assert attrs[GEN_AI_SYSTEM] == "openai"
    assert attrs[GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 14
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 3


def test_token_counts_omitted_when_none(
    traced: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    """Token counts are optional — omitting them does not error."""
    exp, provider = traced
    with request_span("no_tokens", provider=provider) as span:
        set_gen_ai_attributes(span, system="stub", model="stub-model")
    attrs = dict(exp.get_finished_spans()[0].attributes)
    assert attrs[GEN_AI_SYSTEM] == "stub"
    assert GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert GEN_AI_USAGE_OUTPUT_TOKENS not in attrs


# ---------------------------------------------------------------------------
# Full attribute round-trip: every documented attribute on one span
# ---------------------------------------------------------------------------


def test_full_span_carries_all_documented_attributes(
    traced: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    """One span carries every attribute from PRD section 18.3."""
    exp, provider = traced
    with request_span("chat_completion", provider=provider) as span:
        set_gen_ai_attributes(
            span,
            system="openai",
            model="gpt-4o-mini",
            input_tokens=14,
            output_tokens=3,
        )
        set_cache_outcome(
            span,
            cache_status="miss",
            cost_usd=0.001,
            breaker_state="closed",
        )
    attrs = dict(exp.get_finished_spans()[0].attributes)
    expected_keys = {
        GEN_AI_SYSTEM,
        GEN_AI_REQUEST_MODEL,
        GEN_AI_USAGE_INPUT_TOKENS,
        GEN_AI_USAGE_OUTPUT_TOKENS,
        SEMCACHE_CACHE_STATUS,
        SEMCACHE_COST_USD,
        SEMCACHE_BREAKER_STATE,
    }
    assert expected_keys.issubset(set(attrs.keys()))


# ---------------------------------------------------------------------------
# install_tracing tests
# ---------------------------------------------------------------------------


def test_install_tracing_disabled_returns_none() -> None:
    """With tracing disabled, install_tracing returns None and leaves the NoOp."""
    result = install_tracing(endpoint="http://localhost:4318", enabled=False)
    assert result is None


def test_install_tracing_no_endpoint_returns_none() -> None:
    """With no endpoint, install_tracing returns None even if enabled."""
    result = install_tracing(endpoint=None, enabled=True)
    assert result is None


def test_install_tracing_enabled_returns_provider() -> None:
    """With tracing enabled and an endpoint, a TracerProvider is returned."""
    provider = install_tracing(endpoint="http://localhost:4318", enabled=True)
    assert provider is not None
    assert isinstance(provider, TracerProvider)
    # Clean up — shut down so the BatchSpanProcessor thread exits.
    provider.shutdown()


# ---------------------------------------------------------------------------
# NoOp behaviour when tracing is off
# ---------------------------------------------------------------------------


def test_request_span_works_without_provider() -> None:
    """With no explicit provider, request_span uses the global (NoOp) and does not error."""
    with request_span("noop_test") as span:
        set_cache_outcome(span, cache_status="miss", cost_usd=0.0)
        set_gen_ai_attributes(span, system="stub", model="stub-model")
    # No assertion on span content — the point is that it did not raise.
