# Request path instrumentation: one span and one structured log line per request.
#
# Spans go to an in-memory exporter on a provider placed on app.state, so these
# tests never touch the global tracer provider, which OpenTelemetry refuses to
# replace once set, and never need a collector.

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from app.cache.semantic_cache import SemanticHit
from app.providers.stub import StubProvider

AUTH = {"Authorization": "Bearer test-key"}
MODEL = "gpt-4o-mini"
PROMPT = "What is the capital of France? My account number is 4421-8890."
LOG_FIELDS = {
    "request_id", "cache_status", "similarity", "provider", "model", "input_tokens",
    "output_tokens", "cost_usd", "breaker_state", "error_code", "latency_ms",
}


def body(content: str = PROMPT) -> dict[str, Any]:
    """A minimal chat completion request."""
    return {"model": MODEL, "messages": [{"role": "user", "content": content}]}


@pytest.fixture
def exporter(client: TestClient) -> Iterator[InMemorySpanExporter]:
    """Collect this test's spans in memory via a provider on app.state."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client.app.state.tracer_provider = provider  # type: ignore[attr-defined]
    yield exporter
    provider.shutdown()


def only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    """The single span one request should produce."""
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, [span.name for span in spans]
    return spans[0]


def log_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    """Every structured request log line captured so far, parsed."""
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "semcache.request"]


class FixedSemanticHit:
    """A semantic cache that always answers with a known score."""

    async def search(self, model: str, messages: Any) -> SemanticHit:
        return SemanticHit(
            response_body={
                "id": "chatcmpl-x", "created": 1, "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paris."}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            similarity=0.93,
            entry_id="e1",
        )

    async def add(self, *args: Any, **kwargs: Any) -> None:
        return None


# --- spans ----------------------------------------------------------------


def test_a_miss_produces_one_span_with_every_documented_attribute(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """PRD section 18.3, on a single span, named by the gen_ai convention."""
    client.post("/v1/chat/completions", json=body(), headers=AUTH)
    span = only_span(exporter)
    attrs = span.attributes or {}

    assert span.name == f"chat {MODEL}"
    assert attrs["gen_ai.system"] == "stub"
    assert attrs["gen_ai.request.model"] == MODEL
    assert attrs["gen_ai.usage.input_tokens"] > 0
    assert attrs["gen_ai.usage.output_tokens"] > 0
    assert attrs["semcache.cache_status"] == "miss"
    assert attrs["semcache.cost_usd"] == 0.0
    assert attrs["semcache.breaker_state"] == "not_configured"
    assert "semcache.similarity_score" not in attrs
    assert span.status.status_code == StatusCode.UNSET


def test_an_exact_hit_records_no_token_usage(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """A hit skipped the provider, so it spent no tokens and cost nothing."""
    client.post("/v1/chat/completions", json=body(), headers=AUTH)
    exporter.clear()
    client.post("/v1/chat/completions", json=body(), headers=AUTH)
    attrs = only_span(exporter).attributes or {}

    assert attrs["semcache.cache_status"] == "exact_hit"
    assert "gen_ai.usage.input_tokens" not in attrs
    assert attrs["semcache.cost_usd"] == 0.0


def test_a_semantic_hit_carries_its_similarity_score(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """The score is what makes a wrong semantic answer traceable to its decision."""
    client.app.state.semantic_cache = FixedSemanticHit()  # type: ignore[attr-defined]
    client.post("/v1/chat/completions", json=body("A reworded question"), headers=AUTH)
    attrs = only_span(exporter).attributes or {}

    assert attrs["semcache.cache_status"] == "semantic_hit"
    assert attrs["semcache.similarity_score"] == pytest.approx(0.93)


def test_a_failed_request_is_marked_as_an_error_with_its_code(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """Failures must be findable in a trace backend, and groupable by kind."""
    client.post("/v1/chat/completions", json=body())
    span = only_span(exporter)
    attrs = span.attributes or {}

    assert span.status.status_code == StatusCode.ERROR
    assert attrs["error.type"] == "MISSING_API_KEY"
    assert "semcache.cache_status" not in attrs


def test_requests_still_succeed_with_tracing_off(client: TestClient) -> None:
    """The default: no provider, no-op spans, and the request is unaffected."""
    client.app.state.tracer_provider = None  # type: ignore[attr-defined]
    assert client.post("/v1/chat/completions", json=body(), headers=AUTH).status_code == 200


# --- logs -----------------------------------------------------------------


def test_each_request_writes_exactly_one_log_line_with_every_field(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """PRD section 25. Every field is present, null where it does not apply."""
    with caplog.at_level("INFO", logger="semcache.request"):
        client.post("/v1/chat/completions", json=body(), headers=AUTH)
    lines = log_lines(caplog)

    assert len(lines) == 1
    assert set(lines[0]) == LOG_FIELDS
    assert lines[0]["cache_status"] == "miss"
    assert lines[0]["similarity"] is None
    assert lines[0]["latency_ms"] >= 0


def test_the_log_line_carries_the_request_id_the_caller_saw(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """So a caller's report can be matched to one log line."""
    with caplog.at_level("INFO", logger="semcache.request"):
        response = client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert log_lines(caplog)[0]["request_id"] == response.headers["x-request-id"]


@pytest.mark.parametrize(
    ("headers", "provider", "code"),
    [
        ({}, StubProvider(), "MISSING_API_KEY"),
        (AUTH, StubProvider(fail_times=1), "ALL_PROVIDERS_UNAVAILABLE"),
    ],
)
def test_a_failure_is_logged_with_its_error_code(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    headers: dict[str, str],
    provider: StubProvider,
    code: str,
) -> None:
    """At warning level, so failures stand out from routine traffic."""
    client.app.state.provider = provider  # type: ignore[attr-defined]
    with caplog.at_level("INFO", logger="semcache.request"):
        client.post("/v1/chat/completions", json=body(), headers=headers)
    record = next(r for r in caplog.records if r.name == "semcache.request")

    assert record.levelname == "WARNING"
    assert json.loads(record.getMessage())["error_code"] == code


def test_prompt_text_never_reaches_the_log_by_default(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Prompts are caller data. Checked against the raw captured text, not just
    the parsed fields, so no field anywhere can leak it."""
    with caplog.at_level("DEBUG"):
        client.post("/v1/chat/completions", json=body(), headers=AUTH)
        client.post("/v1/chat/completions", json=body(), headers=AUTH)

    assert "capital of France" not in caplog.text
    assert "4421-8890" not in caplog.text
    assert all("prompt" not in line for line in log_lines(caplog))


def test_prompt_text_is_logged_when_explicitly_switched_on(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """For development only, and only when someone asks for it."""
    state = client.app.state  # type: ignore[attr-defined]
    state.settings = state.settings.model_copy(update={"log_prompts": True})
    with caplog.at_level("INFO", logger="semcache.request"):
        client.post("/v1/chat/completions", json=body(), headers=AUTH)

    assert log_lines(caplog)[0]["prompt"] == [{"role": "user", "content": PROMPT}]
